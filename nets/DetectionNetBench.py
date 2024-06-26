import math
import time
from typing import Dict, Tuple, List, Optional, Callable, Union
import copy
import warnings

import torch
from torch import nn
import torch.backends.cudnn
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchmetrics.detection import MeanAveragePrecision

from config import DefaultTrainingConfig
from models.BoundingBox import BoundingBoxDataset, BoundingBox
from project_types import BoundBoxDataset_Item, Bbox_dict_for_nn_t,\
    Losses_dict_t, Checkpoint_t, Infer_freqs_t

class DetectionNetBench():
    """
    This class may host a Neural Network for detection. In order to achieve this
    a class may extend this and assign (or update) the default values for
    model, optimizer, scheduler, profiler.
    Its purpose is to wrap the NN so it is easier for higher level modules
    to use it.

    This is possible because, the two NN we currently use (FasterRCNN, SSD)
    both have a common interface as to how the input must be provided, 
    when in training or evaluation mode, as well as how the loss functions
    are returned.
    We exploit this common interface to:
        1. Handle moving the data to a device, if available.
        2. Create a training function that will also evaluate a test set
           before proceeding to the next epoch and preserving the new weights.
        3. Create a custom collate function for data in batches, since the
           default one does not provide the correct format.
        4. Extract a single bounding box out of the prediction returned by the
           network.
        5. Implement a checkpoint dictionary that can be saved and loaded,
           given only the path.
        6. Provide a visualization function for the evaluation.
        7. Caclulate the inference frequency of the model

    WARNING: The default optimizer has a weight_decay!
    If you remove this weight_decay, the network params will only be updated
    if the running validation loss is less than the overall minimum loss the
    network has ever achieved.
    """
    def __init__(self,
                 model_id: str,
                 model: Optional[nn.Module] = None,
                 model_path: Optional[str] = None,
                 config: Union[DefaultTrainingConfig, str] = DefaultTrainingConfig(),
                 checkpoint_path: Optional[str] = None,
                 load_only_weights: bool = False
            ) -> None:
        """
        Initializes the DetectionNetBench.
        - If you want to train the model, the following arguments should be
        provided:
            1. model_id
            2. model or model_path
            3. config
            4. And config should provide valid:
            root_train_dir, root_test_dir, json_train_labels, json_test_labels
        - If only used for evaluation, you need to provide:
            1. model_id
            2. model or model_path
            3. checkpoint_path
        - If you want to visualize a test image, or get the inference frequency
        of the network you need only to specify:
            1. model_id
            2. model or model_path
            3. checkpoint
            4. And config should provide valid:
            root_test_dir and json_test_labels

        Args:
        - model_id: A string to use when profiling or plotting losses
        - model: The detection model you want to wrap or None, when you
        provide the model via model_path argument.
        - model_path: Path to the entire model you have saved, or None if
        model argument is used.
        - config: The training configuration that will be used, packaged into
        a DefaultTrainingConfig object.
        - load_only_weights: Used only when checkpoint_path is provided.
        If True it only loads the model.state_dict to the model (i.e. weights
        and biases) and all other information like losses, epochs, optimizer.state_dict
        are ignored.

        Raises:
        If both model and model_path are None.
        If both model and model_path are not None.
        """
        if model is None and model_path is None:
            raise Exception("One the arguments: model, model_path should not be none")
        if model is not None and model_path is not None:
            raise Exception("Only one of the arguments model, model_path is allowed to be None")
        
        if isinstance(config, str):
            temp_config = DefaultTrainingConfig()
            temp_config.load(config)
            config = temp_config

        # Decide what methods can be executed based on the arguments provided
        if (config.root_train_dir and config.json_train_labels and
            config.root_test_dir and config.json_test_labels
        ):
            self.can_train = True
        if config.root_test_dir and config.json_test_labels:
            self.can_test = True

        # Organize the arguments into dictionaries
        self.root_dirs = {
            "train": config.root_train_dir, 
            "val": config.root_test_dir
        }
        self.json_labels = {
            "train": config.json_train_labels, 
            "val": config.json_test_labels
        }
        
        # NVIDIAs performance tuning guide:
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        # When to set torch.backends.cudnn.benchmark:
        # 1. The input sizes for your network do not vary.
        # (source: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
        # 2. Your model does not change (i.e. does not have layers that are only
        #    "activated" when certain conditions are met)
        # (source: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not)
        torch.backends.cudnn.benchmark = False
        
        # Determine the default device for the network to train and run on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{model_id} initialized in {str(self.device)}")

        # Model
        self.model_id: str = model_id
        if model is not None:
            self.model: nn.Module = model.to(self.device)
        elif model_path is not None:
            self.model: nn.Module = torch.load(model_path).to(self.device)

        # Model parameters
        self.batch_size: int = config.default_batch_size
        self.num_workers: int = config.num_workers

        self.optimizer = self._get_optimizer(config)
        self.scheduler = self._get_scheduler(config)
        self.prof = self._get_profiler(config)
        self.mAP = MeanAveragePrecision(box_format='xyxy',
                                        iou_type='bbox',
                                        iou_thresholds=[0.5, 0.75, 0.8, 0.85, 0.9, 0.95])

        # Checkpoint info
        self.epoch: int = 0
        self.training_time: float = 0.
        self.losses: List[Losses_dict_t] = []    
        self.mAP_dicts: List[Tuple[int, Dict[str, float]]] = []
        
        self.losses_plot_ylabel = config.losses_plot_ylabel
        self.config = config

        if checkpoint_path:
            self.load(checkpoint_path, load_only_weights)


    def _get_optimizer(self, config: DefaultTrainingConfig) -> optim.SGD:
        # Found out the line of code -after the comments- here:
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        # Did some research and found the link below that may help you
        # understand the relationship between requires_grad and passing the
        # parameters to the optimizer.
        # https://discuss.pytorch.org/t/passing-a-subset-of-the-parameters-to-an-optimizer-equivalent-to-setting-requires-grad-of-subset-only-to-true/42866/2
        # If you do not want parameters that are not trainable to be updated by
        # the optimizer, you need to remove them from the list you pass as args
        # to the optimizer. This update will occure when setting weight_decay or
        # momentum in optim.SGD. More:
        # https://discuss.pytorch.org/t/update-only-sub-elements-of-weights/29101
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params,
                              lr=config.sgd_learning_rate,
                              momentum=config.sgd_momentum,
                              weight_decay=config.sgd_weight_decay)
        return optimizer

    def _get_scheduler(self, config: DefaultTrainingConfig) -> optim.lr_scheduler.MultiStepLR:
        if not self.optimizer:
            raise Exception("Scheduler can not be defined if there is no optimizer")
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                   milestones=config.scheduler_milestones,
                                                   gamma=config.scheduler_gamma)
        return scheduler

    def _get_profiler(self, config: DefaultTrainingConfig) -> Optional[torch.profiler.profile]:
        if not config.profile:
            return None
        profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=config.prof_wait,
                                                 warmup=config.prof_warmup,
                                                 active=config.prof_active, 
                                                 repeat=config.prof_repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/' + self.model_id),
                record_shapes=True,
                with_stack=True,
                profile_memory=True)
        return profiler

    def load(self, checkpoint_path: str, load_only_weights: bool = False) -> None:
        """
        Given a checkpoint_path, loads the checkpoint dictionary
        to the DetectionNetBench attributes.
        Since the scheduler is optional, if the provided checkpoint
        does preserve the state dictionary of a scheduler, but the
        self.scheduler is None, an Attribute error will be raised.
        If the provided checkpoint has not preserved a scheduler's
        state dictionary, but self.scheduler is not None, a warning
        will be printed and the scheduler will be removed
        (i.e. self.scheduler = None)
        """
        checkpoint: Checkpoint_t = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_only_weights:
            return
        self.epoch = checkpoint["epoch"]
        self.losses = checkpoint["losses"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.mAP_dicts = checkpoint["mAPs"]
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            self.scheduler = None
            if self.scheduler:
                warnings.warn(
                    "No scheduler found in the checkpoint, removing current one",
                    RuntimeWarning
                )
            elif checkpoint["scheduler_state_dict"]:
                raise AttributeError(
                    "Scheduler state found in the checkpoint dictionary"
                    "but no scheduler has been specified!"
                )

        self.training_time = checkpoint["training_time"]

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Given the checkpoint path, collect all info into
        a Checkpoint_t dictionary and use torch to save it
        at the given path.
        """
        # https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192/6
        self.model.to(torch.device("cpu"))
        checkpoint: Checkpoint_t = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.epoch,
            "losses": self.losses,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None if not self.scheduler else self.scheduler.state_dict(),
            "training_time": self.training_time,
            "mAPs": self.mAP_dicts
        }
        torch.save(checkpoint, checkpoint_path)
        self.model.to(self.device)

    def save_model(self, model_path: str) -> None:
        # Move the model to the cpu before saving it (enhances portability)
        self.model.to(torch.device("cpu"))
        torch.save(self.model, model_path)
        # Move the model back to the device it should be on
        self.model.to(self.device)

    def reset(self, config: Optional[DefaultTrainingConfig] = None):
        if config:
            self.config = config
        self.epoch = 0
        self.training_time = 0
        self.losses = []
        self.mAP_dicts = []
        self.optimizer = self._get_optimizer(self.config)
        self.scheduler = self._get_scheduler(self.config)
        self.prof = self._get_profiler(self.config)

    def _collate_fn(self, data: list[BoundBoxDataset_Item]) -> Tuple[list[torch.Tensor], list[Dict[str, torch.Tensor]]]:
        """ 
        Created a custom collate_fn, because the default one stacks vertically
        the tensors.

        Based on the example at:
        https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        The fasterrcnn_resnet50_fpn model requires the collate function
        to return images, as a list of tensors and targets, as a list of dictionaries. 
        These dictionaries should be structured as:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

        This function also moves the tensors to the correct device, found at FasterRCNN's initialization.

        TODO(?): Update the code to support multiple BoundingBoxes with different labels for each image.
        """
        images = []
        target = []
        for d in data:
            images.append(d["image"])
            temp_dict = {}
            # Unsqueeze is required here because there are no other boxes.
            # Since FasterRCNN and SSD both need a 2d FloatTensor
            # where dim0 is the number of boxes in the current image and dim1
            # is the bounds of those boxes.
            temp_dict["boxes"] = d["bounding_box"]["boxes"].unsqueeze(0)
            temp_dict["labels"] = d["bounding_box"]["labels"]
            target.append(temp_dict)
        return images, target

    def _show_bounding_boxes_batch(self, images_batch: torch.Tensor, target_batch: List[Bbox_dict_for_nn_t]):
        """Show image with landmarks for a batch of samples."""
        batch_size = len(images_batch)
        fig = plt.figure(figsize=(8,8))
        plt.title('Batch from dataloader')
        plt.axis('off')

        for i in range(batch_size):
            tens = target_batch[i]["boxes"][0]
            rows, cols = tens.shape
            x = []
            y = []
            ax = fig.add_subplot(1, batch_size, i+1) # type: ignore
            ax.imshow(images_batch[i].permute(1, 2, 0))
            for row in range(rows):
                for (i, el) in enumerate(tens[row]):
                        if i % 2 == 0:
                            x.append(el.item())
                        else:
                            y.append(el.item())
                rect = patches.Rectangle(
                        (x[0], y[0]), abs(x[0] - x[1]), abs(y[0] - y[1]), 
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                ax.add_patch(rect)
        plt.show()

    def train(self, num_epochs: int,
              sparse_training_regularizer: Optional[Callable[[nn.Module], None]] = None
    ) -> None:
        """
        Trains the model using the data specified at object initialization.
        Implementation is based on: 
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        # Throw an error if one attempts to train the network without having
        # specified the required paths
        if not self.can_train:
            raise Exception("Paths required for training have not been specified")
        
        weight_decay_off: bool = (self.optimizer.param_groups[0]["weight_decay"] == 0)

        # Get the current time (in seconds) at which the training starts
        since = time.time()

        # Organize the datasets and the dataloaders for training and testing
        datasets = {
            x: BoundingBoxDataset(
                root_dir=self.root_dirs[x], # type: ignore
                json_file=self.json_labels[x] # type: ignore
            ) for x in ["train", "val"]
        }
        dataloaders = {
            x: DataLoader(
                datasets[x], batch_size=self.batch_size, shuffle=True, 
                num_workers=self.num_workers,
                collate_fn=self._collate_fn
            ) for x in ["train", "val"]
        }
        # I will be printing both the validation and training losses, in order
        # to account for overfitting.
        # The model will always be in training mode, but when in validation
        # gradient calculation (autograd) will be disabled.
        self.model.train()

        if not self.losses:
            min_loss = math.inf
        elif weight_decay_off:
            min_loss = min([loss["val"] for loss in self.losses])
        else:
            min_loss = self.losses[-1]["val"]
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # Train for num_epochs
        if self.prof: self.prof.start()
        for epoch in range(self.epoch, self.epoch + num_epochs):
            print(f'\n\nEpoch {epoch}/{self.epoch + num_epochs - 1}')
            print('-' * 10)
            
            curr_loss_dict: Losses_dict_t = {"train": 0., "val": 0., "epoch": epoch}
            # Each epoch has a training phase and a validation phase
            for phase in ["train", "val"]:
                # The proposed way of validating the model in docs is by putting
                # the model in evaluation mode, when calculating the validation
                # losses.
                # self.model.train() if phase == "train" else self.model.eval()

                # The problem is that I could not find any reference on what
                # is the expected output of fasterrcnn_resnet50_fpn, when in
                # evaluation mode and two arguments are passed and probably that
                # is because you should not pass two arguments when in eval mode.
                # After some research, it is suggested to have the model in
                # train mode and get the losses for validation with grads 
                # disabled:
                # https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn

                running_loss = 0.0

                # Iterate over data in batches
                for images, targets in dataloaders[phase]:
                    # Chose not to implement this copy in the collate function,
                    # because when I did, an error occurred when trying to use multiple workers
                    dev_images = [img.to(device=self.device) for img in images]
                    dev_targets = [{
                        "boxes": target["boxes"].to(device=self.device),
                        "labels": target["labels"].to(device=self.device)
                        } for target in targets]
                    
                    # Reset the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward: track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # The model in training mode returns a dictionary containing losses
                        loss_dict = self.model(dev_images, dev_targets)
                        # Sum all the losses returned by the model, as suggested
                        # at: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
                        loss: torch.Tensor = sum(loss for loss in loss_dict.values()) # type: ignore
                        if phase == "train":
                            loss.backward()
                            if sparse_training_regularizer:
                                sparse_training_regularizer(self.model)
                            self.optimizer.step()
                            if self.scheduler: self.scheduler.step()
                            if self.prof: self.prof.step()

                    # statistics
                    running_loss += loss.item()
        
                # Get the mean of all losses
                running_loss /= len(datasets[phase])

                # Print the loss for each phase
                print(f"Phase {phase}: average loss = {running_loss}")
                curr_loss_dict[phase] = running_loss

                # If there is no weight decay and the validation loss is larger
                # than the current min_loss, revert to last epoch
                if weight_decay_off and phase =='val':
                    if running_loss > min_loss:
                        self.model.load_state_dict(best_model_wts)
                    else:
                        min_loss = running_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())

            self.losses.append(curr_loss_dict)

        if self.prof: self.prof.stop()
        
        # Update objects state
        time_elapsed = time.time() - since
        self.epoch = self.epoch + num_epochs
        self.training_time += time_elapsed

        # Print some info about the training session
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Total training time {self.training_time // 60:.0f}m {time_elapsed % 60:.0f}s')

    @torch.no_grad()
    def calculate_metrics(self, preserve_result: bool) -> Dict[str, float]:
        if not self.can_test:
            raise Exception("Paths required for testing have not been specified")

        self.mAP.reset()
        self.model.eval()

        # Organize the datasets and the dataloaders for training and testing
        dataset = BoundingBoxDataset(
            root_dir=self.root_dirs["val"], # type: ignore
            json_file=self.json_labels["val"] # type: ignore
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

        for images, targets in dataloader:
            dev_images = [img.to(device=self.device) for img in images]
            dev_targets = [{
                "boxes": target["boxes"].to(device=self.device),
                "labels": target["labels"].to(device=self.device)
                } for target in targets]

            dev_preds = self.model(dev_images)
            self.mAP.update(dev_preds, dev_targets)

        dev_mAP_dict = self.mAP.compute()
        mAP_dict = {}
        for key, value in dev_mAP_dict.items():
            if torch.is_tensor(value):
                value = value.item()
            mAP_dict.update({key: value})

        if preserve_result:
            self.mAP_dicts.append((self.epoch, mAP_dict,))
        return mAP_dict

    @torch.no_grad()
    def eval(self,
             image: torch.Tensor,
             threshold: float,
             visualize: bool = False
        ) -> Tuple[Optional[BoundingBox], Optional[float]]:
        """
        Uses the network to predict the bounding box on a given image.
        Returns only a single bounding box as a BoundingBox object.
        If there are no bounding boxes, it returns None

        Args:
        - image: The image (in cpu) to run inference for
        - threshold: A float in [0, 1] that determines
                     the acceptable score for a bbox.
                     If the model returns bboxes with scores less
                     than this threshold, None, None will be returned.
        - visualize: When set to True a matplotlib plot
                     will be used to display the bounding
                     box on the image.

        Returns:
        - A BoundingBox object, along with it's score or None
        - The score for the detection returned by the model
        or None.
        """
        
        # Move the image to device
        dev_image = image.to(self.device)
        
        # Prepare the model for evaluation
        self.model.eval()
        
        # Handle output returned by inference
        dict = self.model([dev_image])
        if len(dict[0]["boxes"]) == 0:
            return None, None
        
        # Find the bounding box with the highest score
        # from those returned
        best_score_idx = torch.argmax(dict[0]["scores"])
        
        # If the score is not acceptable return None
        score = dict[0]["scores"][best_score_idx].to("cpu").item()
        if score < threshold:
            return None, None
        
        # There model returned at least one bbox with a score higher
        # that the threshold, move everything to the cpu, back them
        # into a BoundingBox object and return it, along with it's score
        first_box = dict[0]["boxes"][best_score_idx].to("cpu").tolist()
        first_label = dict[0]["labels"][best_score_idx].to("cpu").item()
        bbox = BoundingBox(x1=first_box[0], y1=first_box[1],
                           x2=first_box[2], y2=first_box[3],
                           score=score,
                           label=first_label
                        )
        
        if visualize:
            bbox_dict = bbox.to_nn_dict()
            bbox_dict["boxes"] = bbox_dict["boxes"].unsqueeze(0).unsqueeze(0)
            self._show_bounding_boxes_batch(image.unsqueeze(0), [bbox_dict])

        return bbox, score
    
    @torch.no_grad()
    def visualize_evaluation(self, batch_size: int = 1):
        """
        Runs the model inference on testing data, found at
        root_test_dir location and displays the output in a matplotlib
        figure.
        
        Args:
        - batch_size: The number of images to run the inference and
                      display for.
        """
        # Throw an error if one attempts to visualize the evaluation of the
        # network on the testing data, without having specified the
        # required paths
        if not self.can_test:
            raise Exception("Paths required for visualizing the testing data\
                             have not been specified")
        
        dataset = BoundingBoxDataset(
                    root_dir=self.root_dirs["val"], # type: ignore
                    json_file=self.json_labels["val"] # type: ignore
                  )
        dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=self._collate_fn
                    )

        images, _ = next(iter(dataloader))
        dev_images = [image.to(self.device) for image in images]
        self.model.eval()
        dev_preds = self.model(dev_images)
        # Move predictions (stored in device) to the cpu
        pred = []
        for dev_pred in dev_preds:
          temp = {}
          temp["boxes"] = []
          temp["boxes"].append(dev_pred["boxes"].to(torch.device("cpu")))
          pred.append(temp)
        # Display the predicted bounding boxes
        self._show_bounding_boxes_batch(images, pred)

    def get_inference_frequency(self,
                                num_tests: int,
                                warmup: int,
                                cudnn_benchmark: bool = False
        ) -> Infer_freqs_t:
        """
        Times some inferences in order to calculate the frequency the NN it hosts
        operates at.
        It will print the time required for the first inference to run and
        the average of num_tests inferences. 
        This test is going to be performed twice and thus 4 metrics are going
        to be printed:
        The first 2 metrics are for moving the data to the device and executing
        self.model() and
        The other 2 metrics are derived using self.eval()
        We wanted to observe if there is much delay involved on calling self.model.eval()
        as well as creating a BoundingBox object (in the cpu) with the data returned,
        at each inference. (We observed negligible difference which is probably due to
        moving the inference predictions to the cpu)

        Args:
        - num_tests: The number of inferences to run in order to determine the average frequency
        - warmup: The amount of warmup inferences to execute before starting the timer
        - cudnn_benchmark: Whether to use torch.backends.cudnn.benchmark or not

        Returns:
        A dictionary (TypedDict with the name Infer_freq_t).
        """
        if not self.can_test:
            raise Exception("Paths required for calculating the inference frequency\
                             have not been specified")
        infer_reqs: Infer_freqs_t = {
            "is_cudnn_benchmanrk_on": cudnn_benchmark,
            "num_tests": num_tests,
            "num_warmup": warmup,
            "model_first_meas_ns": 0.,
            "model_avg_freq_Hz": 0.,
            "eval_first_meas_ns": 0.,
            "eval_avg_freq_Hz": 0.
        }
        # Decide whether to benchmark or not
        torch.backends.cudnn.benchmark = cudnn_benchmark

        # Create a dataloader to use in order to fetch images
        dataset = BoundingBoxDataset(self.root_dirs["val"], self.json_labels["val"]) # type: ignore
        dataloader = DataLoader(dataset=dataset,
                                shuffle=True,
                                collate_fn=self._collate_fn,
                                batch_size=max(warmup, num_tests))

        ##########
        # Warmup #
        ##########
        dev_images, _ = next(iter(dataloader))[:warmup]
        dev_images = [img.to(self.device) for img in dev_images]
        # Do some warmup before evaluation
        for i in range(warmup):
            self.eval(image=dev_images[i], threshold=0)

        ###############
        # self.eval() #
        ###############
        dev_images, _ = next(iter(dataloader))[:num_tests]
        dev_images = [img.to(self.device) for img in dev_images]
        first = 0
        start = time.time_ns()
        for i in range(num_tests):
            self.eval(image=dev_images[i], threshold=0)
            if i == 0: first = time.time_ns()
        end = time.time_ns()
        # Calculate the average
        infer_reqs["eval_first_meas_ns"] = first-start
        infer_reqs["eval_avg_freq_Hz"] = num_tests/((end-start)*1e-9)

        ################
        # self.model() #
        ################
        dev_images, _ = next(iter(dataloader))[:num_tests]
        dev_images = [img.to(self.device) for img in dev_images]
        first = 0
        start = time.time_ns()
        self.model.eval()
        first: float = 0
        for i in range(num_tests):
            self.model(dev_images[i].unsqueeze(0))
            if i == 0: first = time.time_ns()
        end = time.time_ns()
        infer_reqs["model_first_meas_ns"] = first-start
        infer_reqs["model_avg_freq_Hz"] = num_tests/((end-start)*1e-9)
        
        return infer_reqs

    def plot_losses(self, filename: str) -> None:
        fig, ax = plt.subplots()
        ax.set_title(f"Losses for model: {self.model_id}")
        for phase in ["train", "val"]:
            losses = [loss[phase] for loss in self.losses]
            epochs = [epoch["epoch"] for epoch in self.losses]
            ax.plot(epochs, losses, label=phase)
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{self.losses_plot_ylabel}")
        fig.savefig(filename)
        plt.close(fig)

    def plot_mAPs(self, filename: str, mAP_key: str) -> None:
        fig, ax = plt.subplots()
        ax.set_title(f"mAP[{mAP_key}] for {self.model_id} at different epochs")
        mAPs = [mAP[mAP_key] for _, mAP in self.mAP_dicts]
        epochs = [epoch for epoch, _ in self.mAP_dicts]
        ax.plot(epochs, mAPs, '--bo')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mean Average Precision (mAP)")
        fig.savefig(filename)
        plt.close(fig)
