import math
import time
from typing import Dict, Tuple
import copy

import torch
import torch.backends.cudnn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.BoundingBox import BoundingBoxDataset, BoundBoxDataset_Item

from GlobalConfig import GlobalConfig as config


class FasterRCNN():
    def __init__(
          self, 
          root_train_dir: str, json_train_labels: str,
          root_test_dir: str, json_test_labels: str
        ) -> None:
        """
        Class that trains and tests fasterrcnn_resnet50_fpn on custom data.
        """
        super().__init__()
        self.root_dirs = {
            "train": root_train_dir, 
            "val": root_test_dir
            }
        self.json_labels = {
            "train": json_train_labels, 
            "val": json_test_labels
            }
        # Determine the default device for the network to train and run on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("FasterRCNN initialized in " + str(self.device))

        # NVIDIAs performance tuning guide:
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        # When to set torch.backends.cudnn.benchmark:
        # 1. The input sizes for your network do not vary.
        # (source: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
        # 2. Your model does not change (i.e. does not have layers that are only
        #    "activated" when certain conditions are met)
        # (source: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not)
        torch.backends.cudnn.benchmark = True

        # Define the model
        # It is important to move it to the GPU, before initializing the
        # optimizer:
        # https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least/66096687#66096687
        self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2).to(self.device)

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
        # Momentum and weight_decay constants were found at the Faster RCNN paper.
        self.optimizer = optim.SGD(params, 
                                   lr=config.sgd_learning_rate,
                                   momentum=config.sgd_momentum, 
                                   weight_decay=config.sgd_weight_decay
                                )

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=config.scheduler_milestones,
                                                        gamma=config.scheduler_gamma
                                                    )
        
        if config.profile:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=config.prof_wait,
                                                 warmup=config.prof_warmup,
                                                 active=config.prof_active, 
                                                 repeat=config.prof_repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fasterrcnn'),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )


    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs: int = config.num_epochs) -> None:
        """
        Trains the model using the data specified at object initialization.
        Implementation is based on: 
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        # Get the current time (in seconds) at which the training starts
        since = time.time()

        # Organize the datasets and the dataloaders for training and testing
        datasets = {
            x: BoundingBoxDataset(
                root_dir=self.root_dirs[x],
                json_file=self.json_labels[x]
            ) for x in ["train", "val"]
        }
        dataloaders = {
            x: DataLoader(
                datasets[x], batch_size=config.batch_size, shuffle=True, 
                num_workers=config.num_workers,
                collate_fn=self._collate_fn
            ) for x in ["train", "val"]
        }
        # I will be printing both the validation and training losses, in order
        # to account for overfitting.
        # The model will always be in training mode, but when in validation
        # gradient calculation (autograd) will be disabled.
        self.model.train()

        min_loss = math.inf
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # Train for num_epochs
        if config.profile: self.prof.start()
        for epoch in range(num_epochs):
            print(f'\n\nEpoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
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
                for batch_idx, (images, targets) in enumerate(dataloaders[phase]):
                    # Move data to device
                    # Chose not to implement this copy in the collate function,
                    # because when I did, an error occurred when trying to use
                    # multiple workers
                    dev_images = [img.to(device=self.device) for img in images]
                    dev_targets = [{
                        "boxes": target["boxes"].to(device=self.device),
                        "labels": target["labels"].to(device=self.device)
                        } for target in targets]
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # The model in training mode returns a dictionary containing
                        # losses
                        loss_dict = self.model(dev_images, dev_targets)
                        
                        # Sum all the losses returned by the model, as suggested
                        # at: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
                        loss = torch.stack(
                                [loss for loss in loss_dict.values()]
                            ).sum()

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()
                            if config.profile: self.prof.step()
                    
                    # statistics
                    running_loss += loss.item()
        
                # Print the loss for each phase
                print(f"Phase {phase}: loss = {running_loss}")
                
                # Revert last epoch if the validation loss is larger
                if phase == 'val' and running_loss > min_loss:
                    self.model.load_state_dict(best_model_wts)
                else:
                    min_loss = running_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        
        if config.profile: self.prof.stop()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


    def visualize_evaluation(self, batch_size: int = 1):
        dataset = BoundingBoxDataset(
                    root_dir="data/empty_map/train/", 
                    json_file="data/empty_map/train/empty_map.json"
                  )
        dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=self._collate_fn
                    )

        images, _ = next(iter(dataloader))
        dev_images = [image.to(torch.device("cpu")) for image in images]
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


    def _collate_fn(self, data: list[BoundBoxDataset_Item]) -> Tuple[list[torch.Tensor], list[Dict[str, torch.Tensor]]]:
        """ 
        Created a custom collate_fn, because the default one stacks vertically
        the tensors.

        Based on the example at:
        https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        The fasterrcnn_resnet50_fpn model requires from the collate function 
        to return images, as a list of tensors 
        and targets, as a list of dictionaries. 
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
            # Since fasterrcnn_resnet50_fpn requires we pass a 2d FloatTensor
            # where dim0 is the number of boxes in the current image and dim1
            # is the bounds of those boxes.
            temp_dict["boxes"] = d["bounding_box"]["boxes"].unsqueeze(0)
            temp_dict["labels"] = d["bounding_box"]["labels"]
            target.append(temp_dict)
        return images, target

    def _show_bounding_boxes_batch(self, images_batch: torch.Tensor, target_batch: list[Dict[str, torch.Tensor]]):
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
