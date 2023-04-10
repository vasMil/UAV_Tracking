import math
from typing import TypedDict, Mapping, Any, Tuple, Dict, Optional
import time
import copy

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD, _vgg_extractor
from torchvision.models.vgg import vgg16
# https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from GlobalConfig import GlobalConfig as config
from models.BoundingBox import BoundingBox, BoundingBoxDataset, BoundBoxDataset_Item

class Checkpoint_t(TypedDict):
    model_state_dict: Mapping[str, Any]
    epoch: int
    loss: float
    optimizer_state_dict: dict[Any, Any]
    training_time: float

class SSD256x144_VGG16():
    def __init__(self,
                 root_train_dir: str = "", json_train_labels: str = "",
                 root_test_dir: str = "", json_test_labels: str = ""
            ) -> None:
        
        # Determine if the required paths for training have been specified
        self.can_train = True if root_train_dir and \
                                 json_train_labels and \
                                 root_test_dir and \
                                 json_test_labels \
                              else False
        
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
        
        # Customize the model
        backbone = vgg16(weights=None, progress=True)
        backbone = _vgg_extractor(backbone, False, 5)
        # Source: https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/models/detection/ssd.py#L466
        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )
        self.model = SSD(backbone=backbone,
                         anchor_generator=anchor_generator,
                         size=(300, 300),
                         num_classes=2
                    ).to(self.device)
        self.epoch: int = 0
        self.loss: float = math.inf
        self.training_time: float = 0.

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(params,
                                   lr=config.sgd_learning_rate,
                                   momentum=config.sgd_momentum,
                                   weight_decay=config.sgd_weight_decay
                                )
        
    def load(self, checkpoint_path: str) -> None:
        checkpoint: Checkpoint_t = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_time = checkpoint["training_time"]

    def save(self, checkpoint_path: str) -> None:
        checkpoint: Checkpoint_t = {
            "model_state_dict": self.model.state_dict(),
            "epoch":self.epoch,
            "loss":self.loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_time":self.training_time
        }
        torch.save(checkpoint, checkpoint_path)
    
    def train(self, num_epochs: int = config.num_epochs) -> None:
        """
        Trains the model using the data specified at object initialization.
        Implementation is based on: 
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        # Throw an error if one attempts to train the network without having
        # specified the required paths
        if not self.can_train:
            raise Exception("Paths required for training have not been specified")
        
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
        self.model.train()

        min_loss = self.loss
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # Train for num_epochs
        for epoch in range(self.epoch, self.epoch + num_epochs):
            print(f'\n\nEpoch {epoch}/{self.epoch + num_epochs - 1}')
            print('-' * 10)
            
            # Each epoch has a training phase and a validation phase
            for phase in ["train", "val"]:

                running_loss = 0.0

                # Iterate over data in batches
                for images, targets in dataloaders[phase]:
                    # Move data to device
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
        
        time_elapsed = time.time() - since
        self.epoch = self.epoch + num_epochs
        self.loss = min_loss
        self.training_time += time_elapsed
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Total training time {self.training_time // 60:.0f}m {time_elapsed % 60:.0f}s')

    @torch.no_grad()
    def eval(self, image: torch.Tensor) -> Optional[BoundingBox]:
        # Move the image to device
        dev_image = image.to(self.device)
        # Prepare the model for evaluation
        self.model.eval()
        # Handle output returned by inference
        dict = self.model([dev_image])
        if len(dict[0]["boxes"]) == 0:
            return None
        first_box = dict[0]["boxes"][0].to("cpu").tolist()
        first_label = dict[0]["labels"][0].to("cpu").item()
        
        bbox = BoundingBox(x1=first_box[0], y1=first_box[1],
                           x2=first_box[2], y2=first_box[3],
                           label=first_label
                        )
        # bbox_dict = bbox.to_dict()
        # bbox_dict["boxes"] = bbox_dict["boxes"].unsqueeze(0).unsqueeze(0)
        # self._show_bounding_boxes_batch(image.unsqueeze(0), [bbox_dict])
        return bbox

    @torch.no_grad()
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
