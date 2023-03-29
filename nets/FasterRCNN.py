import time
from typing import Dict, Tuple

import torch
import torch.backends.cudnn
from torch import optim
from torch.utils.data import DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn

import matplotlib.pyplot as plt

from models.BoundingBox import BoundingBoxDataset
import matplotlib.patches as patches


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
        self.model = fasterrcnn_resnet50_fpn(weights=None).to(device=self.device)

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
        self.optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs = 10) -> None:
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
                datasets[x], batch_size=4, num_workers=4, shuffle=True, 
                collate_fn=self._collate_fn
            ) for x in ["train", "val"]
        }
        # I will be printing both the validation and training losses, in order
        # to account for overfitting.
        # The model will always be in training mode, but when in validation
        # gradient calculation (autograd) will be disabled.
        self.model.train()

        # Train for num_epochs
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
                    dev_images = [img.to(device=self.device) for img in images]
                    dev_target = [{
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
                      loss_dict = self.model(dev_images, dev_target)
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
                print(f"Allocated CUDA memory after epoch: {epoch}: {torch.cuda.memory_allocated(0)}")

    def _collate_fn(self, data) -> Tuple[list[torch.Tensor], list[Dict[str, torch.Tensor]]]:
        """ 
        Created a custom collate_fn, because the default one would merge
        the default one stacks vertically the tensors.

        Based on the example at:
        https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        it should return images, as a list of tensors
        and target, as a list of dictionaries, each with a record for the boxes and their corresponding labels.

        TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
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
            ax = fig.add_subplot(1, batch_size, i+1)
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
