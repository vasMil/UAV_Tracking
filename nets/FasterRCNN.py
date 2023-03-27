from typing import Dict

import torch
from torch import optim, nn
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn

import matplotlib.pyplot as plt

from models.BoundingBox import BoundingBoxDataset


class FasterRCNN(nn.Module):
    def __init__(self, root_dir: str, json_labels: str) -> None:
        super().__init__()
        self.json_labels = json_labels
        self.root_dir = root_dir
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        # self.optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        self.loss = nn.CrossEntropyLoss()

    def train(self, num_epochs = 25) -> None:
        dataset = BoundingBoxDataset(root_dir="data/empty_map/",
                                     json_file="data/empty_map/empty_map.json")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=self._collate_fn)
        self.model.train()
        for batch_idx, (images, target) in enumerate(dataloader):
            if batch_idx == 3:
                self._show_bounding_boxes_batch(images, target)
                plt.show()
            # out = self.model(images, target)
            # print(out)

    def _collate_fn(self, data):
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
        fig = plt.figure(1)
        plt.title('Batch from dataloader')
        plt.axis('off')

        for i in range(batch_size):
            tens = target_batch[i]["boxes"][0]
            x = []
            y = []
            ax = fig.add_subplot(1, batch_size, i+1)
            ax.imshow(images_batch[i].permute(1, 2, 0))
            for (i, el) in enumerate(tens):
                if i % 2 == 0:
                    x.append(el.item())
                else:
                    y.append(el.item())
            ax.scatter(x, y, s=5, c='r')

