from typing import Optional, Dict

import torch
from torch import optim, nn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn


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
            print(images[0].shape, target[0]["boxes"].shape, target[0]["labels"].shape)
            out = self.model(images, target)
            print(out)

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
            # Unsqueeze is required here because there are no other 
            temp_dict["boxes"] = d["bounding_box"]["boxes"].unsqueeze(0)
            temp_dict["labels"] = d["bounding_box"]["labels"]
            target.append(temp_dict)
        return images, target
