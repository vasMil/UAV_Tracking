import os
from typing import List, Dict, Optional

import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype

from project_types import Bbox_dict_t, Bbox_dict_for_nn_t, BoundBoxDataset_Item

class BoundingBox():
    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 label: int, score: Optional[float] = None,
                 img_name: Optional[str] = None, img_height: Optional[int] = None,
                 img_width: Optional[int] = None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.width = x2 - x1
        self.height = y2 - y1
        self.label = label
        self.score = score
        self.img_name = img_name
        self.img_height = img_height
        self.img_width = img_width
        self.x_center = self.x1 + self.width/2
        self.y_center = self.y1 + self.height/2
        self.area = self.width * self.height
        if y1 < 0 or x1 < 0:
            raise Exception(f"A BoundingBox can not have negative coordinates!")
        if img_height and y2 >= img_height:
            raise Exception(f"Cannot instantiate a BoundingBox that exceeds the img_height!")
        if img_width and x2 >= img_width:
            raise Exception(f"Cannot instantiate a BoundingBox that exceeds the img_width!")
        if self.width <= 0:
            raise Exception(f"Bounding box should have a width > 0!")
        if self.height <= 0:
            raise Exception(f"Bounding box should have a height > 0!")

    def __str__(self) -> str:
        return f"BoundingBox: \
            \n\t x1 = {self.x1}, y1 = {self.y1} | top left pixel \
            \n\t x2 = {self.x2}, y2 = {self.y2} | bottom right pixel"

    def __dict__(self) -> Bbox_dict_t:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "label": self.label,
            "img_name": self.img_name,
            "img_height": self.img_height,
            "img_width": self.img_width,
        }

    def to_nn_dict(self) -> Bbox_dict_for_nn_t:
        return {
            "boxes": torch.Tensor([self.x1, self.y1, self.x2, self.y2]),
            "labels": torch.LongTensor([self.label])
        }

    @classmethod
    def from_bbox_dict(cls, bbox_dict: Bbox_dict_t):
        x1, y1 = bbox_dict["x1"], bbox_dict["y1"]
        x2, y2 = bbox_dict["x2"], bbox_dict["y2"]
        img_name = bbox_dict["img_name"]
        img_height = bbox_dict["img_height"]
        img_width = bbox_dict["img_width"]
        label = bbox_dict["label"]
        return cls(x1=x1, y1=y1, y2=y2, x2=x2, label=label,
                   img_name=img_name, img_height=img_height, img_width=img_width)


class BoundingBoxFactory():
    """
    Using the json_file provided it should extract the List with Bbox_dict_t
    objects and convert it to a List of BoundingBox objects.
    """
    def __init__(self, json_file) -> None:
        self.json_file = json_file
        self.bboxes: List[BoundingBox] = []
        with open(json_file) as f:
            bbox_dicts: List[Bbox_dict_t] = json.load(f)
        for bbox_dict in bbox_dicts:
            self.bboxes.append(BoundingBox.from_bbox_dict(bbox_dict))
        

class BoundingBoxDataset(Dataset):
    """
    The Dataset as required for training a pytorch model.
    """
    # TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
    def __init__(self, root_dir: str, json_file: str, transform = None) -> None:
        self.root_dir = root_dir
        self.json_file = json_file
        self._bounding_boxes = BoundingBoxFactory(json_file).bboxes
        self.transform = transform

    def __len__(self):
        return len(self._bounding_boxes)
    
    def __getitem__(self, idx: int) -> BoundBoxDataset_Item:
        img_name = self._bounding_boxes[idx].img_name
        if img_name is None:
            raise Exception(f"img_name for bbox at index: {idx} is None!")
        img_path = os.path.join(self.root_dir, img_name)
        image = ConvertImageDtype(torch.float)(read_image(img_path))
        bbox = self._bounding_boxes[idx]
        sample: BoundBoxDataset_Item = {"image": image, "bounding_box": bbox.to_nn_dict()}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        
        return sample
