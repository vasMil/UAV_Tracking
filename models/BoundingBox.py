import os
from typing import Tuple, Dict, Union

import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms as T, ConvertImageDtype

class BoundingBox():
    """
    Wraps all useful information of each labeled image,
    into an object, for better organization.

    # TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
    """
    def __init__(self, x: float, y: float, height: float, width: float, img_name: str) -> None:
        self.img_name = img_name
        self.x1 = x
        self.y1 = y
        self.height = height
        self.width = width
        self.x2 = x + width
        self.y2 = y + height
        self.area = height*width

    def __str__(self) -> str:
        return f"BoundingBox: \
            \n\t x = {self.x1}, y = {self.y1} \
            \n\t height = {self.height}, width = {self.width}, area = {self.area} \
            \n\t img_name = {self.img_name}"
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "boxes": torch.Tensor([self.x1, self.y1, self.x2, self.y2]),
            "labels": torch.LongTensor([0])
        }


class BoundingBoxFactory():
    """
    Given a json_file that is exported from label_studio,
    extracts all the useful information for the training of the NNs
    into a pandas.DataFrame.

    This DataFrame can later be used to organize all those data into
    BoundingBox objects.

    # TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
    """
    def __init__(self, json_file) -> None:
        self.json_file = json_file
        with open(json_file) as f:
            json_obj = json.load(f)
        file_upload = [str(d["file_upload"]).split("-")[-1] for d in json_obj]
        annotations = [d["annotations"][0]["result"][0]["value"] for d in json_obj]
        file_upload = pd.DataFrame(file_upload)
        annotations = pd.DataFrame(annotations).drop(["rotation", "rectanglelabels"], axis=1)
        self.df = pd.concat([file_upload, annotations], axis=1)
        self.df.columns.values[0] = "img_name"

    # def get_bounding_box(img_name: str) -> BoundingBox:
    #     pass

    def get_all_bounding_boxes(self) -> Tuple[BoundingBox]:
        """
        Returns a Tuple with all the bounding box data that are in json_file,
        organized as BoundingBox objects.
        """
        return tuple(BoundingBox(a.x, a.y, a.height, a.width, a.img_name) for a in self.df.itertuples())


class BoundingBoxDataset(Dataset):
    # TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
    def __init__(self, root_dir: str, json_file: str, transform = None) -> None:
        self.root_dir = root_dir
        self.json_file = json_file
        self._bounding_boxes = BoundingBoxFactory(json_file).get_all_bounding_boxes()
        self.transform = transform

    def __len__(self):
        return len(self._bounding_boxes)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]]:
        img_path = os.path.join(self.root_dir,
                                self._bounding_boxes[idx].img_name)
        image = ConvertImageDtype(torch.float)(read_image(img_path))
        bbox = self._bounding_boxes[idx]
        sample = {"image": image, "bounding_box": bbox.to_dict()}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        
        return sample

