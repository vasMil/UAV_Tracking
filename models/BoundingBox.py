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
    def __init__(self,
                 img_name: str, img_height: int, img_width: int,
                 label: int, x: float, y: float, height: float, width: float
        ) -> None:
        # Why is scale required: https://labelstud.io/tags/rectanglelabels.html
        self.scale_y = lambda y: y * (img_height / 100)
        self.scale_x = lambda x: x * (img_width / 100)

        self.img_name = img_name
        self.label = label
        self.x1 = self.scale_x(x)
        self.y1 = self.scale_y(y)
        self.x2 = self.scale_x(x + width)
        self.y2 = self.scale_y(y + height)
        self.area = self.scale_y(height) * self.scale_x(width)

    def __str__(self) -> str:
        return f"BoundingBox: \
            \n\t x1 = {self.x1}, y1 = {self.y1} | top left pixel \
            \n\t x2 = {self.x2}, y2 = {self.y2} | bottom right pixel \
            \n\t area = {self.area} \
            \n\t img_name = {self.img_name}"
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "boxes": torch.Tensor([self.x1, self.y1, self.x2, self.y2]),
            "labels": torch.LongTensor([self.label])
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
        # Label studio returns a string as the label
        # of each bounding box. In order to use that label
        # in a torch NN I need to convert it to an integer.
        self.label_mapper = {
            "Multirotor": 1
        }
        with open(json_file) as f:
            json_obj = json.load(f)
        # Recover the name of the image file
        file_upload = [str(d["file_upload"]).split("-")[-1] for d in json_obj]
        # Get the first bounding box (wrapped in a result json object)
        result = [d["annotations"][0]["result"][0] for d in json_obj]
        # Extract the value attribute for each of the results.
        # It also is a json object.
        value = [r["value"] for r in result]
        # Create the dataframes containing only the useful information
        # of the above json objects.
        img_dim = pd.DataFrame(result)[["original_width", "original_height"]]
        value = pd.DataFrame(value).drop(["rotation"], axis=1)
        file_upload = pd.DataFrame(file_upload)
        # Concatenate the DataFrames into one
        self.df = pd.concat([file_upload, img_dim, value], axis=1)
        self.df.columns.values[0] = "img_name"

    def get_all_bounding_boxes(self) -> Tuple[BoundingBox]:
        """
        Returns a Tuple with all the bounding box data that are in json_file,
        organized as BoundingBox objects.
        """
        return tuple(
                BoundingBox(
                    a.img_name, a.original_height, a.original_width, 
                    self.label_mapper[a.rectanglelabels[0]], a.x, a.y, a.height, a.width
                ) 
                        for a in self.df.itertuples()
            )
        


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

