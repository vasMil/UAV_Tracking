import os
from typing import Tuple, TypedDict, Dict, Optional

import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype

class BoundingBox():
    """
    Wraps all useful information of each labeled image,
    into an object, for better organization.

    # TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
    """
    def __init__(self,
                 x1: float, y1: float,
                 label: int,
                 x2: Optional[float] = None,
                 y2: Optional[float] = None,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 img_name: Optional[str] = None,
                 img_height: Optional[int] = None,
                 img_width: Optional[int] = None
            ) -> None:
        """
        Two types of constructor can be used:

        To invoke type 1, you need to specify:
        x1, y1: floats that point to the upper left corner of the bounding box
        height: the height of the bounding box
        width: the width of the bounding box
        img_name: the name of the image (you may use this as an id to distinguish between bboxes)
        img_height: the height of the whole image in pixels
        img_width: the width of the whole image in pixels
        label: the label of the object inside the bounding box (0 is reserved for the background)

        To invoke type 2, you need to specify:
        x1, y1: floats that point to the upper left corner of the bounding box
        x2, y2: floats that point to the bottom right corner of the bounding box
        """
        self.label = label
        if (height is not None and 
            width is not None and 
            img_name is not None and 
            img_height is not None and 
            img_width is not None
        ):
            self.__full_init(img_name, img_height, img_width,
                             x1, y1,
                             height, width
                        )
        elif x2 is not None and y2 is not None:
            self.__fast_init(x1, y1, x2, y2)
        else:
            raise Exception("BoundingBox __init__ called with a bad combination of arguments")
        
        self.x_center = self.x1 + self.width/2
        self.y_center = self.y1 + self.height/2


    def __full_init(self,
                    img_name: str, img_height: int, img_width: int,
                    x: float, y: float, height: float, width: float
                ) -> None:
        """
        "First constructor" used by the BoundingBoxFactory in order to parse the
        json file and preserve most useful information
        """
        # Why is scale required: https://labelstud.io/tags/rectanglelabels.html
        self.scale_y = lambda y: y * (img_height / 100)
        self.scale_x = lambda x: x * (img_width / 100)

        self.img_name = img_name
        self.x1 = self.scale_x(x)
        self.y1 = self.scale_y(y)
        self.x2 = self.scale_x(x + width)
        self.y2 = self.scale_y(y + height)
        self.width = self.scale_x(width)
        self.height = self.scale_y(height)
        self.area = self.height * self.width

    def __fast_init(self, x1: float, y1: float, x2: float, y2: float):
        """
        "Second constructor" used to group the data returend by FasterRCNN
        when in inference
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.width = abs(x2 - x1)
        self.height = abs(y2 - y1)

    def __str__(self) -> str:
        return f"BoundingBox: \
            \n\t x1 = {self.x1}, y1 = {self.y1} | top left pixel \
            \n\t x2 = {self.x2}, y2 = {self.y2} | bottom right pixel"
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        A method that will return in dictionary format, the bounding box
        as required by the FasterRCNN model
        """
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
                    x1=a.x,
                    y1=a.y,
                    height=a.height,
                    width=a.width,
                    label=self.label_mapper[a.rectanglelabels[0]],
                    img_name=a.img_name,
                    img_height=a.original_height,
                    img_width=a.original_width
                ) 
                        for a in self.df.itertuples()
            )
        

class BoundBoxDataset_Item(TypedDict):
    image: torch.Tensor
    bounding_box: Dict[str, torch.Tensor]


class BoundingBoxDataset(Dataset):
    # TODO: Update the code to support multiple BoundingBoxes with different labels for each image.
    def __init__(self, root_dir: str, json_file: str, transform = None) -> None:
        self.root_dir = root_dir
        self.json_file = json_file
        self._bounding_boxes = BoundingBoxFactory(json_file).get_all_bounding_boxes()
        self.transform = transform

    def __len__(self):
        return len(self._bounding_boxes)
    
    def __getitem__(self, idx: int) -> BoundBoxDataset_Item:
        img_path = os.path.join(self.root_dir,
                                self._bounding_boxes[idx].img_name)
        image = ConvertImageDtype(torch.float)(read_image(img_path))
        bbox = self._bounding_boxes[idx]
        sample: BoundBoxDataset_Item = {"image": image, "bounding_box": bbox.to_dict()}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        
        return sample
