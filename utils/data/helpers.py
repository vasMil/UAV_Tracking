import os
import json


from constants import FILENAME_LEADING_ZEROS
from models.BoundingBox import BoundingBoxFactory

def rename_images(root_path: str, json_file: str) -> None:
    root_path = "data/empty_map/train/"
    bboxes = BoundingBoxFactory(json_file).bboxes
    
    for bbox in bboxes:
        img_path = os.path.join(root_path, bbox.img_name) # type: ignore
        dest_path = os.path.join(root_path, "temp_" + bbox.img_name) # type: ignore
        os.rename(img_path, dest_path)

    for i, bbox in enumerate(bboxes):
        img_path = os.path.join(root_path, "temp_" + bbox.img_name) # type: ignore
        new_img_name = str(i).zfill(FILENAME_LEADING_ZEROS) + ".png"
        dest_path = os.path.join(root_path, new_img_name) # type: ignore
        os.rename(img_path, dest_path)
        bbox.img_name = new_img_name

    with open(json_file, 'w') as f:
        json.dump([bbox.__dict__() for bbox in bboxes], f)
