from typing import Tuple

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import ImageFont, ImageDraw, Image
import numpy as np

from constants import IMG_HEIGHT, IMG_WIDTH
from models.BoundingBox import BoundingBox
from models.FrameInfo import FrameInfo

def add_bbox_to_image(image: torch.Tensor, bbox: BoundingBox,
                      color: Tuple[float, float, float] = (1., 0., 0.)
    ) -> torch.Tensor:
    image = image.clone()
    x1 = max(round(bbox.x1)-1, 0)
    x2 = min(round(bbox.x2)+1, IMG_WIDTH-1)
    y1 = max(round(bbox.y1)-1, 0)
    y2 = min(round(bbox.y2)+1, IMG_HEIGHT-1)
    for i, c in enumerate(color):
        image[i, y1, x1:(x2+1)] = c
        image[i, y2, x1:(x2+1)] = c
        image[i, y1:(y2+1), x1] = c
        image[i, y1:(y2+1), x2] = c
    return image

def add_info_to_image(image: torch.Tensor,
                      frameInfo: FrameInfo,
                      score_threshold: float
                    ) -> torch.Tensor:
    # Configure font size and spacing between two lines of text
    # as well as default colors
    font_size = 10
    spacing = 2
    first_line_pos = torch.tensor([5, 5])
    next_line_offset = torch.tensor([0, font_size + spacing])
    green = (21, 237, 191)
    red = (237, 21, 92)
    blue = (66, 135, 245)
    default = (0, 0, 0)
    lines_written = 0

    def getNextLinePos() -> Tuple[float, float]:
        """
        This function uses all the constants defined in add_angle_info_to_image
        in order to return the correct position of the next line, in the correct
        format (i.e. Tuple[float, float])
        """
        nonlocal first_line_pos, next_line_offset, lines_written
        tup = tuple((first_line_pos + lines_written*next_line_offset).tolist())
        lines_written += 1
        return tup

    # Convert image to PIL
    pil_img = F.to_pil_image(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    # Add the text
    for key in list(frameInfo.keys()):
        if key.split('_')[0] == "extra":
            continue

        info = frameInfo[key]
        if info == None:
            color = blue
        elif key == "bbox_score":
            color = green if info >= score_threshold else red
        elif key.split('_')[0] == "err":
            color = green if np.linalg.norm(np.array(info)) < 1 else red
        else:
            color = default

        if isinstance(info, Tuple):
            temp = ""
            for x in info: temp += f"{x:.2f},"
            info = temp[:-1]
        elif isinstance(info, float):
            info = f"{info:.2f}"
        elif info is None:
            info = "None"
        else:
            raise Exception(f"Unexpected type: {type(info)}, for {key}, in infoForFrame object")
        
        draw.text(getNextLinePos(),
                  f"{key:15s}: {info:15s}",
                  font=font,
                  fill=color
                )

    return F.to_tensor(pil_img)

def increase_resolution(image: torch.Tensor, increase_factor: int = 2) -> torch.Tensor:
    """
    Given an image and how much larger it should be this function
    preserves the aspect ratio of this image and repeats increase_factor times
    each pixel value at each image dimension (excluding the color channel).

    This is achieved by utilizing the Kronecker product of matrices (tensors in our case).
    """
    # As stated in the documentation: https://pytorch.org/docs/stable/generated/torch.kron.html
    # If the two tensors that are provided to the kron function do not have the same
    # number of dimensions, the one with the less dimensions will be unsqueezed until
    # it has the same number of dimensions.
    # Reminder: tensor of size (2, 2) if unsqueezed will result to a tensor of size (2, 2, 1)
    # If there is a 3rd dimension to the provided image (the color channels) we do not want
    # to increase their number (ie we require the resulting image to have 3 channels aswell)
    # thus unsqueezing the spread_matrix works.
    spread_matrix = torch.ones(increase_factor, increase_factor)
    return torch.kron(image, spread_matrix)

def load_png_as_tensor(png_file_path):
    image = Image.open(png_file_path)
    tensor_image = transforms.ToTensor()(image) 
    return tensor_image
