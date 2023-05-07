from typing import Tuple

import torch
import torchvision.transforms.functional as F
from PIL import ImageFont, ImageDraw

from GlobalConfig import GlobalConfig as config
from models.BoundingBox import BoundingBox

def add_bbox_to_image(image: torch.Tensor, bbox: BoundingBox) -> torch.Tensor:
    image = image.clone()
    x1 = max(round(bbox.x1), 0)
    x2 = min(round(bbox.x2), config.img_width-1)
    y1 = max(round(bbox.y1), 0)
    y2 = min(round(bbox.y2), config.img_height-1)
    for i, color in enumerate([1, 0, 0]):
        image[i, y1, x1:x2] = color
        image[i, y2, x1:x2] = color
        image[i, y1:y2, x1] = color
        image[i, y1:y2, x2] = color
    return image

def add_angle_info_to_image(image: torch.Tensor,
                            estim_angle_deg: float,
                            sim_angle_deg: float
                        ) -> torch.Tensor:
    # Configure font size and spacing between two lines of text
    # as well as default colors
    font_size = 10
    spacing = 2
    first_line_pos = torch.tensor((5, 5))
    next_line_offset = torch.tensor((0, font_size + spacing))
    green = (21, 237, 191)
    red = (237, 21, 92)
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
    font = ImageFont.truetype("arial.ttf", 10)
    # Add the text
    draw.text(getNextLinePos(),
              f"estim_angle: {estim_angle_deg:.2f}",
              font=font,
              fill=default
            )
    draw.text(getNextLinePos(),
              f"actual_angle: {sim_angle_deg:.2f}", font=font,
              fill=default
            )
    
    angle_error = sim_angle_deg - estim_angle_deg
    draw.text(getNextLinePos(),
              f"error: {angle_error:.2f}",
              font=font,
              fill=green if abs(angle_error) < 1 else red
            )
    return F.to_tensor(pil_img)
