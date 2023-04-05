from typing import Optional

import airsim
from msgpackrpc.future import Future
import numpy as np
import torch
from torchvision import transforms as T

from models.UAV import UAV
from GlobalConfig import GlobalConfig as config
from models.BoundingBox import BoundingBox

class EgoUAV(UAV):
    def __init__(self, client: airsim.MultirotorClient, name: str) -> None:
        super().__init__(client, name)


    def _getImage(self, view_mode: bool = False) -> torch.Tensor:
        """
        Returns an RGB image as a tensor, of the EgoUAV.
        
        If view_mode is False:
        - The tensor returned will be of type float and in shape CxHxW
        Else:
        - The tensor returned will be of type uint8 and in shape HxWxC
        """
        # Respond is of type list[airsim.ImageResponse], 
        # since the request is only for a single image,
        # I may extract only the first element of the list
        resp = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)], 
                vehicle_name=self.name
            )[0]
        # Convert the string to a 1D numpy array (dtype = uint8)
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        # Reshape it into the proper tensor dimensions
        img = np.array(np.reshape(img, [resp.height, resp.width, 3]))
        # Convert the numpy array to a pytorch tensor
        if not view_mode:
            # Convert PIL image to a tensor, since it is not for viewing
            img = T.ToTensor()(img)
        else:
            img = torch.from_numpy(img)
        return img
    
    
    def _cheat_move(
            self, 
            position_vec: Optional[torch.Tensor] = None, 
            velocity_vec: Optional[torch.Tensor] = None
        ) -> Future:
        """
        This function is designed to help me determine what information to use as ground truth, when
        training the neural network.
        You should specify exactly one of the two arguments!
        - position_vec: if specified, should contain the latest position of the leadingUAV
        - velocity_vec: if specified, should contain the velocities used by moveByVelocity() method on the last call for the leadingUAV
        """
        if ((position_vec is None and velocity_vec is None) or \
             (position_vec is not None and velocity_vec is not None)):
            raise Exception("EgoUAV::_cheat_move: Exactly one of two arguments should not be None")
        
        if velocity_vec is not None:
            self.lastAction = self.moveByVelocityAsync(*(velocity_vec.tolist()), duration=config.move_duration)
        elif position_vec is not None:
            self.lastAction = self.moveToPositionAsync(*(position_vec.tolist()))

        return self.lastAction


    def move_to_bounding_box(self, bbox: BoundingBox):
        pass