import math
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


    def moveToBoundingBoxAsync(self, bbox: BoundingBox) -> Future:
        """
        Given a BoundingBox object, calculate its relative distance
        (offset) on the x axis, using the focal length.
        Then using some trigonomerty determine the offset on the
        y and z axis.
        Lastly, add to your current coordinates this calculated offset
        and move towards that object, using moveToPositionAsync().
        """
        offset = airsim.Vector3r()
        # Calculate the distance (x axis) of the two UAV's, using the
        # camera's focal length
        offset.x_val = config.focal_length_const / bbox.width

        # Since the Horizontal FOV is 90deg we may calculate the distance
        # on the y axis that the camera captures.
        img_width_meters = 2 * offset.x_val
        y_box_displacement = bbox.x_center - config.img_width/2
        offset.y_val = (y_box_displacement * img_width_meters) / config.img_width

        # Using the same logic we can calculate the distance on the z axis.
        # We should also take into account that the aspect ratio changes the
        # FOV.
        img_height_meters = 2 * offset.x_val * math.tan(config.vert_fov)
        z_box_displacement = bbox.y_center - config.img_height/2
        offset.z_val = (z_box_displacement * img_height_meters) / config.img_height
        # print("first approach: ", offset.z_val)
        # offset.z_val = z_box_displacement * (config.pawn_size_z / bbox.height)
        # print("second approach: ", offset.z_val)

        # Now that you used trigonometry to get the distance on the y and z axis
        # you should fix the offset on the x axis, since the current one is the
        # distance between the camera and the back side of the leadingUAV, but
        # you need to calculate the distance between the centers of the two UAVs.
        offset.x_val += config.pawn_size_x
        
        # Calculate the new position on EgoUAV's coordinate system to move at.
        new_pos = self.getMultirotorState().kinematics_estimated.position + offset
        print(f"Current estimated pos: {self.getMultirotorState().kinematics_estimated.position}")
        print(f"Predicted offset: {offset}")
        self.lastAction = self.moveToPositionAsync(*new_pos)
        return self.lastAction
