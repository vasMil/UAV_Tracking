import math
from typing import Optional, Literal, Tuple

import airsim
from msgpackrpc.future import Future
import numpy as np
import torch
from torchvision import transforms as T

from models.UAV import UAV
from project_types import Filter_t, Motion_model_t
from constants import EGO_UAV_NAME, PORT
from models.BoundingBox import BoundingBox
from nets.DetectionNets import Detection_FasterRCNN
from nets.DetectionNets import Detection_SSD
from controller.Controller import Controller
from controller.CheatController import CheatController
from controller.PIDController import PIDController
from controller.KalmanFilter import KalmanFilter
from models.FrameInfo import EstimatedFrameInfo

class EgoUAV(UAV):
    def __init__(self,
                 name: str = EGO_UAV_NAME,
                 inference_freq_Hz: int = 1,
                 vel_magn: float = 0,
                 filter_type: Filter_t = "KF",
                 motion_model: Motion_model_t = "CA",
                 use_pepper_filter: bool = True,
                 weight_vel: Tuple[float, float, float] = (1, 1, 1,),
                 port: int = PORT,
                 genmode: bool = False
            ) -> None:
        super().__init__(name, vel_magn, port, genmode=genmode)
        # Initialize the NN
        if genmode:
            return
        # self.net = Detection_FasterRCNN()
        # self.net.load("nets/checkpoints/rcnn100.checkpoint")
        self.net = Detection_SSD()
        self.net.load("nets/checkpoints/ssd/rand_init/ssd60.checkpoint")
        # self.controller = Controller(vel_magn=vel_magn,
        #                              dt=(1/inference_freq_Hz),
        #                              weight_vel=weight_vel,
        #                              filter_type=filter_type,
        #                              motion_model=motion_model,
        #                              use_pepper_filter=use_pepper_filter)
        self.controller = PIDController(vel_magn=vel_magn,
                                        dt=(1/inference_freq_Hz),
                                        Kp=np.array([10., -10., 10]),
                                        Ki=np.array([8., -2., 3]),
                                        Kd=np.array([10., -5., 5]),
                                        cutoff_freqs=np.array([0, 0, 0]))


    def _getImage(self,
                  view_mode: bool = False,
                  img_type: airsim.ImageType = airsim.ImageType.Scene # type: ignore
        ) -> torch.Tensor:
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
                [airsim.ImageRequest(0, img_type, False, False)],
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

    def moveToBoundingBoxAsync(self,
                               bbox: Optional[BoundingBox],
                               orient: Tuple[float, float, float],
                               dt: float
                            ) -> Tuple[Future, EstimatedFrameInfo]:
        """
        Given a BoundingBox object, calculate its relative distance
        (offset) on the x axis, using the focal length.
        Then using some trigonomerty determine the offset on the
        y and z axis.
        Lastly, use a Controller to decide upon the target velocity of
        the EgoUAV.

        Args:
        - bbox: The BoundingBox object to move towards | or None
        (If None and there is a Kalman filter present target's position
        will be estimated, else EgoUAV will preserve it's velocity and direction)
        - orient: The orientation of the EgoUAV, when the image frame was captured.
        (This allows us to get the offset back to the original axis, from the camera's
        axis)
        - dt: The amount of (simulation) time that has passed from previous advance 
        on the controller.

        Returns:
        A Tuple with:
        - The Future returned by AirSim
        - An EstimatedFrameInfo object, that contains estimations of target's
        velocity and target's position etc. (If available)
        """
        pitch_roll_yaw_deg = np.array(orient)
        current_pos = self.getMultirotorState().kinematics_estimated.position
        current_pos = np.expand_dims(current_pos.to_numpy_array(), axis=1)
        # velocity, yaw_deg, est_frame_info = self.controller.step(offset=offset,
        #                                                          pitch_roll_yaw_deg=pitch_roll_yaw_deg,
        #                                                          ego_pos=current_pos)
        velocity, yaw_deg, est_frame_info = self.controller.step(bbox, current_pos)
        self.lastAction = self.moveByVelocityAsync(*(velocity.squeeze()),
                                                   duration=dt,
                                                   yaw_mode=airsim.YawMode(False, yaw_deg)
        )
        est_frame_info["still_tracking"] = (bbox is not None)
        return self.lastAction, est_frame_info
