from typing import Optional, Tuple
import math

import numpy as np
import airsim

from constants import EPS
from project_types import Filter_t, Motion_model_t
from controller.Controller import Controller
from controller.KalmanFilter import KalmanFilter
from controller.PepperFilter import PepperFilter
from models.FrameInfo import EstimatedFrameInfo

class CheatController(Controller):
    def __init__(self,
                 vel_magn: float,
                 dt: float,
                 weight_vel: Tuple[float, float, float],
                 filter_type: Filter_t = "None",
                 motion_model: Motion_model_t = "CA",
                 use_pepper_filter: bool = True
            ) -> None:
        super().__init__(vel_magn=vel_magn,
                         dt=dt,
                         weight_vel=weight_vel,
                         filter_type=filter_type,
                         motion_model=motion_model,
                         use_pepper_filter=use_pepper_filter)
        self.client = airsim.MultirotorClient()

    def step(self, *args, **kwargs) -> Tuple[np.ndarray, float, EstimatedFrameInfo]:
        """
        All arguments are ignored, since we will cheat...

        Returns:
        A Tuple containing:
        - A 3x1 column vector with the target velocity for the EgoUAV
        - A yaw target angle (in degrees), so the EgoUAV rotates at an \
        angle at which the LeadingUAV is at the center of it's frame
        - A TypedDict (EstimatedFrameInfo) containing all the estimated values \
        that we want to log.
        """

        ego_pos = self.client.simGetGroundTruthKinematics("EgoUAV").position.to_numpy_array().reshape([3,1])
        lead_pos = self.client.simGetGroundTruthKinematics("LeadingUAV").position.to_numpy_array().reshape([3,1])
        offset = lead_pos - ego_pos
        offset[0,0] += 3.5
        # If you want to use the rest of the pipeline use the return statement below
        # return super().step(offset, np.zeros([3]), ego_pos=ego_pos)
        lead_vel = self.client.simGetGroundTruthKinematics("LeadingUAV").linear_velocity.to_numpy_array().reshape([3,1])
        velocity = super().offset_to_velocity(offset)
        yaw = super().offset_to_yaw_angle(offset, lead_velocity=lead_vel, ego_velocity=velocity)

        info: EstimatedFrameInfo = {
            "angle_deg": yaw,
            "egoUAV_position": tuple(ego_pos.tolist()),
            "leadingUAV_position": tuple((ego_pos + offset).tolist()),
            "egoUAV_target_velocity": tuple(velocity.tolist()),
            "leadingUAV_velocity": tuple(lead_vel.tolist()),
            "still_tracking": True
        }
        return velocity, yaw, info
