from typing import Literal, Optional, Tuple
import math

import numpy as np

from GlobalConfig import GlobalConfig as config
from controller.KalmanFilter import KalmanFilter
from utils.operations import rotate3d

class Controller():
    def __init__(self,
                 filter_type: Literal["None", "KF"] = "None"
            ) -> None:
        self.filter = None        
        self.prev_vel = np.zeros([3, 1])
        self.prev_yaw = 0

    def step(self, offset: Optional[np.ndarray], pitch_roll_yaw_deg: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """
        Calculates a velocity, using the target velocity found in config and
        the offset from the target position passed as argument.

        Args:
        offset: A 3x1 column vector with the offset from EgoUAV to LeadingUAV.
        dt: The time interval between two consecutive calls.
        """
        if offset is None:
            return self.prev_vel, self.prev_yaw
        
        yaw_deg = pitch_roll_yaw_deg[2] + math.degrees(math.atan(offset[1] / offset[0]))

        offset = rotate3d(*(pitch_roll_yaw_deg), point=offset)
        offset = np.multiply(offset, np.array(
            [[config.weight_vel_x], [config.weight_vel_y], [config.weight_vel_z]]
        ))
        
        offset_magn = np.linalg.norm(offset)
        if offset_magn != 0:
            offset /= offset_magn
            velocity = np.multiply(offset, config.uav_velocity)
            assert(config.uav_velocity - np.linalg.norm(velocity) < config.eps)
        else:
            velocity = np.zeros([3,1])
        
        self.prev_vel = velocity
        self.prev_yaw = yaw_deg
        return velocity, yaw_deg
