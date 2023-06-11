from typing import Literal, Tuple
import math

import numpy as np

from GlobalConfig import GlobalConfig as config
from controller.KalmanFilter import KalmanFilter

class Controller():
    def __init__(self,
                 filter_type: Literal["None", "KF"] = "None"
            ) -> None:
        if filter_type == "KF":
            X_init = np.zeros([6, 1]); X_init[0, 0] = 3.5; X_init[2, 0] = -1.6
            P_init = np.zeros([6, 6])#; P_init[3, 3] = 0.5; P_init[4, 4] = 0; P_init[5, 5] = 0.5
            R = np.array([[7.9882659, 3.17199785, 1.58456132],
                          [3.17199785, 15.04112204, 0.14100749],
                          [1.58456132, 0.14100749, 3.98863264]]
            )
            self.filter = KalmanFilter(X_init, P_init, np.zeros([6, 6]), R)
            self.prev_estim = X_init[0:3, :]
        else:
            self.filter = None
        
        self.prev_vel = np.zeros([3, 1])

    def step(self, offset: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """
        Calculates a velocity, using the target velocity found in config and
        the offset from the target position passed as argument.

        Args:
        offset: A 3x1 column vector with the offset from EgoUAV to LeadingUAV.
        dt: The time interval between two consecutive calls.
        """

        # Check if the offset is zeros and if it is,
        # use prev_vel to predict the position of the LeadingUAV
        if np.all(offset == np.zeros([3, 1])) and not self.filter:
            return self.prev_vel, 0
        
        if not self.filter:
            # Estimate the distance traveled by the LeadingUAV between the time we took the frame
            # and the time we got the bbox
            estim_vel = (offset / np.linalg.norm(offset)) * config.uav_velocity
            # offset += estim_vel * dt
            # Take into account the fact that there is a timegap between the frame capture and the network
            # output.
            # offset -= self.prev_vel * dt
            # offset_updatved += 
            # It is important to preserve the LeadingUAV in our FOV.
            # Thus you do an element-wise multiplication with the weights,
            # in order to favour moving on the z axis.
            fixed_offset = np.multiply(offset, np.array(
                [[config.weight_vel_x], [config.weight_vel_y], [config.weight_vel_z]]
            ))
        elif np.all(offset == np.zeros([3, 1])): # and self.filter (implied)
            fixed_offset = self.filter.step(np.pad(self.prev_estim, ((0,3), (0,0))), dt)[0:3, :]
            self.prev_estim = fixed_offset
        else:
            # Fix the offset to be valid as a measurement for the Kalman Filter
            fixed_offset = self.filter.step(np.pad(offset, ((0,3), (0,0))), dt)[0:3, :]
            self.prev_estim = fixed_offset

        # Handle the edge case, in which the filter
        # returns a zero for the offset on all axis.
        if np.linalg.norm(fixed_offset) != 0:
            # Normalize the weighted offset (fixed_offset)
            fixed_offset /= np.linalg.norm(fixed_offset)
            # Multiply with the magnitude of the target velocity
            velocity = np.multiply(fixed_offset, config.uav_velocity)
            # Check if the calculated velocity has the desired magnitude (specified in config)
            assert(config.uav_velocity - np.linalg.norm(velocity) < config.eps)
        else:
            velocity = self.prev_vel

        self.prev_vel = velocity
        return velocity, math.atan(offset[1] / offset[0])
