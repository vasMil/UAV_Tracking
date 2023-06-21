from typing import Literal, Optional, Tuple
import math

import numpy as np
import airsim

from GlobalConfig import GlobalConfig as config
from controller.KalmanFilter import KalmanFilter
from utils.operations import vector_transformation
from models.FrameInfo import EstimatedFrameInfo

class Controller():
    def __init__(self,
                 filter_type: Literal["None", "KF"] = "None"
            ) -> None:
        self.filter = None
        if filter_type == "KF":
            X_init = np.zeros([6, 1]); X_init[0, 0] = 3.5; X_init[1, 0] = 0; X_init[2, 0] = -50
            P_init = np.zeros([6, 6]); P_init[3, 3] = 1; P_init[4, 4] = 1; P_init[5, 5] = 1
            R = np.array([[30.45326648, -3.27817233, -5.51873313],
                          [-3.27817233, 33.64031622,  1.07687012],
                          [-5.51873313,  1.07687012, 13.53362295]])
            self.filter = KalmanFilter(X_init, P_init, np.zeros([6, 6]), R)
        self.prev_vel = np.zeros([3, 1])
        self.prev_yaw = 0

    def step(self,
             offset: Optional[np.ndarray],
             pitch_roll_yaw_deg: Optional[np.ndarray],
             dt: float,
             ego_pos: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, float, EstimatedFrameInfo]:
        """
        Calculates a velocity, using the target velocity found in config and
        the offset from the target position passed as argument.

        Args:
        offset: A 3x1 column vector with the offset from EgoUAV to LeadingUAV.
        dt: The time interval between two consecutive calls.
        """
        est_frame_info: EstimatedFrameInfo = {
            "egoUAV_target_velocity": None,
            "angle_deg": None,
            "leadingUAV_position": None,
            "leadingUAV_velocity": None,
            "egoUAV_position": None,
            "still_tracking": None
        }

        if ego_pos is not None:
            ego_pos -= self.prev_vel*dt
            est_frame_info["egoUAV_position"] = tuple(ego_pos.squeeze())
        
        if (offset is not None and
            pitch_roll_yaw_deg is not None
        ):
            # Transform the vector from the camera axis to those of the EgoUAV
            # coordinate frame
            offset = vector_transformation(*(pitch_roll_yaw_deg), vec=offset)
        elif offset is None and self.filter is None:
            # est_frame_info["egoUAV_target_velocity"] = tuple(self.prev_vel.squeeze())
            # est_frame_info["angle_deg"] = self.prev_yaw
            return self.prev_vel, self.prev_yaw, est_frame_info
        
        if isinstance(self.filter, KalmanFilter) and ego_pos is not None:
            if offset is None:
                leading_state = self.filter.step(None, dt)
            else:
                leading_pos = ego_pos + offset
                leading_state = self.filter.step(np.pad(leading_pos, ((0,3),(0,0))), dt)

            leading_pos, leading_vel = np.vsplit(leading_state, 2)
            offset = leading_pos - ego_pos
            est_frame_info["leadingUAV_position"] = tuple(leading_pos.squeeze())
            est_frame_info["leadingUAV_velocity"] = tuple(leading_vel.squeeze())
        elif isinstance(self.filter, KalmanFilter) and ego_pos is None:
            raise AttributeError("Estimation of ego position is required when using a KF")
        elif self.filter is None and offset is not None:
            est_frame_info["leadingUAV_position"] = tuple((ego_pos + offset).squeeze())
        elif self.filter is not None and not isinstance(self.filter, KalmanFilter):
            raise AttributeError("Unexpected filter")
        else:
            raise Exception("Impossible path!") # Required for typing

        # Calculate the yaw angle at which the body should be rotated at.
        yaw_deg = math.degrees(math.atan(offset[1] / offset[0]))
        est_frame_info["angle_deg"] = yaw_deg

        # Multiply with the weights
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

        est_frame_info["egoUAV_target_velocity"] = tuple(velocity.squeeze())

        # Update previous values
        self.prev_vel = velocity
        self.prev_yaw = yaw_deg
        return velocity, yaw_deg, est_frame_info
