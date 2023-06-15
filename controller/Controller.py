from typing import Literal, Optional, Tuple
import math

import numpy as np
import airsim

from GlobalConfig import GlobalConfig as config
from controller.KalmanFilter import KalmanFilter
from utils.operations import rotate3d, rotate_to_yaw

class Controller():
    def __init__(self,
                 filter_type: Literal["None", "KF"] = "None"
            ) -> None:
        self.filter = None
        if filter_type == "KF":
            client = airsim.MultirotorClient()
            gt_pos = client.simGetGroundTruthKinematics(vehicle_name="LeadingUAV").position
            X_init = np.zeros([6, 1]); X_init[0, 0] = gt_pos.x_val; X_init[1, 0] = gt_pos.y_val; X_init[2, 0] = gt_pos.z_val
            P_init = np.zeros([6, 6]); P_init[3, 3] = 1; P_init[4, 4] = 1; P_init[5, 5] = 1
            R = np.array([[30.45326648, -3.27817233, -5.51873313],
                          [-3.27817233, 33.64031622,  1.07687012],
                          [-5.51873313,  1.07687012, 13.53362295]])
            self.filter = KalmanFilter(X_init, P_init, np.zeros([6, 6]), R)
        self.prev_vel = np.zeros([3, 1])
        self.prev_yaw = 0

    def step(self,
             offset: Optional[np.ndarray],
             pitch_roll_yaw_deg: np.ndarray,
             dt: float,
             ego_pos: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, float]:
        """
        Calculates a velocity, using the target velocity found in config and
        the offset from the target position passed as argument.

        Args:
        offset: A 3x1 column vector with the offset from EgoUAV to LeadingUAV.
        dt: The time interval between two consecutive calls.
        """
        if offset is None and self.filter is None:
            return self.prev_vel, self.prev_yaw

        # Add weights to or process the measurement before converting it to a velocity
        if offset is not None and self.filter is None:
            yaw_deg = pitch_roll_yaw_deg[2] + math.degrees(math.atan(offset[1] / offset[0]))
            # Rotate the offset vector so we may view the camera coordinate system
            # as the EgoUAVs coordinate system.
            offset = rotate3d(*(pitch_roll_yaw_deg), point=offset)
            # Adjust the offset by subtracting the expected amount the EgoUAV moved in the time
            # between the frame capture and the step of the controller
            offset -= np.multiply(self.prev_vel, dt)
        elif isinstance(self.filter, KalmanFilter) and ego_pos is not None:
            ego_pos -= self.prev_vel*dt
            if offset is None:
                leading_state = self.filter.step(None, dt)
            else:
                offset = rotate3d(*(np.multiply(pitch_roll_yaw_deg, -1)), point=offset)
                leading_pos = ego_pos + offset
                leading_state = self.filter.step(np.pad(leading_pos, ((0,3),(0,0))), dt)

            leading_pos, leading_vel = np.vsplit(leading_state, 2)
            offset = leading_pos - ego_pos
            yaw_deg = math.degrees(math.atan(offset[1] / offset[0]))
        else:
            raise AttributeError("Filter type is wrong or ego_pos is not provided!")

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

        self.prev_vel = velocity
        self.prev_yaw = yaw_deg
        return velocity, yaw_deg
