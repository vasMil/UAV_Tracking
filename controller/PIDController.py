from typing import Tuple, Optional

import numpy as np

from constants import IMG_HEIGHT, IMG_WIDTH
from models.BoundingBox import BoundingBox
from models.FrameInfo import EstimatedFrameInfo

class PIDController():
    def __init__(self,
                 vel_magn: float,
                 dt: float,
                 Kp: np.ndarray,
                 Ki: np.ndarray,
                 Kd: np.ndarray,
                 tau: np.ndarray,
                 img_height: int = IMG_HEIGHT,
                 img_width: int = IMG_WIDTH
        ) -> None:
        self.vel_magn = vel_magn
        self.dt = dt
        self.Kp = Kp.reshape([3,1])
        self.Ki = Ki.reshape([3,1])
        self.Kd = Kd.reshape([3,1])
        self.tau = tau.reshape([3,1])
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width

        self.sum_error = np.zeros([3,1])
        self.prev_error = np.zeros([3,1])
        self.prev_control = np.zeros([3,1])
        self.prev_measurement = np.zeros([3,1])
        self.integrator = np.zeros([3,1])
        self.differentiator = np.zeros([3,1])
        self.limMax = np.zeros([3,1])
        self.limMin = np.zeros([3,1])


    def bbox_to_td_state(self, bbox: BoundingBox, ego_pos: np.ndarray) -> np.ndarray:
        td_state = ego_pos + np.diag([-0.05, -6., 3.5]) @ np.array([[50*18 - bbox.area],
                                                                    [IMG_WIDTH/2 - bbox.x_center],
                                                                    [IMG_HEIGHT/2 - bbox.y_center]])
        return td_state
    
    def clamp(self, control: np.ndarray) -> np.ndarray:
        return np.max(
            np.hstack([
                np.min(np.hstack([control, np.ones([3,1])*self.vel_magn]), axis=1).reshape([3,1]),
                np.ones([3,1])*(-self.vel_magn)
            ]), axis=1).reshape([3,1])

    def step(self, bbox: Optional[BoundingBox], ego_pos: np.ndarray) -> Tuple[np.ndarray, float, EstimatedFrameInfo]:
        estInfo: EstimatedFrameInfo = {
            "angle_deg": 0.,
            "egoUAV_position": tuple(ego_pos.squeeze().tolist()),
            "egoUAV_target_velocity": None,
            "leadingUAV_position": None,
            "leadingUAV_velocity": None,
            "still_tracking": False,
            "extra_pid_p": None,
            "extra_pid_i": None,
            "extra_pid_d": None
        }
        if bbox is None:
            estInfo["egoUAV_target_velocity"] = tuple(self.prev_control.squeeze().tolist())
            return (self.prev_control, 0., estInfo)
        td_state = self.bbox_to_td_state(bbox, ego_pos)

        error = ego_pos - td_state

        # Proportional
        proportional = np.multiply(self.Kp, error)
        estInfo["extra_pid_p"] = tuple(proportional.squeeze().tolist())

        # Integrator
        self.integrator = self.integrator + (1/2) * np.multiply(self.Ki * self.dt, error + self.prev_error)
        # limits = np.array([[0.15, 0.15, 0.15]]).T
        limits = np.array([[5, 5, 5]]).T
        p_mask = self.integrator > limits
        n_mask = self.integrator < -limits
        if np.any(p_mask):
            self.integrator[p_mask] = limits[p_mask]
        elif np.any(n_mask):
            self.integrator[n_mask] = -limits[n_mask]
        estInfo["extra_pid_i"] = tuple(self.integrator.squeeze().tolist())

        # Derivative
        self.differentiator = np.divide(np.multiply(-2*self.Kd, ego_pos - self.prev_measurement) + np.multiply(2*self.tau - self.dt, self.differentiator),
                                        2*self.tau + self.dt)
        estInfo["extra_pid_d"] = tuple(self.differentiator.squeeze().tolist())
        
        # Calculate the output and clamp it's values
        control = self.clamp(proportional + self.integrator + self.differentiator)
        estInfo["egoUAV_target_velocity"] = tuple(control.squeeze().tolist())

        self.prev_control = control
        self.prev_error = error
        self.prev_measurement = ego_pos

        estInfo["still_tracking"] = True
        return (control, 0., estInfo)
