from typing import Tuple, Optional

import numpy as np

from constants import IMG_HEIGHT, IMG_WIDTH
from models.BoundingBox import BoundingBox
from models.FrameInfo import EstimatedFrameInfo
from utils.operations import vector_transformation

class PIDController():
    def __init__(self,
                 P_GAIN: float,
                 I_GAIN: float,
                 D_GAIN: float,
                 img_height: int = IMG_HEIGHT,
                 img_width: int = IMG_WIDTH
        ) -> None:
        self.prev_error = 0.
        self.sum_error = 0.
        self.prev_control = np.zeros([3,1])
        self.P_GAIN = P_GAIN
        self.I_GAIN = I_GAIN
        self.D_GAIN = D_GAIN
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width

    def compute_target_drone_state(self,
                                   bbox: BoundingBox,
                                   ego_pos: np.ndarray
        ) -> np.ndarray:
        td_state = ego_pos + np.diag([-0.05, -6., 3.5]) @ np.array([[46*13 - bbox.area],
                                                                    [IMG_WIDTH/2 - bbox.x_center],
                                                                    [IMG_HEIGHT/2 - bbox.y_center]])
        return td_state.reshape([3,1])

    def step(self,
            #  bbox: Optional[BoundingBox],
             offset: Optional[np.ndarray],
             pitch_roll_yaw_deg: np.ndarray,
             ego_pos: np.ndarray
        ) -> Tuple[np.ndarray, float, EstimatedFrameInfo]:
        estInfo: EstimatedFrameInfo = {
            "angle_deg": 0.,
            "egoUAV_position": tuple(ego_pos.squeeze().tolist()),
            "egoUAV_target_velocity": None,
            "leadingUAV_position": None,
            "leadingUAV_velocity": None,
            "still_tracking": False
        }
        # if bbox is None:
        #     return (self.prev_control, 0., estInfo)
        # td_state = self.compute_target_drone_state(bbox, ego_pos)
        if offset is None:
            return (self.prev_control, 0., estInfo)
        offset = vector_transformation(*(pitch_roll_yaw_deg), vec=offset)
        td_state = (ego_pos + offset).astype(np.float64)
        ctrl_error = td_state - ego_pos.reshape([3,1])
        delta_error = ctrl_error - self.prev_error
        control = self.P_GAIN * ctrl_error\
                + self.I_GAIN * self.sum_error\
                + self.D_GAIN * delta_error
        control = (control/np.linalg.norm(control))*5
        self.sum_error += ctrl_error
        self.prev_error = ctrl_error
        self.prev_control = control
        estInfo["still_tracking"] = True
        estInfo["egoUAV_target_velocity"] = tuple(control.squeeze().tolist())
        return (control, 0., estInfo)
