from typing import Tuple, Optional, Literal

import numpy as np
from scipy import signal

from constants import FOCAL_LENGTH_X, FOCAL_LENGTH_Y,\
                      PAWN_SIZE_X, PAWN_SIZE_Y,\
                      IMG_HEIGHT, IMG_WIDTH,\
                      CAMERA_OFFSET_X
from models.BoundingBox import BoundingBox
from models.FrameInfo import EstimatedFrameInfo

import airsim

class PIDController():
    def __init__(self,
                 vel_magn: float,
                 dt: float,
                 Kp: np.ndarray,
                 Ki: np.ndarray,
                 Kd: np.ndarray,
                 cutoff_freqs: np.ndarray,
                 img_height: int = IMG_HEIGHT,
                 img_width: int = IMG_WIDTH
        ) -> None:
        self.vel_magn = vel_magn
        self.dt = dt
        self.Kp = Kp.reshape([3,1])
        self.Ki = Ki.reshape([3,1])
        self.Kd = Kd.reshape([3,1])
        self.cutoff_freqs = cutoff_freqs.reshape([3,1])
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
        self.lost_meas_cnt = 0
        # Cheat
        self.client = airsim.MultirotorClient()
        # Lowpass filter
        if np.any(cutoff_freqs == 0):
            self.filter = None
        else:
            self.filter = Lowpass_3Dfilter(orders=np.ones([3])*2, cutoffs=cutoff_freqs, fs=(1/dt))

    def bbox_to_td_state(self,
                         bbox: BoundingBox,
                         ego_pos: np.ndarray,
                         method: Literal["Muvva", "focal_length", "cheat"]
        ) -> np.ndarray:
        if method == "Muvva":
            return ego_pos + np.diag([-0.05, -6., 3.5]) @ np.array([[46*13 - bbox.area],
                                                              [IMG_WIDTH/2 - bbox.x_center],
                                                              [IMG_HEIGHT/2 - bbox.y_center]])
        elif method == "focal_length":
            x_offset = FOCAL_LENGTH_X * PAWN_SIZE_Y / bbox.width
            y_offset = (bbox.x_center - IMG_WIDTH/2) * x_offset / FOCAL_LENGTH_X
            z_offset = (bbox.y_center - IMG_HEIGHT/2) * x_offset / FOCAL_LENGTH_Y
            x_offset += (CAMERA_OFFSET_X + PAWN_SIZE_X/2)
            return ego_pos + np.array([[x_offset], [y_offset], [z_offset]])
        elif method == "cheat":
            return (self.client.getMultirotorState(vehicle_name="LeadingUAV")
                    .kinematics_estimated.position
                    .to_numpy_array()
                    .reshape([3,1]))
        else:
            raise ValueError("Invalid method")
    
    def clamp(self, control: np.ndarray, mode: Literal["sep_axis", "3d_magnitude"]) -> np.ndarray:
        if mode == "3d_magnitude":
            return (control/np.linalg.norm(control))*self.vel_magn
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
            "extra_pid_d": None,
            "extra_filtered_dist": None
        }
        if bbox is None:
            estInfo["egoUAV_target_velocity"] = tuple(self.prev_control.squeeze().tolist())
            self.lost_meas_cnt += 1
            return (self.prev_control, 0., estInfo)

        td_state = self.bbox_to_td_state(bbox, ego_pos, "Muvva")
        # estInfo["leadingUAV_position"] = tuple(td_state.squeeze().tolist())

        error = ego_pos - td_state

        # dt = (self.lost_meas_cnt+1) * self.dt
        dt = self.dt
        self.lost_meas_cnt = 0

        # Proportional
        proportional = error
        estInfo["extra_pid_p"] = tuple(proportional.squeeze().tolist())

        # Integrator
        self.integrator += np.multiply(error, dt)
        estInfo["extra_pid_i"] = tuple(self.integrator.squeeze().tolist())

        # Derivative
        self.differentiator = (error - self.prev_error) / dt
        estInfo["extra_pid_d"] = tuple(self.differentiator.squeeze().tolist())
        estInfo["egoUAV_target_velocity"] = tuple(self.differentiator.squeeze().tolist())

        # Calculate the output
        control = self.clamp(np.multiply(self.Kp, proportional)
                             + np.multiply(self.Ki, self.integrator) 
                             + np.multiply(self.Kd, self.differentiator), mode='sep_axis')
        # control = self.clamp(proportional + self.integrator + self.differentiator, mode='sep_axis')
        estInfo["egoUAV_target_velocity"] = tuple(control.squeeze().tolist())

        self.prev_control = control
        self.prev_error = error
        self.prev_measurement = ego_pos

        estInfo["still_tracking"] = True
        return (control, 0., estInfo)


class Lowpass_3Dfilter():
    def __init__(self,
                 orders: np.ndarray,
                 cutoffs: np.ndarray,
                 fs: float
    ) -> None:
        self.orders = orders
        self.cutoffs = cutoffs
        self.b = np.empty([3], dtype=np.ndarray)
        self.a = np.empty([3], dtype=np.ndarray)
        self.z = np.empty([3], dtype=np.ndarray)
        for i in range(3):
            self.b[i], self.a[i] = signal.butter(N=orders[i],
                                                Wn=cutoffs[i],
                                                fs=fs,
                                                btype='low',
                                                analog=False)
            self.z[i] = signal.lfilter_zi(self.b[i], self.a[i])

    def filter_sample(self, sample: np.ndarray) -> np.ndarray:
        filt_sample = np.empty([3,1])
        for i in range(3):
            x, self.z[i] = signal.lfilter(self.b[i], self.a[i], [sample[i].item()], zi=self.z[i])
            filt_sample[i,0] = x
        return filt_sample
