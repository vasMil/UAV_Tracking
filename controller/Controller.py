from typing import Optional, Tuple
import math

import numpy as np

from constants import EPS
from project_types import Filter_t, Motion_model_t
from controller.KalmanFilter import KalmanFilter
from controller.PepperFilter import PepperFilter
from utils.operations import vector_transformation
from models.FrameInfo import EstimatedFrameInfo

class Controller():
    """
    Converts measurements to EgoUAV target velocity. If no measurement is\
    available it predicts the position of the target.\

    The steps followed by the controller:
    1. Transform the offset vector from the camera's axis to the EgoUAV's axis,\
    using the orientation of the EgoUAV when the frame was captured.
    2. If a Kalman Filter is to be used:
        - Convert the transformed vector to a target position in 3D space, using\
        the EgoUAV's position.
        - From EgoUAV's position we subtract (prev Ego veloc)*dt to retrieve\
        EgoUAV's position at the time the frame was captured.
        - We add the transformed offset to this EgoUAV's position estimate in order to\
        the estimate of the LeadingUAV's position.
        - We use this "measurement" (of the LeadingUAV's position) as input for the KF.\
        and it return's a better estimate of the LeadingUAV's position.
        - We subtract the EgoUAV's estimated position we added at the first bullet point.\
        and get a better estimate of the offset between the EgoUAV and the target.
    3. Multiply the offset on each axis with a weight. This way we may prioritize movement\
       on the z axis, since we have a really limited vertical FOV and we do not want to lose\
       the target, since otherwise we would not be able to get any measurements.
    4. Convert the weighted offset to a velocity:\
       If the magnitude of the offset vector is not 0, we normalize it
       (thus preserving it's direction) and we then multiply it by the\
       desired velocity (i.e. config.uav_velocity).
    """
    def __init__(self,
                 vel_magn: float,
                 dt: float,
                 weight_vel: Tuple[float, float, float],
                 filter_type: Filter_t = "None",
                 motion_model: Motion_model_t = "CA",
                 use_pepper_filter: bool = True
            ) -> None:
        self.vel_magn = vel_magn
        self.dt = dt
        self.weight_vel = weight_vel
        self.use_pepper_filter = use_pepper_filter
        # Helper variables that preserve the previous instruction
        # of the controller
        self.prev_vel = np.zeros([3, 1])
        self.prev_yaw = 0

        # A filter that will remove impossible measurements
        self.pepper_filter = PepperFilter(2)
        self.filter = None
        # Configure the KF
        if filter_type == "KF":
            X_init = np.zeros([9, 1]); X_init[0, 0] = 3.5; X_init[1, 0] = 0; X_init[2, 0] = -1.6
            P_init = np.diag([0.01, 0.01, 1, 1, 1, 1, 1, 1, 1])
            self.filter = KalmanFilter(X_init=X_init,
                                       P_init=P_init,
                                       dt=dt,
                                       motion_model=motion_model,
                                       forget_factor_a=1.01)

    def offset_to_velocity(self, offset: np.ndarray) -> np.ndarray:
        # Multiply with the weights
        offset = np.multiply(offset.T, self.weight_vel).T

        # Preserve the direction of the offset, normalize the vector
        # and multiply with the desired velocity magnitude.
        offset_magn = np.linalg.norm(offset) # ignore: type
        if offset_magn != 0:
            offset /= offset_magn
            velocity = np.multiply(offset, self.vel_magn)
            assert(self.vel_magn - np.linalg.norm(velocity) < EPS)
        else:
            velocity = np.zeros([3,1])
        return velocity

    def offset_to_yaw_angle(self,
                            offset: np.ndarray,
                            lead_velocity: Optional[np.ndarray] = None,
                            ego_velocity: Optional[np.ndarray] = None
        ) -> float:
        """
        Using the offset that is recorded on any translation of the global coordinate
        system to calculate the yaw angle at which the EgoUAV should be rotated at, in
        order to preserve (on the y axis) the target in it's FOV.
        
        Definition of: "any offset of the global coordinate system"
            No rotation of any of the global coordinate axis is allowed, 
            but all possible translations are
        
        The velocities of the LeadingUAV and the EgoUAV, if provided they will predict
        the offset of the two vehicles (1/inference_frequenct) seconds ahead and use this
        to calculate the yaw angle.

        Args:
        - offset: The offset between the EgoUAV and the target, IN GLOBAL COORDINATE SYSTEM.
        - lead_velocity: The predicted velocity of the LeadingUAV.
        - ego_velocity: The target velocity of the EgoUAV.
    
        Returns:
        The yaw angle at which the target will be found (in degrees).
        """
        if lead_velocity is not None:
            offset += lead_velocity*self.dt
        if ego_velocity is not None:
            offset -= ego_velocity*self.dt
        if offset[0] == 0:
            return 0
        angle_deg = math.degrees(math.atan(offset[1] / offset[0]))
        if angle_deg > 0 and offset[1] < 0:
            angle_deg -= 180
        elif angle_deg < 0 and offset[1] > 0:
            angle_deg += 180
        return angle_deg

    def step(self,
             offset: Optional[np.ndarray],
             pitch_roll_yaw_deg: np.ndarray,
             ego_pos: np.ndarray
        ) -> Tuple[np.ndarray, float, EstimatedFrameInfo]:
        """
        Calculates a velocity, using the target velocity found in config and
        the offset from the target position passed as argument.
        If a filter is used ("KF") it will also cleanup the measurement if there
        is one, or predict the position of the LeadingUAV and use this to calculate
        the target velocity for the egoUAV.

        Args:
        - offset: A 3x1 column vector with the offset from EgoUAV to LeadingUAV.
        - pitch_roll_yaw_deg: A numpy array with the pirch, roll and yaw of the \
        EgoUAV at the time of measurement.
        - dt: The time interval between two consecutive calls.
        - ego_pos: Important for when using the KF, in order to have the offset \
        added to it and be able to estimate the position of the LeadingUAV. In \
        all other cases it is used just for logging purposes.

        Returns:
        A Tuple containing:
        - A 3x1 column vector with the target velocity for the EgoUAV
        - A yaw target angle (in degrees), so the EgoUAV rotates at an \
        angle at which the LeadingUAV is at the center of it's frame
        - A TypedDict (EstimatedFrameInfo) containing all the estimated values \
        that we want to log.
        """
        # Define an empty EstimatedFrameInfo to populate
        # along the way
        est_frame_info: EstimatedFrameInfo = {
            "egoUAV_target_velocity": None,
            "angle_deg": None,
            "leadingUAV_position": None,
            "leadingUAV_velocity": None,
            "egoUAV_position": None,
            "still_tracking": None
        }

        # Subtract the distance covered by the EgoUAV, while waiting
        # for the inference to finish and/or the controller to advance
        ego_pos -= self.prev_vel*self.dt
        
        # Cleanup any pepper noise in the measurements
        # (i.e. discard measurements that are impossible, since there is an upper limit to the velocity of a UAV)
        if self.use_pepper_filter:
            offset = self.pepper_filter.step(meas=offset, max_magn_uav_vel=self.vel_magn, time_interval=self.dt)

        # If the measurement is None and we have no filter to make up for it
        # continue your movement.
        if offset is None and self.filter is None:
            est_frame_info["egoUAV_position"] = tuple(ego_pos.squeeze())
            return self.prev_vel, self.prev_yaw, est_frame_info

        # Transform the vector from the camera axis to those of the EgoUAV
        # coordinate frame.
        if offset is not None:
            offset = vector_transformation(*(pitch_roll_yaw_deg), vec=offset)
            yaw_deg = self.offset_to_yaw_angle(offset)
        
        # If we have a KF available use it here
        leading_vel = None
        if isinstance(self.filter, KalmanFilter):
            if offset is None: # Predict
                leading_state = self.filter.step(None)
            else: # Filter
                leading_pos = ego_pos + offset
                leading_state = self.filter.step(self.filter.pad_measurement(leading_pos))

            # Recover the distance vector (offset) from the position vector retuned by the filter
            leading_pos, leading_vel = leading_state[:3,:], leading_state[3:6,:]
            offset = leading_pos - ego_pos
            est_frame_info["leadingUAV_velocity"] = tuple(leading_vel.squeeze())

        # Convert the offset
        velocity = self.offset_to_velocity(offset=offset) # type: ignore
        yaw_deg = self.offset_to_yaw_angle(offset, # type: ignore
                                           lead_velocity=leading_vel,
                                           ego_velocity=velocity)

        # Add info for the logger
        est_frame_info["egoUAV_position"] = tuple(ego_pos.squeeze())
        est_frame_info["angle_deg"] = yaw_deg
        est_frame_info["leadingUAV_position"] = tuple((ego_pos + offset).squeeze()) # type: ignore
        est_frame_info["egoUAV_target_velocity"] = tuple(velocity.squeeze())

        # Update previous values
        self.prev_vel = velocity
        self.prev_yaw = yaw_deg
        return velocity, yaw_deg, est_frame_info
