from typing import Literal, Optional, Tuple

import numpy as np

from GlobalConfig import GlobalConfig as config
from controller.KalmanFilter import KalmanFilter
from utils.operations import vector_transformation, offset_to_yaw_angle
from models.FrameInfo import EstimatedFrameInfo

class Controller():
    """
    Converts measurements to EgoUAV target velocity. If no measurement is
    available it predicts the position of the target.

    The steps followed by the controller:
    1. Transform the offset vector from the camera's axis to the EgoUAV's axis,
    using the orientation of the EgoUAV when the frame was captured.
    2. If a Kalman Filter is to be used:
        - Convert the transformed vector to a target position in 3D space, using
        the EgoUAV's position. 
        From EgoUAV's position we subtract (prev Ego veloc)*dt
        in order to estimate EgoUAV's position at the time the frame was captured.
        Then we may add the transformed offset to this position estimate and get
        the estimate of the LeadingUAV's position.
        - We use this "measurement" (of the LeadingUAV's position) as input for the KF.
        and it return's a better estimate of the LeadingUAV's position.
        - We subtract the EgoUAV's estimated position we added at the first bullet point.
        and get a better estimate of the offset between the EgoUAV and the target.
    3. Multiply the offset on each axis with a weight. This way we may prioritize movement
       on the z axis, since we have a really limited vertical FOV and we do not want to lose
       the target, since otherwise we would not be able to get any measurements.
    4. Convert the weighted offset to a velocity:
       If the magnitude of the offset vector is not 0, we normalize it
       (thus preserving it's direction) and we then multiply it by the
       desired velocity (i.e. config.uav_velocity).
    """
    def __init__(self,
                 filter_type: Literal["None", "KF"] = "None"
            ) -> None:
        self.filter = None
        if filter_type == "KF":
            X_init = np.zeros([6, 1]); X_init[0, 0] = 3.5; X_init[1, 0] = 0; X_init[2, 0] = -100
            P_init = np.diag([0.01, 0.01, 0.01, 1, 1, 1])
            # Measurement noise Covariance Matrix
            R = np.diag([30.45326648, 33.64031622, 13.53362295])
            self.filter = KalmanFilter(X_init=X_init,
                                       P_init=P_init,
                                       R=R,
                                       Q=None,
                                       forget_factor_a=1.01)
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
        all other cases it may be ommitted. If you want to log the estimated \
        LeadingUAV position and the estimated EgoUAV position on each frame, \
        you need to include it in all cases.

        Returns:
        A Tuple containing:
        - A 3x1 column vector with the target velocity for the EgoUAV
        - A yaw target angle (in degrees), so the EgoUAV rotates at an \
        angle at which the LeadingUAV is at the center of it's frame
        - A TypedDict (EstimatedFrameInfo) containing all the estimated values \
        that we want to log.
        """
        est_frame_info: EstimatedFrameInfo = {
            "egoUAV_target_velocity": None,
            "angle_deg": None,
            "leadingUAV_position": None,
            "leadingUAV_velocity": None,
            "egoUAV_position": None,
            "still_tracking": None
        }

        yaw_deg = None

        if ego_pos is not None:
            # Subtract the distance covered by the EgoUAV, while waiting
            # for the inference to finish and/or the controller to advance
            ego_pos -= self.prev_vel*dt
            est_frame_info["egoUAV_position"] = tuple(ego_pos.squeeze())
        
        if (offset is not None and
            pitch_roll_yaw_deg is not None
        ):
            # Transform the vector from the camera axis to those of the EgoUAV
            # coordinate frame
            offset = vector_transformation(*(pitch_roll_yaw_deg), vec=offset)
            yaw_deg = offset_to_yaw_angle(offset)
        elif offset is None and self.filter is None:
            # In this case we do not have a measurement and we also have no filter
            # that is able to predict the current location of the LeadingUAV.
            # Thus we instruct the EgoUAV to continue on it's previous path.
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
            yaw_deg = offset_to_yaw_angle(offset)
            est_frame_info["leadingUAV_position"] = tuple(leading_pos.squeeze())
            est_frame_info["leadingUAV_velocity"] = tuple(leading_vel.squeeze())
        elif isinstance(self.filter, KalmanFilter) and ego_pos is None:
            raise AttributeError("Estimation of ego position is required when using a KF")

        # At this point we know that we do not have a KF.
        # As well as that if self.filter is None, then the offset cannot be None.
        # We include the condition (about the offset) only for completeness.
        elif self.filter is None and offset is not None and ego_pos is not None:
            est_frame_info["leadingUAV_position"] = tuple((ego_pos + offset).squeeze())

        if offset is None:
            raise Exception("This is impossible at this point of code, but Pylance requires it!")

        # Calculate the yaw angle at which the body should be rotated at.
        if not yaw_deg:
            yaw_deg = offset_to_yaw_angle(offset)
        est_frame_info["angle_deg"] = yaw_deg

        # Multiply with the weights
        offset = np.multiply(offset, np.array(
            [[config.weight_vel_x], [config.weight_vel_y], [config.weight_vel_z]]
        ))

        # Preserve the direction of the offset, normalize the vector
        # and multiply with the desired velocity magnitude.
        offset_magn = np.linalg.norm(offset) # ignore: type
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
