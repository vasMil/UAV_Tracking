import numpy as np

from GlobalConfig import GlobalConfig as config
from models.UAV import UAV

class Controller():
    def __init__(self, uav: UAV) -> None:
        self.uav = uav
        self.prev_vel = np.zeros([3, 1])

    def step(self, offset: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculates a velocity, using the target velocity found in config and
        the offset from the target position passed as argument.

        Args:
        offset: A 3x1 column vector with the offset from EgoUAV to LeadingUAV.
        dt: The time interval between two consecutive calls.
        """

        # Check if the offset is zeros and if it is,
        # use prev_vel to predict the position of the LeadingUAV
        if np.all(offset == np.zeros([3, 1])):
            # Using the previous estimation for the velocity, "predict"
            # the expected position of the leadingUAV.
            fixed_offset = np.multiply(self.prev_vel, dt)
        else:
            # It is important to preserve the LeadingUAV in our FOV.
            # Thus you do an element-wise multiplication with the weights,
            # in order to favour moving on the z axis.
            fixed_offset = np.multiply(offset, np.array(
                [[config.weight_vel_x], [config.weight_vel_y], [config.weight_vel_z]]
            ))
        
        # This if statement handles the edge case, where we have no prior estimation
        # for the velocity and there was no bbox found
        # (i.e. the offset is 0 in all 3 axis)
        if np.linalg.norm(fixed_offset) != 0:
            # Normalize the weighted offset (fixed_offset)
            fixed_offset /= np.linalg.norm(fixed_offset)
            # Multiply with the magnitude of the target velocity
            velocity = np.multiply(fixed_offset, config.uav_velocity)
            # Check if the calculated velocity has the desired magnitude (specified in config)
            assert(config.uav_velocity - np.linalg.norm(velocity) < config.eps)
        else:
            velocity = np.zeros([3, 1])
        
        self.prev_vel = velocity
        return velocity
