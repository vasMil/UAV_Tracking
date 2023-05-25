import numpy as np

from GlobalConfig import GlobalConfig as config

class Controller():
    def __init__(self) -> None:
        self.prev_vel = np.zeros([3, 1])

    def step(self, X_meas: np.ndarray, dt: float) -> np.ndarray:
        """
        Args:
        X_meas: A 6x1 column vector with the current measured state of the system.
        dt: The time interval between two consecutive step calls.

        Returns:
        A 6x1 column vector with the controller's target position and velocity.
        """
        # Keep only first 3 elements od the state measurement
        # since these are the position offsets.
        measurement = X_meas[0:3]

        if np.all(measurement == np.zeros([3, 1])):
            # Using the previous estimation for the velocity "predict"
            # the expected position of the leadingUAV.
            measurement = self.prev_vel * dt
            fixed_measurement = measurement
        else:
            # It is important to preserve the LeadingUAV in our FOV.
            # Thus you do an element-wise multiplication with the weights,
            # in order to favour moving on the z axis.
            fixed_measurement = np.multiply(measurement, np.array(
                [[config.weight_vel_x], [config.weight_vel_y], [config.weight_vel_z]]
            ))
        
        # Normalize the weighted offset
        meas_magn = np.linalg.norm(fixed_measurement)
        if meas_magn != 0:
            fixed_measurement /= meas_magn
            # Multiply with the magnitude of the target velocity
            velocity = fixed_measurement * config.uav_velocity
            assert(config.uav_velocity - np.linalg.norm(velocity) < config.eps)
        else:
            velocity = np.zeros([3, 1])
        
        self.prev_vel = velocity
        return np.vstack([measurement, velocity])
    