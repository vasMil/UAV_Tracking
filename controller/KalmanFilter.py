from typing import Optional

import numpy as np

class KalmanFilter():
    def __init__(self,
                 X_init: np.ndarray,
                 P_init: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray
            ) -> None:
        # Time constant variables
        self.Q = Q                      # Process Noise Covariance Matrix
        self.R = R                      # Measurement Covariance Matrix
        # self.F = np.eye(6, 6) + np.eye(6, 6, 3)*dt
        # self.B = np.zeros([6, 6])
        self.C = np.eye(3, 6)
        self.H = np.eye(3, 6)           # Transformation matrix, that maps the state vactor to the measurement vector.

        # Previous state matrices
        self.X_prev = X_init            # Previous state matrix (at time t-1)
        self.P_prev = P_init            # Previous process covariance matrix (at time t-1)
        
        # Noise matrices
        self.wt = np.zeros([6, 1])      # Predicted state noise matrix
        self.zt = np.zeros([3, 1])

        # Yet unknown matrices
        self.K = np.zeros([6, 3])       # Kalman Gain
        self.X_curr = np.zeros([6, 1])  # Current state matrix (at time t)
        self.P_curr = np.zeros([6, 1])  # Current process covariance matrix (at time t)
        self.X_pred = np.zeros([6, 1])  # Predicted state matrix (at time t)
        self.P_pred = np.zeros([6, 6])  # Predicted process covariance matrix (at time t)

    def step(self, X_meas: Optional[np.ndarray], dt: float) -> np.ndarray:
        """
        Update the estimation and output the new prediction.

        Args:
        - dt: The time interval between two consecutive calls.
        - X_meas: The 6x1 matrix that contains the measurements for
        the current state of the system | None. If None only the prediction
        step will be performed.

        Returns:
        The 6x1 state matrix as an numpy ndarray.
        """
        # Create matrix A (the one that models our system)
        self.F = np.eye(6, 6) + np.eye(6, 6, 3)*dt

        # Predict the new state
        self.X_pred = self.F @ self.X_prev + self.wt
        self.P_pred = self.F @ self.P_prev @ self.F.T + self.Q
        
        if X_meas is not None:
            # Calculate the Kalman Gain
            self.K = self.P_pred @ self.H.T @ np.linalg.inv(self.H @ self.P_pred @ self.H.T + self.R)
            # Reshape the measurement
            Y_curr = self.C @ X_meas + self.zt
            self.X_curr = self.X_pred + self.K @ (Y_curr - (self.H @ self.X_pred))
            # Since we have calculated and output the current estimation of the
            # Kalman filter for timestep t, we now may re-calibrate the process
            # covariance matrix P and continue to the next timestep. That is
            # assign t -> t-1 (curr variables to prev)
            self.P_curr = (np.eye(6, 6) - (self.K @ self.H)) @ self.P_pred
        else:
            self.X_curr = self.X_pred
            self.P_curr = self.P_pred
        
        # Update step
        self.X_prev = self.X_curr
        self.P_prev = self.P_curr

        # Convert the output state of the Kalman Filter to a velocity, which will be
        # applied for dt seconds
        return self.X_curr
