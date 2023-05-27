import numpy as np

from controller.Controller import Controller

class KalmanFilter(Controller):
    def __init__(self,
                 X_init: np.ndarray,
                 P_init: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray
            ) -> None:
        super().__init__()
        # Time constant variables
        self.Q = Q
        self.R = R
        # self.A = np.eye(6, 6) + np.eye(6, 6, 3)*dt
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
        # self.P_curr = np.zeros([6, 1])  # Current process covariance matrix (at time t)
        self.X_pred = np.zeros([6, 1])  # Predicted state matrix (at time t)
        self.P_pred = np.zeros([6, 6])  # Predicted process covariance matrix (at time t)

    def step(self, X_meas: np.ndarray, dt: float) -> np.ndarray:
        """
        Using the preparation from prepare_step and the provided measurement.
        Update the estimation and output the new prediction.

        Args:
        dt
        X_meas: The 6x1 matrix that contains the measurements for
        the current state of the system

        Returns:
        The velocity, extracted out of the estimated state of the system.
        This estimate is a combination of the predicted state X_pred and the
        measured state X_meas.
        It is a 3x1 matrix of form: [vx vy vz]'
        """
        # Create matrix A (the one that models our system)
        self.A = np.eye(6, 6) + np.eye(6, 6, 3)*dt

        # Predict the new state
        self.X_pred = self.A @ self.X_prev + self.wt
        self.P_pred = self.A @ self.P_prev @ self.A.T + self.Q
        
        # Calculate the Kalman Gain
        self.K = self.P_pred @ self.H.T @ (self.H @ self.P_pred @ self.H.T + self.R)
        print(f"X_pred: {self.X_pred}")
        # If there is no valid measurement, just perform the prediction step
        # else use the Kalman Gain to incorporate the measurement aswell.
        if np.any(X_meas != np.zeros([6, 1])):
            # Reshape the measurement
            Y_curr = self.C @ X_meas + self.zt
            self.X_curr = self.X_pred + self.K @ (Y_curr - (self.H @ self.X_pred))
        else:
            self.X_curr = self.X_pred

        # Since we have calculated and output the current estimation of the
        # Kalman filter for timestep t, we now may re-calibrate the process
        # covariance matrix P and continue to the next timestep. That is
        # assign t -> t-1 (curr variables to prev)
        self.P_prev = (np.eye(6, 6) - (self.K @ self.H)) @ self.P_pred
        self.X_prev = self.X_curr
        return self.X_curr
