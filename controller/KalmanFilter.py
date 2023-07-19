from typing import Optional, Literal

import numpy as np

class KalmanFilter():
    def __init__(self,
                 X_init: np.ndarray,
                 P_init: np.ndarray,
                 R: np.ndarray,
                 Q: Optional[np.ndarray] = None,
                 forget_factor_a: float = 1
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

        # Fading memory implementation
        self.forget_factor_a_squared = forget_factor_a**2

    def _calculate_P_curr(self,
                          method: Literal["computational_efficiency", "Joseph"],
                          force_symmetry: Optional[bool] = None
        ) -> np.ndarray:
        """
        A helper function that computes the updated P matrix (P_k <-> P_t <-> P_curr),
        using a specified method.

        Args:
        method: The method to use in order to calculate the matrix.
            - computational_efficieny: Utilizes the formula (I - K_k @ H_k) @ P^-_k
            - Joseph: Utilizes the most stable version, formulated by Peter Joseph:
            (I - K_k @ H_k) @ P^-_k @ (I - K_k @ H_k).T + (K_k @ R_k @ K_k.T)
            P^-_k: The prediction of P matrix at timestep k (current timestep)
        
        force_symmetry: Forces the matrix return to be symmetric. Since the Joseph\
                        method always returns a symmetric matrix, this argument is\
                        only important when selecting computational_efficiency as the\
                        method.
        """
        if method == "computational_efficiency":
            pcurr = (np.eye(6, 6) - (self.K @ self.H)) @ self.P_pred
            if force_symmetry:
                return (pcurr + pcurr.T) / 2
            return pcurr
        else: # method == "Joseph"
            pcurr = (np.eye(6, 6) - (self.K @ self.H))
            return pcurr @ self.P_pred @ pcurr.T + self.K @ self.R @ self.K.T

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

        # State process covariance matrix
        if self.Q is None:
            # 
            up_q = np.diag([(dt**3)/2, (dt**3)/2, (dt**3)/2], k=3)
            Q = np.diag([(dt**4)/4, (dt**4)/4, (dt**4)/4, dt**2, dt**2, dt**2]) + up_q + up_q.T
            # fictitious process noise
            Q += np.diag([1, 1, 1, 1, 1, 1])
        else:
            Q = self.Q

        # Predict the new state
        self.X_pred = self.F @ self.X_prev + self.wt
        self.P_pred =  self.forget_factor_a_squared * self.F @ self.P_prev @ self.F.T + Q
        
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
            self.P_curr = self._calculate_P_curr(method="Joseph")
        else:
            self.X_curr = self.X_pred
            self.P_curr = self.P_pred
        
        # Update step
        self.X_prev = self.X_curr
        self.P_prev = self.P_curr

        # Convert the output state of the Kalman Filter to a velocity, which will be
        # applied for dt seconds
        return self.X_curr
