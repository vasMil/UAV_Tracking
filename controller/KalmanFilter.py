from typing import Optional, Literal

import numpy as np

class KalmanFilter():
    def __init__(self,
                 X_init: np.ndarray,
                 P_init: np.ndarray,
                 dt: float,
                 acc_std: float = 1.,
                 motion_model: Literal["CV", "CA"] = "CA",
                 forget_factor_a: float = 1.
            ) -> None:
        # Since we only measure the position of the target, the shape of the
        # measurement is
        meas_shape = [3,1]
        self.meas_shape = meas_shape

        # Using the motion model we may also know the shape of the state vector
        x_shape = [9,1] if motion_model == "CA" else [6,1]
        self.x_shape = x_shape

        # Time constant variables
        self.Q = np.diag(np.ones([1, x_shape[0]]).squeeze())           # Process Noise Covariance Matrix

        # If using the constant velocity model of motion add the
        # noise introduced since we model the acceleration of the
        # LeadingUAV as a Discrete White Noise.
        if motion_model == "CV":
            up_q = np.diag([(dt**3)/2, (dt**3)/2, (dt**3)/2], k=3)
            up_q = np.diag([(dt**4)/4, (dt**4)/4, (dt**4)/4, dt**2, dt**2, dt**2]) + up_q + up_q.T
            self.Q += up_q*(acc_std**2)

        self.R = np.array([[21.934, 2.743,  1.497],         # Measurement Covariance Matrix
                           [2.743,  58.147, 0.407],
                           [1.497,  0.407,  9.53 ]])

        self.F = np.eye(x_shape[0], x_shape[0])             # Motion model matrix
        self.F += np.eye(x_shape[0], x_shape[0], 3)*dt
        if motion_model == "CA":
            self.F += np.eye(9,9,6)*(dt**2)/2           
        
        self.B = np.zeros([6, 6])                           # Control Matrix (not important)
        self.C = np.eye(3, x_shape[0])                      # Transformation matrix, that maps the measurement vector to the correct dimension.
        self.H = np.eye(3, x_shape[0])                      # Transformation matrix, that maps the state vactor to the measurement vector.
        self.I = np.eye(x_shape[0], x_shape[0])             # The identity matrix

        # Previous state matrices
        self.X_prev = self.crop_init_matrix(X_init)         # Previous state matrix (at time t-1)
        self.P_prev = self.crop_init_matrix(P_init)         # Previous process covariance matrix (at time t-1)
        
        # Noise vectors
        self.wt = np.zeros([x_shape[0], 1])                 # Additive noise to the predicted state (not important)
        self.zt = np.zeros([meas_shape[0], 1])              # Additive noise to the measurement state (not important)

        # Yet unknown matrices
        self.K = np.zeros([x_shape[0], meas_shape[0]])      # Kalman Gain
        self.X_curr = np.zeros(x_shape)                     # Current state matrix (at time t)
        self.P_curr = np.zeros([x_shape[0], x_shape[0]])    # Current process covariance matrix (at time t)
        self.X_pred = np.zeros(x_shape)                     # Predicted state matrix (at time t)
        self.P_pred = np.zeros([x_shape[0], x_shape[0]])    # Predicted process covariance matrix (at time t)

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
            pcurr = (self.I - (self.K @ self.H)) @ self.P_pred
            if force_symmetry:
                return (pcurr + pcurr.T) / 2
            return pcurr
        else: # method == "Joseph"
            pcurr = (self.I - (self.K @ self.H))
            return pcurr @ self.P_pred @ pcurr.T + self.K @ self.R @ self.K.T

    def step(self, X_meas: Optional[np.ndarray]) -> np.ndarray:
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
        # Predict the new state
        self.X_pred = self.F @ self.X_prev + self.wt
        self.P_pred = self.forget_factor_a_squared * self.F @ self.P_prev @ self.F.T + self.Q
        
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

    def pad_measurement(self, meas: np.ndarray) -> np.ndarray:
        """
        Adds the correct amount of padding to the measurement\
        meas, so it is in the expected shape for the step function.

        Args:
        meas: An mx1 measurement.

        Returns:
        A 6x1 vector if motion_model of the KF is "CV" or\
        a 9x1 vector if the motion_model is "CA"
        """
        m = meas.shape[0]
        n_rows = self.x_shape[0]
        num_pad_rows = n_rows-m
        if num_pad_rows < 0:
            raise Exception("Unexpected number of rows in meas")
        return np.pad(meas, ((0,num_pad_rows),(0,0)))

    def crop_init_matrix(self, mat: np.ndarray) -> np.ndarray:
        m = self.x_shape[0]
        return mat[:m, :m]
