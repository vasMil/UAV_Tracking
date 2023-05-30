import math

import numpy as np

def rotate_to_yaw(yaw_rad: float, vect: np.ndarray) -> np.ndarray:
    """
    Args:
    yaw_rad: Yaw angle, at which the axis is rotated at.
    vect: The vector to rotate.

    Returns:
    The rotated vector.
    """
    rot_mat = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                        [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                        [0,                                  0, 1]
                    ], dtype=np.float64)
    return rot_mat @ vect