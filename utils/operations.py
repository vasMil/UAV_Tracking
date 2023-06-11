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

def rotate3d(pitch_deg: float, roll_deg: float, yaw_deg: float, point: np.ndarray) -> np.ndarray:
    # Convert angles to radians and get their opposites
    # since we need to project the point, which is measured in the
    # coordinate frame that is rotated by pitch, roll and yaw,
    # onto the original axis (i.e. pitch = roll = yaw = 0).
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)
    yaw_rad = math.radians(yaw_deg)
    rot_mat_pitch = np.array([[math.cos(pitch_rad),  0, math.sin(pitch_rad)],
                              [0,                    1,                   0],
                              [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
                        ], dtype=np.float64)
    rot_mat_roll = np.array([[1,                  0,                   0],
                             [0, math.cos(roll_rad), -math.sin(roll_rad)],
                             [0, math.sin(roll_rad),  math.cos(roll_rad)]
                        ], dtype=np.float64)
    rot_mat_yaw = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                            [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                            [0,                                  0, 1]
                        ], dtype=np.float64)
    rot_mat = rot_mat_yaw @ rot_mat_pitch @ rot_mat_roll
    return rot_mat @ point

def normalize_angle(angle_deg: float) -> float:
    angle_deg =  angle_deg % 360
    if (angle_deg > 180):
        angle_deg -= 360
    return angle_deg
