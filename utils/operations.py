import math

import numpy as np

def rotate_to_yaw(yaw_rad: float, vect: np.ndarray) -> np.ndarray:
    """
    Args:
    yaw_rad: Yaw angle, at which to rotate the vector.
    vect: The vector to rotate.

    Returns:
    The rotated vector.
    """
    rot_mat = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                        [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                        [0,                                  0, 1]
                    ], dtype=np.float64)
    return rot_mat @ vect

def rotate_to_roll(roll_rad: float, vect: np.ndarray) -> np.ndarray:
    """
    Args:
    roll_rad: Roll angle, at which to rotate the vector.
    vect: The vector to rotate.

    Returns:
    The rotated vector.
    """
    rot_mat = np.array([[1,                  0,                   0],
                        [0, math.cos(roll_rad), -math.sin(roll_rad)],
                        [0, math.sin(roll_rad),  math.cos(roll_rad)]
                    ], dtype=np.float64)
    return rot_mat @ vect

def rotate_to_pitch(pitch_rad: float, vect: np.ndarray) -> np.ndarray:
    """
    Args:
    pitch_rad: Pitch angle, at which to rotate the vector.
    vect: The vector to rotate.

    Returns:
    The rotated vector.
    """
    rot_mat = np.array([[math.cos(pitch_rad),  0, math.sin(pitch_rad)],
                        [0,                    1,                   0],
                        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
                        ], dtype=np.float64)
    return rot_mat @ vect

def rotate3d(pitch_deg: float, roll_deg: float, yaw_deg: float, point: np.ndarray) -> np.ndarray:
    """
    Rotate the point by some desired angle on each axis of the 3D space.
    Args:
    pitch_deg: Pitch angle, at which to rotate the vector.
    roll_deg: Roll angle, at which to rotate the vector.
    yaw_deg: Yaw angle, at which to rotate the vector.
    point: The vector to rotate.

    Returns:
    The rotated vector.
    """
    pitch_rad = math.radians(pitch_deg) # θ
    roll_rad = math.radians(roll_deg) # ψ
    yaw_rad = math.radians(yaw_deg) # φ
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

def vector_transformation(pitch_deg: float,
                          roll_deg: float,
                          yaw_deg: float,
                          vec: np.ndarray,
                          to: bool = False
                    ) -> np.ndarray:
    """
    Moves vec that is in a coordinate frame rotated by pitch_deg,
    roll_deg, yaw_deg. To the original coordinate frame.

    Args:
    pitch_deg: The pitch angle at which the original coordinate frame has been rotated
    roll_deg: The roll angle at which the original coordinate frame has been rotated
    yaw_deg: The yaw angle at which the original coordinate frame has been rotated
    vec: A (3x1) column vector inside the coordinate frame defined by the previous angles

    Returns:
    A (3x1) column vector that contains the projection on each one of the three axis of the
    input vector vec
    """
    # Create a unit vectors for each axis of the two coord systems
    # ix, iy, iz = np.hsplit(np.eye(3,3), 3)
    ix = np.array([1, 0, 0])
    iy = np.array([0, 1, 0])
    iz = np.array([0, 0, 1])
    ix_bar = rotate3d(pitch_deg, roll_deg, yaw_deg, ix)
    iy_bar = rotate3d(pitch_deg, roll_deg, yaw_deg, iy)
    iz_bar = rotate3d(pitch_deg, roll_deg, yaw_deg, iz)

    # Create transformation matrix
    T_mat = np.array([[np.dot(ix_bar, ix), np.dot(ix_bar, iy), np.dot(ix_bar, iz)],
                      [np.dot(iy_bar, ix), np.dot(iy_bar, iy), np.dot(iy_bar, iz)],
                      [np.dot(iz_bar, ix), np.dot(iz_bar, iy), np.dot(iz_bar, iz)]])

    if to:
        return T_mat @ vec
    else:
        return T_mat.T @ vec
