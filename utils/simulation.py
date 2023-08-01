from typing import List
import math

import numpy as np
import airsim

from models.UAV import UAV
from project_types import Path_version_t

def sim_calculate_angle(uav_source: UAV, uav_target: UAV) -> float:
    source_pos = np.expand_dims(
        uav_source.simGetObjectPose().position.to_numpy_array(),
        axis=1
    )
    # source_pos = vector_transformation(*(uav_source.getPitchRollYaw()), vec=source_pos, to=True)
    target_pos = np.expand_dims(
        uav_target.simGetObjectPose().position.to_numpy_array(),
        axis=1
    )
    # target_pos = vector_transformation(*(uav_source.getPitchRollYaw()), vec=target_pos, to=True)
    dist = target_pos - source_pos

    rad = 0. if dist[0] == 0 else math.atan(dist[1]/dist[0])
    deg = math.degrees(rad)
    if deg > 0 and dist[1] < 0:
        deg -= 180
    elif deg < 0 and dist[1] > 0:
        deg += 180
    return deg

def createPathFromPoints(path: List[airsim.Vector3r]) -> List[airsim.Vector3r]:
    """
    Given a list of points. Sum each point with the previous, in order to get
    the actual points on the coordinate system and allow for a smooth movement
    on the resulted path.
    """
    for i, _ in enumerate(path):
        if i == 0: continue
        path[i] += path[i-1]

    return path

def getSpiralPath(radius: float,
                  height_limit: float,
                  num_points: int,
                  rotational_velocity_z: float
    ) -> List[airsim.Vector3r]:
    path = []

    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        # Calculate the rotational movement around the z-axis
        z_angle = rotational_velocity_z * i / num_points
        rotation_matrix = np.array([[math.cos(z_angle), -math.sin(z_angle)], 
                                    [math.sin(z_angle), math.cos(z_angle)]])
        x, y = np.dot(rotation_matrix, [x, y])

        z = height_limit * i / num_points
        path.append(airsim.Vector3r(x, y, -z))

    for i in reversed(range(num_points)):
        if i == 0: continue
        path[i] -= path[i-1]

    return path

def getSinusoidalPath(num_points: int = 1000,
                      x_length: float = 50.,
                      y_amplitude: float = 10.,
                      z_amplitude: float = 10.,
                      rotational_velocity_y: float = 2*math.pi,
                      rotational_velocity_z: float = 2*math.pi
    ) -> List[airsim.Vector3r]:
    path = []
    for i in range(num_points):
        angle_y = rotational_velocity_y * i / num_points
        angle_z = rotational_velocity_z * i / num_points
        x = x_length * i / num_points
        y = y_amplitude * math.cos(angle_y)
        z = z_amplitude * math.sin(angle_z)
        path.append(airsim.Vector3r(x, y, -z))

    for i in reversed(range(num_points)):
        if i == 0: continue
        path[i] -= path[i-1]

    return path

def getTestPath(start_pos: airsim.Vector3r, version: Path_version_t = "v2") -> List[airsim.Vector3r]:
    """
    Get a predefined path, given the starting position of the vehicle.
    """
    path_v0 = [
        start_pos,
        airsim.Vector3r(10, 0, 0),    # Test x axis - moving forward
        airsim.Vector3r(2, 10, 0),    # Test y axis - moving fast right
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(2, -10, 0),   # Test y axis - moving fast left
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, 10, 0),    # Test y axis - moving faster right
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, -10, 0),   # Test y axis - moving faster left
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
    ]
    path_v0 += getSpiralPath(radius=4, height_limit=5, num_points=100, rotational_velocity_z=4*math.pi)

    path_v1 = [start_pos, airsim.Vector3r(20, 0, 0)]
    path_v1 += getSinusoidalPath(rotational_velocity_y=1*math.pi, rotational_velocity_z=0)
    path_v1 += [airsim.Vector3r(20, 0, 0)]
    path_v1 += getSinusoidalPath(rotational_velocity_y=2*math.pi, rotational_velocity_z=0)
    path_v1 += [airsim.Vector3r(20, 0, 0)]
    path_v1 += getSinusoidalPath(rotational_velocity_y=3*math.pi, rotational_velocity_z=0)
    path_v1 += [airsim.Vector3r(20, 0, 0)]

    path_v2 = [start_pos, airsim.Vector3r(20, 0, 0)]
    path_v2 += getSinusoidalPath(rotational_velocity_y=1*math.pi, rotational_velocity_z=1*math.pi)
    path_v2 += [airsim.Vector3r(20, 0, 0)]
    path_v2 += getSinusoidalPath(rotational_velocity_y=2*math.pi, rotational_velocity_z=2*math.pi)
    path_v2 += [airsim.Vector3r(20, 0, 0)]
    path_v2 += getSinusoidalPath(rotational_velocity_y=3*math.pi, rotational_velocity_z=3*math.pi)
    path_v2 += [airsim.Vector3r(20, 0, 0)]

    str8line = [start_pos, airsim.Vector3r(500, 0, 0)]

    return createPathFromPoints(path_v0 if version == "v0" else\
                                path_v1 if version == "v1" else\
                                path_v2 if version == "v2" else
                                str8line
            )
