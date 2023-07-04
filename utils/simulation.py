from typing import List
import math

import numpy as np
import airsim

from models.UAV import UAV

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


def getSquarePathAroundPoint(pointx: float,
                             pointy: float,
                             pointz: float,
                             coord_frame_offset: airsim.Vector3r,
                             square_width: float
                        ) -> List[airsim.Vector3r]:
    g_init_point = airsim.Vector3r(pointx + square_width/2, pointy, pointz)
    init_point = g_init_point - coord_frame_offset

    ret_l = [airsim.Vector3r(0, -square_width/2, pointz),
             airsim.Vector3r(-square_width, -square_width/2, pointz),
             airsim.Vector3r(-square_width, square_width/2, pointz),
             airsim.Vector3r(0, square_width/2, pointz),
             airsim.Vector3r(0, 0, pointz)
            ]
    
    for point in ret_l:
        point += init_point

    return ret_l


def getTestPath(start_pos: airsim.Vector3r) -> List[airsim.Vector3r]:
    # Define list of points.
    # Each point is defined around (0, 0, 0).
    # This allows us to easily think about the angles at which the vehicle
    # will move along, if it was at point (0, 0, 0) and thus we can design
    # with ease difficult paths.
    path = [
        start_pos,
        airsim.Vector3r(10, 0, 0),    # Test x axis - moving forward
        # airsim.Vector3r(-10, 3, 0), # Test x axis - moving backwards, this is not allowed and thus will be skipped

        airsim.Vector3r(2, 10, 0),    # Test y axis - moving fast right
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(2, -10, 0),   # Test y axis - moving fast left
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, 10, 0),    # Test y axis - moving faster right
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, -10, 0),   # Test y axis - moving faster left

        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(2, 0, -10),   # Test z axis - moving fast up
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(2, 0, 10),    # Test z axis - moving fast down
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, 0, -10),   # Test z axis - moving faster up
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, 0, 10),    # Test z axis - moving faster down
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
    ]

    # Sum each point with the previous, in order to get
    # the actual points on the coordinate system and allow
    # for a smooth movement on the resulted path.
    for i, _ in enumerate(path):
        if i == 0: continue
        path[i] += path[i-1]

    return path
