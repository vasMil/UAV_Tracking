import math

import numpy as np

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
