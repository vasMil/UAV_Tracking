import math

from models.UAV import UAV

def sim_calculate_angle(uav_source: UAV, uav_target: UAV) -> float:
    source_pos = uav_source.simGetObjectPose().position
    target_pos = uav_target.simGetObjectPose().position
    disty = target_pos.y_val - source_pos.y_val
    distx = target_pos.x_val - source_pos.x_val
    rad = 0. if distx == 0 else math.atan(disty/distx)
    return math.degrees(rad)
