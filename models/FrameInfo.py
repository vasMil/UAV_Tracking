from typing import List, TypedDict, Tuple, Optional

class GroundTruthFrameInfo(TypedDict):
    frame_idx: int
    timestamp: int
    egoUAV_position: Tuple[float, float, float]
    egoUAV_orientation_quartanion: Tuple[float, float, float, float]
    egoUAV_velocity: Tuple[float, float, float]
    leadingUAV_position: Tuple[float, float, float]
    leadingUAV_orientation_quartanion: Tuple[float, float, float, float]
    leadingUAV_velocity: Tuple[float, float, float]
    angle_deg: float

class EstimatedFrameInfo(TypedDict):
    egoUAV_position: Optional[Tuple[float, float, float]]
    egoUAV_target_velocity: Optional[Tuple[float, float, float]]
    leadingUAV_position: Optional[Tuple[float, float, float]]
    leadingUAV_velocity: Optional[Tuple[float, float, float]]
    angle_deg: Optional[float]
    still_tracking: Optional[bool]

class FrameInfo(TypedDict):
    bbox_score: Optional[float]
    # LeadingUAV position
    sim_lead_pos: Tuple[float, float, float]
    est_lead_pos: Optional[Tuple[float, float, float]]
    err_lead_pos: Optional[Tuple[float, float, float]]

    # EgoUAV position
    sim_ego_pos: Tuple[float, float, float]
    est_ego_pos: Optional[Tuple[float, float, float]]
    err_ego_pos: Optional[Tuple[float, float, float]]

    # LeadingUAV velocity
    sim_lead_vel: Tuple[float, float, float]
    est_lead_vel: Optional[Tuple[float, float, float]]
    err_lead_vel: Optional[Tuple[float, float, float]]

    # EgoUAV velocity
    sim_ego_vel: Tuple[float, float, float]
    target_ego_vel: Optional[Tuple[float, float, float]]
    err_ego_vel: Optional[Tuple[float, float, float]]

    # Angle bewteen the two UAV's
    sim_angle_deg: float
    est_angle_deg: Optional[float]
    err_angle: Optional[float]

    # Extra information will not be included on the frame but will
    # get pickled into the pkl file created by the logger
    extra_frame_idx: int
    extra_timestamp: int
    extra_leading_orientation_quartanion: Tuple[float, float, float, float]
    extra_ego_orientation_quartanion: Tuple[float, float, float, float]
    extra_still_tracking: Optional[bool]