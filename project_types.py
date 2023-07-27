from typing import Literal, Tuple, TypedDict, Optional, get_args

from constants import EGO_UAV_NAME, LEADING_UAV_NAME, STATUS_COLORS

# Define a Status_t (type) so you may use a str to define the status when the coSimulator exits
# but have the status be an int as it is expected
Status_t = Literal["Error", "Running", "Time's up", "LeadingUAV lost", "EgoUAV and LeadingUAV collision", "EgoUAV collision", "LeadingUAV collision"]

# Make sure that if I update the config file in the future, I will be reminded to also update the messages
# defined is Status_t
if EGO_UAV_NAME != "EgoUAV" or LEADING_UAV_NAME != "LeadingUAV":
    Warning("Noticed that you changed the names of the EgoUAV but forgot to update the Status_t type defined in project_types.py, or maybe just this if statement!")

Filter_t = Literal["KF", "None"]
Motion_model_t = Literal["CA", "CV"]
Path_version_t = Literal["v0", "v1", "v2"]
Movement_t = Literal["Random", "Path"]

def _map_to_status_code(status: Status_t) -> int:
    """
    Maps the status to the appropriate integer code.
    - status = Error => -1
    - status = Running -> 0
    - status = Time's up -> 1
    - status = EgoUAV and LeadingUAV collision -> 2
    - status = EgoUAV collision -> 3
    - status = LeadingUAV collision -> 4

    Returns:
    The code
    """
    return get_args(Status_t).index(status) - 1

def map_status_to_color(status: Status_t) -> str:
    return STATUS_COLORS[get_args(Status_t).index(status)]

class Statistics_t(TypedDict):
    dist_mse: float
    lead_vel_mse: Optional[float]
    avg_true_dist: float

class ExtendedCoSimulatorConfig_t(TypedDict):
    uav_velocity: float
    score_threshold: float
    max_vel: Tuple[float, float, float]
    min_vel: Tuple[float, float, float]
    weight_vel: Tuple[float, float, float]
    sim_fps: int
    simulation_time_s: int
    camera_fps: int
    infer_freq_Hz: int
    filter_freq_Hz: int
    filter_type: Filter_t
    motion_model: Motion_model_t
    use_pepper_filter: bool
    leadingUAV_update_vel_interval_s: int
    max_time_lead_is_lost_s: int
    status: Status_t
    frame_count: int
    dist_mse: float
    lead_vel_mse: Optional[float]
    avg_true_dist: float