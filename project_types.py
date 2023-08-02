from typing import Literal, List, Tuple, Dict, TypedDict, Mapping, Optional, Any, get_args

import torch

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

class Bbox_dict_t(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    label: int
    img_name: Optional[str]
    img_height: Optional[int]
    img_width: Optional[int]

class Bbox_dict_for_nn_t(TypedDict):
    boxes: torch.Tensor
    labels: torch.LongTensor

class BoundBoxDataset_Item(TypedDict):
    image: torch.Tensor
    bounding_box: Bbox_dict_for_nn_t

class Losses_dict_t(TypedDict):
    epoch: int
    train: float
    val: float

class Checkpoint_t(TypedDict):
    epoch: int
    model_state_dict: Mapping[str, Any]
    optimizer_state_dict: Dict[Any, Any]
    scheduler_state_dict: Optional[dict[Any, Any]]
    losses: List[Losses_dict_t]
    # List[(Epoch: int, mAP: Dict[str, float],)]
    mAPs: List[Tuple[int, Dict[str, float]]]
    training_time: float
