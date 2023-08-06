from typing import List, Tuple, Dict, Literal, Union
import os

import pickle
import json
import numpy as np

from models.FrameInfo import FrameInfo
from project_types import Status_t, ExtendedCoSimulatorConfig_t, map_status_to_int, map_status_to_binary_bool

def get_folders_in_path(path: str) -> List[str]:
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(os.path.join(path, entry.name))
    return folders

def folder_to_info(path: str) -> Tuple[List[FrameInfo], ExtendedCoSimulatorConfig_t, Status_t]:
    pkl_file = os.path.join(path, "log.pkl")
    json_file = os.path.join(path, "config.json")
    with open(pkl_file, 'rb') as f:
        frames_info: List[FrameInfo] = pickle.load(file=f)

    with open(json_file, 'r') as f:
        extended_config: ExtendedCoSimulatorConfig_t = json.load(fp=f)
    
    status: Status_t = extended_config["status"]
    return (frames_info, extended_config, status)

def extract_recording_data(recordings_path: str,
                           constant_key: Literal["uav_velocity", "infer_freq_Hz"],
                           constant_value: Union[int, float],
                           mode: Literal["mean", "binary", "all"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
    - recordings_path: The root folder path with all the folders for each recording.
    - constant_key: An ExtendedCoSimulatorConfig_t key, goes hand to hand with the constant_value.\
        All config[constant_key] != constant_value will be ignored.
    - constant_value: The value with which all config[constant_key] should be equal to.
    - mode: Decide whether to return the mean for each unique value.

    Returns:
    A Tuple containing information extracted from the desired recordings (i.e. those with config[constant_key] == constant_value).
    This Tuple contains "parallel" arrays that depending on the selected mode will either be contain the mean of or all the info \
        concerning the unique values that should not be constant.\
            If constant_key is "uav_velocity" then all different unique values will be "infer_freq_Hz".
    1. The values for the key, whose values are not constant.
    2. The statusses in the form derived by the mode.
    3. The avg_true_dists between the two UAVs, for each value.
    4. The sim_times, for each value.

    Note: The statuses returned (i.e. the second np.ndarray) can have 3 types of elements:\
    - If mode is "all" integer values to which Status_t is mapped.\
    - If mode is "mean" float values in range [0,1], representing the success rate.\
    - If mode is "binary" bool values to which Binary_status_t is mapped.
    """
    folders = get_folders_in_path(recordings_path)
    n = len(folders)
    x = np.zeros(n)
    status_codes = np.zeros(n, dtype=np.int64)
    avg_true_dists = np.zeros(n, dtype=np.float64)
    sim_times = np.zeros(n, dtype=np.float64)
    var_key = "infer_freq_Hz" if constant_key == "uav_velocity" else "uav_velocity"

    # Load data and extract useful info
    i = 0
    for folder in folders:
        _, config, status = folder_to_info(folder)
        if config[constant_key] != constant_value:
            continue
        x[i] = config[var_key]
        status_codes[i] = map_status_to_int(status)
        avg_true_dists[i] = config["avg_true_dist"]
        sim_times[i] = config["frame_count"]/config["camera_fps"]
        i += 1
    x = x[:i]
    status_codes = status_codes[:i]
    avg_true_dists = avg_true_dists[:i]
    sim_times = sim_times[:i]

    if mode == "all":
        return (x, status_codes, avg_true_dists, sim_times)

    # Convert Status_t to success rate and
    # Calculate the mean for the other variables
    unq_x = np.unique(x)
    unq_succ_rate = np.zeros(len(unq_x))
    unq_avg_true_dists = np.zeros(len(unq_x))
    unq_sim_times = np.zeros(len(unq_x))
    for i, x_val in enumerate(unq_x):
        mask = (x == x_val)
        unq_avg_true_dists[i] = avg_true_dists[mask].mean()
        unq_sim_times[i] = sim_times[mask].mean()
        unq_succ_rate[i] = np.vectorize(lambda sc: map_status_to_binary_bool(sc)
                                        )(status_codes[mask]).mean()
    if mode == "binary":
        unq_succ_rate = (unq_succ_rate >= 0.5).astype(np.bool_)
    
    return (unq_x, unq_succ_rate, unq_avg_true_dists, unq_sim_times)
