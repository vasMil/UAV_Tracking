from typing import List, Tuple
import os

import pickle
import json

from models.FrameInfo import FrameInfo
from project_types import Status_t, ExtendedCoSimulatorConfig_t

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
