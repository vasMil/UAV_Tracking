import os

import json
import pickle

from project_types import ExtendedCoSimulatorConfig_t
from models.FrameInfo import FrameInfo

if __name__ == '__main__':
    root_paths = ["recordings/20230801_195333",
                  "recordings/20230801_200731"]

    configs: list[ExtendedCoSimulatorConfig_t] = []
    infos: list[list[FrameInfo]] = []
    for root_path in root_paths:
        with open(os.path.join(root_path, "config.json"), 'r') as f:
            configs.append(json.load(f))
        with open(os.path.join(root_path, "log.pkl"), 'rb') as f:
            infos.append(pickle.load(f))

    for key in configs[0].keys():
        if any(config[key] != configs[0][key] for config in configs[1:]):
            print(f"{key} missmatch")
    
    for key in ["sim_lead_pos", "sim_ego_pos", "sim_lead_vel", "sim_ego_vel", "sim_angle_deg", "bbox_score", "extra_timestamp"]:
        if key == "extra_timestamp":
            intervals = []
            for info_list in infos:
                timestamps = [(frame_info[key] 
                               - info_list[0][key]
                               ) for frame_info in info_list
                ]
                intervals.append(timestamps)
            for interval_list in intervals[1:]:
                for i, interval in enumerate(interval_list):
                    if interval != intervals[0][i]:
                        print(f"Interval {i} missmatch")
                        break
            continue

        for info_list in infos[1:]:
            for i, info in enumerate(info_list):
                if(info[key] != infos[0][i][key]):
                    print(f"{key} {i} missmatch")
                    break

