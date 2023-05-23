from typing import List, TypedDict, Tuple, Optional
import datetime, time
import os
import math

import torch
from torchvision.utils import save_image
import pickle
import matplotlib.pyplot as plt

from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from models.BoundingBox import BoundingBox
from GlobalConfig import GlobalConfig as config
from utils.image import add_bbox_to_image, add_angle_info_to_image
from utils.simulation import sim_calculate_angle

__all__ = (
    "Logger",
    "GraphLogs",
)

class FrameInfo(TypedDict):
    frame_idx: int
    egoUAV_position: Tuple[float, float, float]
    egoUAV_orientation_quartanion: Tuple[float, float, float, float]
    egoUAV_velocity: Tuple[float, float, float]
    leadingUAV_position: Tuple[float, float, float]
    leadingUAV_orientation_quartanion: Tuple[float, float, float, float]
    leadingUAV_velocity: Tuple[float, float, float]
    still_tracking: bool

class Logger:
    def __init__(self,
                 egoUAV: EgoUAV,
                 leadingUAV: LeadingUAV,
                 sim_fps: int,
                 simulation_time_s: int,
                 camera_fps: int,
                 infer_freq_Hz: int,
                 leadingUAV_update_vel_interval_s: int
            ) -> None:
        dt = datetime.datetime.now()

        self.egoUAV = egoUAV
        self.leadingUAV = leadingUAV
        self.sim_fps = sim_fps
        self.simulation_time_s = simulation_time_s
        self.camera_fps = camera_fps
        self.infer_freq_Hz = infer_freq_Hz
        self.leadingUAV_update_vel_interval_s = leadingUAV_update_vel_interval_s

        self.parent_folder = f"recordings/{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}{dt.second}"
        self.images_path = f"{self.parent_folder}/images"
        self.logfile = f"{self.parent_folder}/log.pkl"
        self.setup_file = f"{self.parent_folder}/setup.txt"
        os.makedirs(self.images_path)
        self.info_per_frame: List[FrameInfo] = []
        self.frame_cnt: int = 0

    def step(self, still_tracking: bool):
        # Collect data from simulation to variables
        ego_pos = self.egoUAV.simGetObjectPose().position
        ego_vel = self.egoUAV.simGetGroundTruthKinematics().linear_velocity
        ego_quart = self.egoUAV.simGetObjectPose().orientation
        lead_pos = self.leadingUAV.simGetObjectPose().position
        lead_vel = self.leadingUAV.simGetGroundTruthKinematics().linear_velocity
        lead_quart = self.leadingUAV.simGetObjectPose().orientation

        # Organize the variables above into a FrameInfo
        frame_info: FrameInfo = {
            "frame_idx": self.frame_cnt,
            "still_tracking": still_tracking,
            "egoUAV_position": (ego_pos.x_val, ego_pos.y_val, ego_pos.z_val),
            "egoUAV_orientation_quartanion": (ego_quart.x_val, ego_quart.y_val, ego_quart.z_val, ego_quart.w_val),
            "egoUAV_velocity": (ego_vel.x_val, ego_vel.y_val, ego_vel.z_val),
            "leadingUAV_position": (lead_pos.x_val, lead_pos.y_val, lead_pos.z_val),
            "leadingUAV_orientation_quartanion": (lead_quart.x_val, lead_quart.y_val, lead_quart.z_val, lead_quart.w_val),
            "leadingUAV_velocity": (lead_vel.x_val, lead_vel.y_val, lead_vel.z_val),
        }

        # Update the Logger state
        self.frame_cnt += 1
        self.info_per_frame.append(frame_info)

    def save_frame(self, frame: torch.Tensor, bbox: Optional[BoundingBox]):
        if bbox:
            # Calculate the ground truth angle
            sim_angle = sim_calculate_angle(self.egoUAV, self.leadingUAV)
            # Calculate the estimated angle
            estim_angle = self.egoUAV.get_yaw_angle_from_bbox(bbox)
            # Add info on the camera frame
            frame = add_bbox_to_image(frame, bbox)
            frame = add_angle_info_to_image(frame, estim_angle, sim_angle)
        save_image(frame, f"{self.images_path}/img_EgoUAV_{time.time_ns()}.png")

    def write_setup(self):
        with open(self.setup_file, 'w') as f:
            f.write(f"# The upper an lower limit for the velocity on each axis of both UAVs\n"
                    f"max_vx, max_vy, max_vz = {config.max_vx},  {config.max_vy},  {config.max_vz}\n"
                    f"min_vx, min_vy, min_vz = {config.min_vx}, {config.min_vy}, {config.min_vz}\n"
                    f"\n"
                    f"# The seed used for the random movement of the LeadingUAV\n"
                    f"leadingUAV_seed = {config.leadingUAV_seed}\n"
                    f"\n"
                    f"# The minimum score, for which a detection is considered\n"
                    f"# valid and thus is translated to EgoUAV movement.\n"
                    f"score_threshold = {config.score_threshold}\n"
                    f"\n"
                    f"# The magnitude of the velocity vector (in 3D space)\n"
                    f"uav_velocity = {config.uav_velocity}\n"
                    f"\n"
                    f"# The weights applied when converting bbox to move command\n"
                    f"weight_vel_x, weight_vel_y, weight_vel_z = {config.weight_vel_x}, {config.weight_vel_y}, {config.weight_vel_z}\n"
                    f"\n"
                    f"# Recording setup\n"
                    f"sim_fps = {self.sim_fps}\n"
                    f"simulation_time_s = {self.simulation_time_s}\n"
                    f"camera_fps = {self.camera_fps}\n"
                    f"infer_freq_Hz = {self.infer_freq_Hz}\n"
                    f"leadingUAV_update_vel_interval_s = {self.leadingUAV_update_vel_interval_s}\n"
            )

    def dump_logs(self):
        with open(self.logfile, 'wb') as f:
            pickle.dump(self.info_per_frame, f)


class GraphLogs:
    def __init__(self,
                 pickle_file: Optional[str] = None,
                 info_per_frame: Optional[List[FrameInfo]] = None
            ) -> None:
        if not pickle_file or info_per_frame:
            raise Exception("One of two arguments must be not none")
        if pickle_file and info_per_frame:
            raise Exception("You should only pass one of the two arguments")

        if pickle_file:
            with open(pickle_file, "rb") as f:
                self.info_per_frame: List[FrameInfo] = pickle.load(f)
        elif info_per_frame:
            self.info_per_frame = info_per_frame

    def graph_distance(self, sim_fps: Optional[int] = None):
        # Calculate the distance between the two UAVs for each frame
        dist = []
        for info in self.info_per_frame:
            ego_pos = info["egoUAV_position"]
            lead_pos = info["leadingUAV_position"]
            dist.append(
                math.sqrt((ego_pos[0] - lead_pos[0])**2 + (ego_pos[1] - lead_pos[1])**2 + (ego_pos[2] - lead_pos[2])**2)
            )
        mult_factor = 1 if not sim_fps else (1/sim_fps)
        x_axis = [x*mult_factor for x in range(0, len(self.info_per_frame))]
        plt.plot(x_axis, dist)
        plt.show()
