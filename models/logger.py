from typing import List, Optional, Literal, Tuple, Callable
from operator import itemgetter
import datetime, time
import os

import torch
from torchvision.utils import save_image
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import plotext as tplt
import airsim
import json

from project_types import Status_t, Statistics_t, ExtendedCoSimulatorConfig_t
from config import DefaultCoSimulatorConfig
from constants import IMG_RESOLUTION_INCR_FACTOR,\
    MOVEMENT_PLOT_MIN_RANGE,\
    IMG_HEIGHT, IMG_WIDTH
from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from models.BoundingBox import BoundingBox
from utils.image import add_bbox_to_image, add_info_to_image, increase_resolution
from utils.simulation import sim_calculate_angle
from models.FrameInfo import FrameInfo, EstimatedFrameInfo, GroundTruthFrameInfo

__all__ = (
    "Logger",
    "GraphLogs",
)

class Logger:
    """
    A class that handles captured frames, by appending the bbox if there is one, along
    with ground truth and estimated information.

    It is important to note that the bbox displayed on the frame, along with the angle information
    is not exactly as viewed by the egoUAV. That is because the egoUAV does not have a bbox instantly.
    There is a small delay between the time at which the frame is captured and the time at which
    the result from inference can be converted to the velocity. That delay is dt = 1/inference_frequency.

    You should always keep in mind that the information added to each frame is the information that would
    be available if the inference was instant. In reality the EgoUAV will become aware of this "displayed"
    state after dt time and that is when all decisions will be applied.
    """
    def __init__(self,
                 egoUAV: EgoUAV,
                 leadingUAV: LeadingUAV,
                 config: DefaultCoSimulatorConfig = DefaultCoSimulatorConfig(),
                 folder: str = "recordings/",
                 display_terminal_progress: bool = True,
                 keep_frames: bool = False,
                 get_video: bool = True
            ) -> None:
        # Settings
        self.egoUAV = egoUAV
        self.leadingUAV = leadingUAV
        self.config = config
        self.keep_frames = keep_frames
        self.get_video = get_video

        # Folder and File names
        dt = datetime.datetime.now()
        self.parent_folder = os.path.join(folder ,f"{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}")
        self.images_path = f"{self.parent_folder}/images"
        self.logfile = f"{self.parent_folder}/log.pkl"
        self.setup_file = f"{self.parent_folder}/setup.txt"
        self.config_file = f"{self.parent_folder}/config.json"
        self.video_path = f"{self.parent_folder}/output.mp4"
        os.makedirs(self.images_path if self.keep_frames else self.parent_folder)

        # Lists to preserve information in
        self.info_per_frame: List[GroundTruthFrameInfo] = []
        self.frames: List[torch.Tensor] = []
        self.updated_info_per_frame: List[FrameInfo] = []

        # Logger Utility Counters
        self.frame_cnt: int = 0
        self.next_frame_idx_to_save: int = 0
        self.last_frames_with_bbox_idx: List[int] = []

        # Terminal Plots
        # Counter that will provide us with info on how many frames the LeadingUAV
        # has been lost.
        self.lost_for_frames = 0
        self.tprog = None
        if display_terminal_progress:
            self.tprog = TerminalProgress(names=["Progress", "Distance_x", "Distance_y", "Distance_z", "LeadingUAV Lost For"],
                                         limits=[config.simulation_time_s*config.camera_fps,
                                                 config.max_allowed_uav_distance_m,
                                                 config.max_allowed_uav_distance_m,
                                                 config.max_allowed_uav_distance_m,
                                                 config.max_time_lead_is_lost_s*config.camera_fps],
                                         green_area_func=[lambda p: p > config.simulation_time_s*config.camera_fps*0.9, 
                                                          lambda d: d < 3.5,
                                                          lambda d: d < 3.5,
                                                          lambda d: d < 3.5,
                                                          lambda l: l < config.max_time_lead_is_lost_s*config.camera_fps*0.5])

    def create_frame(self, frame: torch.Tensor, is_bbox_frame: bool):
        """
        Creates a frame and append's it to loggers list of frames for future use.

        It also creates a GroundTruthFrameInfo object that also saves in a list. This
        object contains all ground truth information provided by the simulation at the
        time this function is called. Thus you need to be careful and invoce it right
        after you have captured the frame.

        Args:
        - frame: The torch tensor with the expected dimensions defined in GlobalConfig.
        - is_bbox_frame: A boolean that when set to true signals the fact that in the
        future a bbox will be provided for this frame. Else it is just a "filler" frame
        that is between bbox frames in order to achieve camera_fps (which is greater than
        inference_freq)
        """
        # Collect data from simulation to variables
        ego_pos = self.egoUAV.simGetObjectPose().position
        ego_vel = self.egoUAV.simGetGroundTruthKinematics().linear_velocity
        ego_quart = self.egoUAV.simGetObjectPose().orientation
        lead_pos = self.leadingUAV.simGetObjectPose().position
        lead_vel = self.leadingUAV.simGetGroundTruthKinematics().linear_velocity
        lead_quart = self.leadingUAV.simGetObjectPose().orientation

        # Organize the variables above into a GroundTruthFrameInfo
        frame_info: GroundTruthFrameInfo = {
            "frame_idx": self.frame_cnt,
            "timestamp": time.time_ns(),
            "egoUAV_position": (ego_pos.x_val, ego_pos.y_val, ego_pos.z_val),
            "egoUAV_orientation_quartanion": (ego_quart.x_val, ego_quart.y_val, ego_quart.z_val, ego_quart.w_val),
            "egoUAV_velocity": (ego_vel.x_val, ego_vel.y_val, ego_vel.z_val),
            "leadingUAV_position": (lead_pos.x_val, lead_pos.y_val, lead_pos.z_val),
            "leadingUAV_orientation_quartanion": (lead_quart.x_val, lead_quart.y_val, lead_quart.z_val, lead_quart.w_val),
            "leadingUAV_velocity": (lead_vel.x_val, lead_vel.y_val, lead_vel.z_val),
            "angle_deg": sim_calculate_angle(self.egoUAV, self.leadingUAV)
        }

        # Update the Logger state
        if is_bbox_frame:
            self.last_frames_with_bbox_idx.append(self.frame_cnt)
        self.info_per_frame.append(frame_info)
        self.frames.append(frame)
        self.frame_cnt += 1

    def update_frame(self,
                     bbox: Optional[BoundingBox],
                     est_frame_info: EstimatedFrameInfo
                ):
        """
        This method should be called when a bbox is available. This bbox is found\
        on the frame with idx camera_frame_idx in the list with the frames preserved.\

        Due to the way this logger has been implemented, frames can be saved before the\
        simulation is complete, so we may not run out of memory, if the simulation is run for too long.\
        This creates an indexing chaos, since we also need to be able to track up to which frame we have\
        already saved to the disk and thus remove from the list of frames.\
        Thus even though the camera_frame_idx is not required, it is there to help us assert some conditions,\
        so we make sure everything works as expected. Can be removed in the future.\

        This function mostly helps with the indexing and invokes draw_frame to add the information to the frame.\
        
        Args:
        - bbox: The bbox we have found.\
        - est_frame_info: It is an object returned from EgoUAV. It is all the estimations it \
        has made, based on the offset extracted by the bbox. We also log EgoUAV's point of view \
        in order to be able to compare it with the ground truth that has been preserved upon \
        frame's creation (create_frame method).
        """
        # Figure out the global index
        if len(self.last_frames_with_bbox_idx) == 0:
            return
        # Calculate the global index of the frame, used to index
        # the info lists. Global meaning the actual index of the frame captured
        # by the camera, considering the first frame captured be 0.
        g_frame_idx = self.last_frames_with_bbox_idx.pop(0)
        
        # The index of the frame on the list (local index, local to the list)
        frame_list_idx = g_frame_idx - self.next_frame_idx_to_save
        # Get the frame using the local (to the frame list) index
        frame = self.frames[frame_list_idx]
            

        # Merge all info into a FrameInfo object
        frame_info = self.merge_to_FrameInfo(bbox=bbox,
                                             sim_info=self.info_per_frame[g_frame_idx],
                                             est_info=est_frame_info)

        # Add the updated information for the frame to a list
        self.updated_info_per_frame.append(frame_info)

        # Add the information on the frame
        self.frames[frame_list_idx] = self.draw_frame(frame, bbox, frame_info)
        
        # Update the counter that shows us for how many frames the LeadingUAV
        # was not found inside our FOV, so you may add it to the progress bar
        # displayed on the terminal, as the simulation runs.
        self.lost_for_frames = 0 if frame_info["extra_still_tracking"] == True else self.lost_for_frames+1
        
        # Update the progress bars displayed on the terminal
        ego_pos = np.array(frame_info["sim_ego_pos"])
        lead_pos = np.array(frame_info["sim_lead_pos"])
        if self.tprog:
            dist = np.abs(ego_pos - lead_pos)
            self.tprog.update([g_frame_idx, dist[0], dist[1], dist[2], self.lost_for_frames])

    def draw_frame(self,
                   frame: torch.Tensor,
                   bbox: Optional[BoundingBox],
                   frame_info: FrameInfo
                ) -> torch.Tensor:
        """
        Merges all information that it is provided and uses utils.image functions
        to append all this information onto a frame.

        Before the information is added the resolution of the frame is increased,
        so the font is more readable and it does not take a lot of space on the image.

        It also preserves the merge FrameInfo object to a new list.
        """
        if bbox:
            frame = add_bbox_to_image(frame, bbox)

        frame = increase_resolution(frame, IMG_RESOLUTION_INCR_FACTOR)
        frame = add_info_to_image(frame, frame_info, self.config.score_threshold)
        return frame

    def fix_frame_size(self, frame: torch.Tensor, sim_info: GroundTruthFrameInfo):
        if frame.size() == torch.Size([3,
                                       IMG_RESOLUTION_INCR_FACTOR*IMG_HEIGHT,
                                       IMG_RESOLUTION_INCR_FACTOR*IMG_WIDTH
                                      ]
            ):
            return frame
        if frame.size() == torch.Size([3, IMG_HEIGHT, IMG_WIDTH]):
            frame_info = self.merge_to_FrameInfo(bbox=None, sim_info=sim_info)
            frame = self.draw_frame(frame=frame, bbox=None, frame_info=frame_info)
            return frame
        raise Exception("Unexpected frame size!")

    def save_frames(self, finalize: bool = False):
        """
        Saves all the frames that have been drawn so far.
        If finalize is True, it will draw any leftover frames,
        with the ground truth information only and save them aswell.
        """
        if not self.keep_frames:
            return
        idx = 0
        for idx, frame in enumerate(self.frames):
            # Calculate the global frame index
            info_idx = self.next_frame_idx_to_save + idx
            # Do not save frames that will be updated in the future
            # these are the one that will include the next bbox and the frames
            # after that.
            if not finalize \
                and self.last_frames_with_bbox_idx \
                and info_idx >= self.last_frames_with_bbox_idx[0]:
                break
            # Make sure that all frames you save have the desired resolution
            self.fix_frame_size(frame, self.info_per_frame[info_idx])
            save_image(frame, f"{self.images_path}/img_EgoUAV_{self.info_per_frame[info_idx]['timestamp']}.png")
        
        # Delete the frames you saved
        del self.frames[0:idx]
        # Update the index of the next frame you want to save
        self.next_frame_idx_to_save += idx

    def write_setup(self, status: Status_t, statistics: Statistics_t):
        """
        Writes a txt file that contains useful information for the simulation run.
        It is mostly about GlobalConfig variables.
        """
        conf_stat_dict: ExtendedCoSimulatorConfig_t = {
            "uav_velocity": self.config.uav_velocity,
            "score_threshold": self.config.score_threshold,
            "max_vel": self.config.max_vel,
            "min_vel": self.config.min_vel,
            "weight_vel": self.config.weight_vel,
            "sim_fps": self.config.sim_fps,
            "simulation_time_s": self.config.simulation_time_s,
            "camera_fps": self.config.camera_fps,
            "model_id": self.config.model_id,
            "model_path": self.config.model_path,
            "checkpoint_path": self.config.checkpoint_path,
            "infer_freq_Hz": self.config.infer_freq_Hz,
            "filter_freq_Hz": self.config.filter_freq_Hz,
            "filter_type": self.config.filter_type,
            "motion_model": self.config.motion_model,
            "use_pepper_filter": self.config.use_pepper_filter,
            "leadingUAV_update_vel_interval_s": self.config.leadingUAV_update_vel_interval_s,
            "max_time_lead_is_lost_s": self.config.max_time_lead_is_lost_s,
            "status": status,
            "frame_count": self.frame_cnt,
            "dist_mse": statistics["dist_mse"],
            "lead_vel_mse": statistics["lead_vel_mse"],
            "avg_true_dist": statistics["avg_true_dist"],
            "use_gimbal": self.config.use_gimbal
        }
        with open(self.config_file, 'w') as f:
            f.write(json.dumps(conf_stat_dict))

    def dump_logs(self):
        """
        Sorts all the FrameIngo objects, just be absolutely sure they are on the right order
        (it may not be required but it is safer to do so)

        Pickles the list and saves it to a file.
        """
        # source: https://stackoverflow.com/questions/72899/how-to-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary-in-python
        self.updated_info_per_frame = sorted(self.updated_info_per_frame,
                                             key=itemgetter("extra_frame_idx"),
                                             reverse=False
                                        )
        with open(self.logfile, 'wb') as f:
            pickle.dump(self.updated_info_per_frame, f)

    def write_video(self):
        """
        Using all frames created and drawn by the logger, use OpenCV's
        VideoWriter to write the output video of this run.
        """
        # Use cv2's video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.video_path,
                                fourcc,
                                self.config.camera_fps,
                                (IMG_WIDTH*IMG_RESOLUTION_INCR_FACTOR,
                                 IMG_HEIGHT*IMG_RESOLUTION_INCR_FACTOR)
                )
        if self.keep_frames:
            # Load images into a list
            files = os.listdir(self.images_path)
            files.sort()
            # Add the images to the video one by one
            for imgf in files: 
                video.write(cv2.imread(os.path.join(self.images_path, imgf))) 
        else:
            for i, frame in enumerate(self.frames):
                frame = self.fix_frame_size(frame, self.info_per_frame[i])
                frame_for_cv2 = (frame*255).to(torch.uint8).permute(1, 2, 0).numpy()
                frame_for_cv2 = cv2.cvtColor(frame_for_cv2, cv2.COLOR_RGB2BGR)
                video.write(frame_for_cv2)

        # Save the video to the video path
        video.release()
        cv2.destroyAllWindows()

    def merge_to_FrameInfo(self,
                           bbox: Optional[BoundingBox],
                           sim_info: GroundTruthFrameInfo,
                           est_info: EstimatedFrameInfo = {
                               "egoUAV_target_velocity": None,
                               "angle_deg": None,
                               "leadingUAV_position": None,
                               "leadingUAV_velocity": None,
                               "egoUAV_position": None,
                               "still_tracking": None
                            }
        ) -> FrameInfo:
        """
        Merges all provided arguments to a single FrameInfo object,
        that can be later be saved (dump_logs) and/or used to add information
        on the frame, this information refers to.
        """
        err_lead_pos = None if est_info["leadingUAV_position"] is None \
                            else tuple(np.array(sim_info["leadingUAV_position"]) -
                                       np.array(est_info["leadingUAV_position"])
                                    )
        err_lead_vel = None if est_info["leadingUAV_velocity"] is None \
                            else tuple(np.array(sim_info["leadingUAV_velocity"]) -
                                       np.array(est_info["leadingUAV_velocity"])
                                    )
        err_ego_pos = None if est_info["egoUAV_position"] is None \
                           else tuple(np.array(sim_info["egoUAV_position"]) - 
                                      np.array(est_info["egoUAV_position"])
                                    )
        err_ego_vel = None if est_info["egoUAV_target_velocity"] is None \
                           else tuple(np.array(sim_info["egoUAV_velocity"]) - 
                                      np.array(est_info["egoUAV_target_velocity"]))
        err_angle = None if est_info["angle_deg"] is None \
                         else sim_info["angle_deg"] - est_info["angle_deg"]
        
        frameInfo: FrameInfo = {
            "bbox_score": None if not bbox else bbox.score,
            "sim_lead_pos": sim_info["leadingUAV_position"],
            "est_lead_pos": est_info["leadingUAV_position"],
            "err_lead_pos": err_lead_pos,

            "sim_ego_pos": sim_info["egoUAV_position"],
            "est_ego_pos": est_info["egoUAV_position"],
            "err_ego_pos": err_ego_pos,

            "sim_lead_vel": sim_info["leadingUAV_velocity"],
            "est_lead_vel": est_info["leadingUAV_velocity"],
            "err_lead_vel": err_lead_vel,

            "sim_ego_vel": sim_info["egoUAV_velocity"],
            "target_ego_vel": est_info["egoUAV_target_velocity"],
            "err_ego_vel": err_ego_vel,

            "sim_angle_deg": sim_info["angle_deg"],
            "est_angle_deg": est_info["angle_deg"],
            "err_angle": err_angle,

            "extra_frame_idx": sim_info["frame_idx"],
            "extra_timestamp": sim_info["timestamp"],
            "extra_leading_orientation_quartanion": sim_info["leadingUAV_orientation_quartanion"],
            "extra_ego_orientation_quartanion": sim_info["egoUAV_orientation_quartanion"],
            "extra_still_tracking": est_info["still_tracking"]
        }
        return frameInfo

    def get_statistics(self) -> Statistics_t:
        n = len(self.updated_info_per_frame)
        avg_true_dist: float = 0.
        # Some missing values may occure due to frames not
        # being fed to our Controller and thus merging a GroundTruthFrameInfo
        # with an empty EstimatedFrameInfo object (empty -> all dict value none).
        dist_mse: float = 0.
        dist_meas_cnt = 0
        lead_vel_mse: Optional[float] = 0.
        lead_vel_meas_cnt = 0
        for info in self.updated_info_per_frame:
            sim_dist = np.linalg.norm(np.array(info["sim_ego_pos"]) - np.array(info["sim_lead_pos"]))
            avg_true_dist += sim_dist.item()
            
            if info["est_lead_pos"] and info["est_ego_pos"]:
                est_dist = np.linalg.norm(np.array(info["est_ego_pos"]) - np.array(info["est_lead_pos"]))
                dist_mse += (sim_dist - est_dist).item()**2
                dist_meas_cnt += 1

            if self.config.filter_type == "None":
                lead_vel_mse = None
            elif info["err_lead_vel"] and lead_vel_mse is not None:
                lead_vel_mse += np.linalg.norm(np.array(info["err_lead_vel"])).item()**2
                lead_vel_meas_cnt += 1

        return {
                "dist_mse": dist_mse/dist_meas_cnt,
                "lead_vel_mse": lead_vel_mse/lead_vel_meas_cnt if lead_vel_mse else None,
                "avg_true_dist": avg_true_dist/n
               }

    def exit(self, status: Status_t):
        # Write the mp4 file
        if self.get_video:
            print("\nWriting the video...")
            self.save_frames(finalize=True)
            self.write_video()

        # Write a setup.txt file containing all the important configuration options used for
        # this run
        print("\nWriting the setup file and dumping the logs...")
        stats = self.get_statistics()
        self.write_setup(status=status,
                         statistics=stats)
        self.dump_logs()

        print("\nStatistics...")
        for key in stats:
            print(f"{key}: {stats[key]}")

        print(f"Camera frames: {self.frame_cnt}")
        print(f"Evaluated frames: {len(self.updated_info_per_frame)}")


class GraphLogs:
    """
    A class that utilizes, the information logged and creates useful Graphs,
    for the state of the simulation at each timestep (at which the logger recorded info).
    """
    def __init__(self, frame_info: List[FrameInfo]) -> None:
        self.frame_info = frame_info

    def _map_axis_to_idx(self,
                         axis: Literal["x", "y", "z", "all"]
    ) -> int:
        return 0 if axis == "x" else 1 if axis == "y" else 2

    def _map_gtype_vname_to_keys(self,
                                graph_type: Literal["position", "velocity", "distance"],
                                vname: Literal["EgoUAV", "LeadingUAV"]
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        if vname == "EgoUAV":
            if graph_type == "position":
                sim_key, est_key, sim_key2, est_key2 = "sim_ego_pos", "est_ego_pos", None, None
            elif graph_type == "velocity":
                sim_key, est_key, sim_key2, est_key2 = "sim_ego_vel", "target_ego_vel", None, None
            else:
                sim_key, est_key, sim_key2, est_key2 = "sim_ego_pos", "est_ego_pos", "sim_lead_pos", "est_lead_pos"
        else:
            if graph_type == "position":
                sim_key, est_key, sim_key2, est_key2 = "sim_lead_pos", "est_lead_pos", None, None
            elif graph_type == "velocity":
                sim_key, est_key, sim_key2, est_key2 = "sim_lead_vel", "est_lead_vel", None, None
            else:
                sim_key, est_key, sim_key2, est_key2 = "sim_lead_pos", "est_lead_pos", "sim_ego_pos", "est_ego_pos"
        return sim_key, est_key, sim_key2, est_key2
                
    def _graph(self,
              ax: plt.Axes,
              fps: int,
              graph_type: Literal["position", "velocity", "distance"],
              axis: Literal["x", "y", "z", "all"],
              vehicle_name: Literal["EgoUAV", "LeadingUAV"],
              sim_color: str,
              est_color: str
    ) -> None:
        def index_info(info, key, axis):
            reg = info[key]
            if reg is None:
                return np.nan
            if isinstance(reg, Tuple) and axis != "all":
                return info[key][self._map_axis_to_idx(axis)]
            return reg
        
        def extract_values(info, sim_key, est_key, sim_key2, est_key2, axis):
            simi = np.array(index_info(info, sim_key, axis))
            esti = np.array(index_info(info, est_key, axis))
            if sim_key2 is None or est_key2 is None:
                if axis == "all":
                    return np.linalg.norm(simi), np.linalg.norm(esti)
                else:
                    return simi, esti
            
            simi2 = np.array(index_info(info, sim_key2, axis))
            esti2 = np.array(index_info(info, est_key2, axis))
            return np.linalg.norm(simi - simi2), np.linalg.norm(esti - esti2)

        sim_key, est_key, sim_key2, est_key2 = self._map_gtype_vname_to_keys(graph_type, vehicle_name)
        sim_info = []
        est_info = []
        for info in self.frame_info:
            simi, esti = extract_values(info, sim_key, est_key, sim_key2, est_key2, axis)
            sim_info.append(simi)
            est_info.append(esti)
        
        mult_factor = 1/fps
        x_axis = [x*mult_factor for x in range(0, len(self.frame_info))]
        ax.plot(x_axis, sim_info, color=sim_color, label="ground truth")
        ax.plot(x_axis, est_info, color=est_color, label="estimation")
        ax.legend()

    def graph(self,
              fps: int,
              graph_type: Literal["position", "velocity", "distance"],
              axis: Literal["x", "y", "z", "all"],
              vehicle_name: Literal["EgoUAV", "LeadingUAV"],
              filename: str,
              sim_color: str = "blue",
              est_color: str = "red"
    ) -> None:
        fig, ax = plt.subplots()
        self._graph(ax, fps, graph_type, axis, vehicle_name, sim_color, est_color)
        fig.savefig(filename)
        plt.close(fig)

    def plot_movement_3d(self, filename: str, path: Optional[List[airsim.Vector3r]] = None):
        """
        Plots the positions of the EgoUAV and the LeadingUAV on a 3D plot.
        If the path argument is provided, it also plots the expected path the
        LeadingUAV moves on.

        Args:
        path: The List of points in 3D space the LeadingUAV will be provided with
        in order to visit. If None this line will not be plotted.
        """
        # Extract the positions at which the EgoUAV registers
        lead_pos = np.array([[], [], []])
        ego_pos = np.array([[], [], []])
        for info in self.frame_info:
            lead_pos = np.hstack([lead_pos, np.array(info["sim_lead_pos"]).reshape([3,1])])
            ego_pos = np.hstack([ego_pos, np.array(info["sim_ego_pos"]).reshape([3,1])])
        
        # Convert the path to the correct format
        path_array = None
        if path:
            path_array = np.array([[], [], []])
            for point in path:
                path_array = np.hstack([path_array, point.to_numpy_array().reshape([3, 1])])

        # Create the 3D plot using matplotlib
        fig = plt.figure()
        ax: plt.Axes = fig.add_subplot(projection='3d')
        ax.plot(xs=lead_pos[0, :], ys=lead_pos[1, :], zs=lead_pos[2, :], label="LeadingUAV position")
        ax.plot(xs=ego_pos[0, :], ys=ego_pos[1, :], zs=ego_pos[2, :], label="EgoUAV position")
        if path_array is not None:
            ax.plot(xs=path_array[0, :], ys=path_array[1, :], zs=path_array[2, :], label="Path")

        # If the range of values in one direction is really small the graph can be misleading.
        # So we fix that by setting the minimum range for the ticks to 5m
        for i, ticks_func in enumerate(["set_xlim3d", "set_ylim3d", "set_zlim3d"]):
            vals = np.hstack([ego_pos[i, :], lead_pos[i, :]])
            max, min = vals.max(), vals.min()
            if max - min < 5:
                avg = (max + min)/2
                half_range = MOVEMENT_PLOT_MIN_RANGE / 2
                getattr(ax, ticks_func)(avg-half_range, avg+half_range)

        ax.legend()
        ax.view_init(elev=20., azim=-120, roll=0) # type: ignore
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore
        ax.invert_yaxis()
        ax.invert_zaxis() # type: ignore
        fig.savefig(filename)
        plt.close(fig)

class TerminalProgress():
    def __init__(self,
                 names: List[str],
                 limits: List[int],
                 green_area_func: List[Optional[Callable[[float], bool]]]
        ):
        self.limits = limits
        self.green_area_func = green_area_func
        self.names = names
        self.num_lines_to_clear = 3*len(names)
        self.init_plot = True
        tplt.interactive(True)
        self.update([0 for _ in names])

    def _sel_color(self, value: float, idx: int) -> str:
        if not self.green_area_func[idx]:
            return "blue"
        return "green" if self.green_area_func[idx](value) else "red" # type: ignore
    
    def update(self, values: List[float]) -> Optional[bool]:
        assert(len(values) == len(self.names))
        if not self.init_plot:
            tplt.clt(lines=self.num_lines_to_clear)
        self.init_plot = False

        for idx, value in enumerate(values):
            tplt.simple_bar([self.names[idx], "Limit"],
                            [value, self.limits[idx]],
                            color=[self._sel_color(value, idx), "blue"],
                            width=50)
            print("")
