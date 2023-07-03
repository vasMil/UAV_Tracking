from typing import List, Optional, Literal
from operator import itemgetter
import datetime, time
import os

import torch
from torchvision.utils import save_image
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from models.BoundingBox import BoundingBox
from GlobalConfig import GlobalConfig as config
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
                 sim_fps: int = config.sim_fps,
                 simulation_time_s: int = config.simulation_time_s,
                 camera_fps: int = config.camera_fps,
                 infer_freq_Hz: int = config.infer_freq_Hz,
                 leadingUAV_update_vel_interval_s: int = config.leadingUAV_update_vel_interval_s
            ) -> None:
        # Settings
        self.egoUAV = egoUAV
        self.leadingUAV = leadingUAV
        self.sim_fps = sim_fps
        self.simulation_time_s = simulation_time_s
        self.camera_fps = camera_fps
        self.infer_freq_Hz = infer_freq_Hz
        self.leadingUAV_update_vel_interval_s = leadingUAV_update_vel_interval_s
        self.image_res_incr_factor = 3

        # Folder and File namse
        dt = datetime.datetime.now()
        self.parent_folder = f"recordings/{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
        self.images_path = f"{self.parent_folder}/images"
        self.logfile = f"{self.parent_folder}/log.pkl"
        self.setup_file = f"{self.parent_folder}/setup.txt"
        self.video_path = f"{self.parent_folder}/output.mp4"
        os.makedirs(self.images_path)

        # Lists to preserve information in
        self.info_per_frame: List[GroundTruthFrameInfo] = []
        self.frames: List[torch.Tensor] = []
        self.updated_info_per_frame: List[FrameInfo] = []

        # Counters
        self.frame_cnt: int = 0
        self.next_frame_idx_to_save: int = 0
        self.last_frames_with_bbox_idx: List[int] = []

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
                     est_frame_info: EstimatedFrameInfo,
                     camera_frame_idx: int
                ):
        """
        This method should be called when a bbox is available. This bbox is found
        on the frame with idx camera_frame_idx in the list with the frames preserved.

        Due to the way this logger has been implemented, frames can be saved before the
        simulation is complete, so we may not run out of memory, if the simulation is run for too long.
        This creates an indexing chaos, since we also need to be able to track up to which frame we have
        already saved to the disk and thus remove from the list of frames. Thus even though the camera_frame_idx
        is not required, it is there to help us assert some conditions, so we are sure everything works as expected.
        It may be removed in the future.

        This function mostly helps with the indexing and invokes draw_frame to add the information to the frame.
        
        Args:
        - bbox: The bbox we have found.
        - est_frame_info: It is an object returned from EgoUAV. It is all the estimations it \
        has made, based on the offset extracted by the bbox. We also log EgoUAV's point of view \
        in order to be able to compare it with the ground truth that has been preserved upon \
        frame's creation (create_frame method).
        - camera_frame_idx: The index of the camera frame to be updated.
        """
        g_frame_idx = self.last_frames_with_bbox_idx.pop(0)
        frame_list_idx = g_frame_idx - self.next_frame_idx_to_save
        frame = self.frames[frame_list_idx]
        try:
            assert(self.info_per_frame[g_frame_idx]["frame_idx"] == camera_frame_idx)
        except:
            print(f"Saved frame idx: {self.info_per_frame[g_frame_idx]['frame_idx']}")
            print(f"True frame idx: {camera_frame_idx}")
            raise Exception("FRAME INDEXING ERROR!")
        self.frames[frame_list_idx] = self.draw_frame(frame, bbox, self.info_per_frame[g_frame_idx], est_frame_info)

    def draw_frame(self,
                   frame: torch.Tensor,
                   bbox: Optional[BoundingBox],
                   sim_frame_info: GroundTruthFrameInfo,
                   est_frame_info: EstimatedFrameInfo = {
                       "egoUAV_target_velocity": None,
                       "angle_deg": None,
                       "leadingUAV_position": None,
                       "leadingUAV_velocity": None,
                       "egoUAV_position": None,
                       "still_tracking": None
                    }
                ) -> torch.Tensor:
        """
        Merges all information that it is provided and uses utils.image functions
        to append all this information onto a frame.

        Before the information is added the resolution of the frame is increased,
        so the font is more readable and it does not take a lot of space on the image.

        It also preserves the merge FrameInfo object to a new list.
        """
        if bbox:
            # Add info on the camera frame
            frame = add_bbox_to_image(frame, bbox)

        frame = increase_resolution(frame, self.image_res_incr_factor)

        # Merge all info into a FrameInfo object and append the information onto the frame
        frame_info = self.merge_to_FrameInfo(bbox=bbox, sim_info=sim_frame_info, est_info=est_frame_info)
        frame = add_info_to_image(frame, frame_info)

        # Add the updated information for the frame to a list
        # I do not really worry about the order of the frame_info objects,
        # since they preserve their global frame index in the field "extra_frame_idx",
        # we may sort them later
        self.updated_info_per_frame.append(frame_info)
        return frame

    def save_frames(self, finalize: bool = False):
        """
        Saves all the frames that have been drawn so far.
        If finalize is True, it will draw any leftover frames,
        with the ground truth information only and save them aswell.
        """
        idx = 0
        for idx, frame in enumerate(self.frames):
            # Calculate the global frame index
            info_idx = self.next_frame_idx_to_save + idx
            # Do not save frames that will be updated in the future
            # these are the one that will include the next bbox and the frames
            # after that.
            if not finalize and info_idx >= self.last_frames_with_bbox_idx[0]:
                break
            # Make sure that all frames you save have the desired resolution
            if frame.size() == torch.Size([3, config.img_height, config.img_width]):
                frame = self.draw_frame(frame, None, self.info_per_frame[info_idx])
            elif frame.size() != torch.Size([3,
                                             self.image_res_incr_factor*config.img_height,
                                             self.image_res_incr_factor*config.img_width
                                           ]):
                raise Exception("Unexpected frame size!")
            save_image(frame, f"{self.images_path}/img_EgoUAV_{self.info_per_frame[info_idx]['timestamp']}.png")
        
        # Delete the frames you saved
        del self.frames[0:idx]
        # Update the index of the next frame you want to save
        self.next_frame_idx_to_save += idx

    def write_setup(self):
        """
        Writes a txt file that contains useful information for the simulation run.
        It is mostly about GlobalConfig variables.
        """
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
        # Load images into a list
        files = os.listdir(self.images_path)
        files.sort()
        # Use cv2's video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(self.video_path,
                                fourcc,
                                self.camera_fps,
                                (config.img_width*self.image_res_incr_factor,
                                 config.img_height*self.image_res_incr_factor)
                )
    
        # Appending the images to the video one by one
        for imgf in files: 
            video.write(cv2.imread(os.path.join(self.images_path, imgf))) 
        
        # Save the video to the video path
        video.release()
        cv2.destroyAllWindows()

    def merge_to_FrameInfo(self,
                           bbox: Optional[BoundingBox],
                           sim_info: GroundTruthFrameInfo,
                           est_info: EstimatedFrameInfo
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

class GraphLogs:
    """
    A class that utilizes, the information logged and creates useful Graphs,
    for the state of the simulation at each timestep (at which the logger recorded info).
    """
    def __init__(self,
                 pickle_file: Optional[str] = None,
                 frame_info: Optional[List[FrameInfo]] = None
            ) -> None:
        if pickle_file is None and frame_info is None:
            raise Exception("One of two arguments must be not none")
        if pickle_file and frame_info:
            raise Exception("You should only pass one of the two arguments")

        if pickle_file:
            with open(pickle_file, "rb") as f:
                self.frame_info: List[FrameInfo] = pickle.load(f)
        elif frame_info:
            self.frame_info = frame_info

    def _graph(self,
               ax: plt.Axes,
               fps: int,
               sim_key: str,
               est_key: str,
               sim_color: str = "blue",
               est_color: str = "red"
            ):
        # Initialize two lists, one for ground truth info
        # and one for estimated info
        gt_info = []
        est_info = []
        # Populate the ground truth list
        for info in self.frame_info:
            gt_info.append(np.linalg.norm(np.array(info[sim_key])))
        # Populate the estimation list
        for info in self.frame_info:
            if info[est_key] is None:
                est_info.append(None)
            est_info.append(np.linalg.norm(np.array(info[est_key])))
        # Covert the x axis from frame index to time
        mult_factor = 1/fps
        x_axis = [x*mult_factor for x in range(0, len(self.frame_info))]
        ax.plot(x_axis, gt_info, color=sim_color, label=sim_key)
        ax.plot(x_axis, est_info, color=est_color, label=est_key)
        ax.legend()

    def graph_distance(self, fps: int, filename: str):
        # Calculate the distance between the two UAVs for each frame
        sim_dist = []
        est_dist = []
        for info in self.frame_info:
            ego_pos = np.array(info["sim_ego_pos"])
            lead_pos = np.array(info["sim_lead_pos"])
            sim_dist.append(np.linalg.norm(lead_pos - ego_pos))

        for info in self.frame_info:
            if info["est_ego_pos"] is None or info["est_lead_pos"] is None:
                est_dist.append(None)
                break
            ego_pos = np.array(info["est_ego_pos"])
            lead_pos = np.array(info["est_lead_pos"])
            est_dist.append(np.linalg.norm(lead_pos - ego_pos))

        mult_factor = 1/fps
        x_axis = [x*mult_factor for x in range(0, len(self.frame_info))]
        plt.plot(x_axis, sim_dist, color="blue", label="sim_dist")
        plt.plot(x_axis, est_dist, color="red", label="est_dist")
        plt.legend()
        plt.savefig(filename)

    def graph_velocities(self, fps: int, filename: str):
        fig, ax = plt.subplots()
        self._graph(ax, fps, "sim_lead_vel", "est_lead_vel")
        # self._graph(ax, fps, "sim_ego_vel", "target_ego_vel", "purple", "orange")
        fig.savefig(filename)

    def graph_positions(self, fps: int, filename: str):
        fig, ax = plt.subplots()
        self._graph(ax, fps, "sim_lead_pos", "est_lead_pos")
        # self._graph(ax, fps, "sim_ego_pos", "est_ego_pos", "purple", "orange")
        fig.savefig(filename)

    def graph_velocity_on_axis(self, fps: int,
                               filename: str,
                               axis: Literal["x", "y", "z"],
                               vehicle_name: Literal["ego", "lead"]
        ):
        # Setup the figure
        fig, ax = plt.subplots()
        # Decide upon the keys you want to plot
        sim_key, est_key = ("sim_ego_vel", "target_ego_vel") if vehicle_name == "ego" else ("sim_lead_vel", "est_lead_vel")
        # Extract the correct dimension (instructed by the axis argument)
        idx = 0 if axis == "x" else 1 if axis == "y" else 2

        def extract_values(finfo: FrameInfo, sim_key: str, est_key: str, idx: int):
            sim_info = finfo[sim_key][idx]
            est_info = None
            if finfo[est_key] is not None:
                est_info = finfo[est_key][idx]
            return (sim_info, est_info,)

        # Initialize two lists, one for ground truth info
        # and one for estimated info
        gt_info = []
        est_info = []
        for info in self.frame_info:
            simi, esti = extract_values(info, sim_key, est_key, idx)
            gt_info.append(simi)
            est_info.append(esti)

        # Plot the lists
        mult_factor = 1/fps
        x_axis = [x*mult_factor for x in range(0, len(self.frame_info))]
        ax.plot(x_axis, gt_info, color="blue", label=sim_key)
        ax.plot(x_axis, est_info, color="red", label=est_key)
        ax.legend()
        fig.savefig(filename)
