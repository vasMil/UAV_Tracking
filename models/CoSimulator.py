from typing import Literal
import traceback
import os
import time

import airsim
import torch

from project_types import Status_t, _map_to_status_code
from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from models.logger import Logger, GraphLogs
from utils.simulation import getTestPath

class CoSimulator():
    def __init__(self,
                 simulation_time_s: int = config.simulation_time_s,
                 leadingUAV_update_vel_interval_s: int = config.leadingUAV_update_vel_interval_s,
                 sim_fps: int = config.sim_fps,
                 camera_fps: int = config.camera_fps,
                 infer_freq_Hz: int = config.infer_freq_Hz,
                 filter_type: Literal["None", "KF"] = config.filter_type,
                 filter_freq_Hz: int = config.filter_freq_Hz,
                 max_time_lead_is_lost_s: int = config.max_time_lead_is_lost_s
        ):
            if sim_fps < camera_fps:
                raise Exception("sim_fps cannot be less than camera_fps")
            if camera_fps < infer_freq_Hz:
                raise Exception("camera_fps cannot be less that infer_freq_Hz")
            if filter_type == "KF" and camera_fps < infer_freq_Hz:
                Warning("Since you allow the UAV to move (change direction) faster than the camera may record, you may experience some flickering")

            self.simulation_time_s = simulation_time_s
            self.leadingUAV_update_vel_interval_s = leadingUAV_update_vel_interval_s
            self.sim_fps = sim_fps
            self.camera_fps = camera_fps
            self.infer_freq_Hz = infer_freq_Hz
            self.filter_type = filter_type
            self.max_time_lead_is_lost_s = max_time_lead_is_lost_s
            if config.filter_type == "KF":
                self.filter_freq_Hz = filter_freq_Hz
            else:
                self.filter_freq_Hz = infer_freq_Hz
            self.done: bool = False
            self.status: int = _map_to_status_code("Running")
            self.lost_lead_infer_frame_cnt = 0

            # Create a client to communicate with the UE
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print(f"Vehicle List: {self.client.listVehicles()}\n")
            # Reset the position of the UAVs (just to make sure)
            self.client.reset()
            # Wait for the takeoff to complete
            self.leadingUAV = LeadingUAV("LeadingUAV")
            self.egoUAV = EgoUAV("EgoUAV", filter_type=filter_type)
            self.egoUAV.lastAction.join()
            self.leadingUAV.lastAction.join()

            # Create a Logger
            self.logger = Logger(egoUAV=self.egoUAV,
                                 leadingUAV=self.leadingUAV,
                                 sim_fps=sim_fps,
                                 simulation_time_s=simulation_time_s,
                                 camera_fps=camera_fps,
                                 infer_freq_Hz=infer_freq_Hz,
                                 leadingUAV_update_vel_interval_s=leadingUAV_update_vel_interval_s,
                                 filter_type = filter_type,
                                 filter_freq_Hz = filter_freq_Hz,
                                 max_time_lead_is_lost_s = max_time_lead_is_lost_s
                            )
  
    def start(self):
        # Move up so you minimize shadows
        self.leadingUAV.moveByVelocityAsync(0, 0, -5, 10)
        self.egoUAV.moveByVelocityAsync(0, 0, -5, 10)
        self.egoUAV.lastAction.join()
        self.leadingUAV.lastAction.join()
        # Wait for the vehicles to stabilize
        time.sleep(10)

        # Pause the simulation
        self.client.simPause(True)
        
        # Initialize the control variables
        self.frame_idx = 0
        self.camera_frame = self.egoUAV._getImage()
        self.bbox, self.prev_bbox = None, None
        self.score, self.prev_score = None, None
        self.orient, self.prev_orient = self.egoUAV.getPitchRollYaw(), self.egoUAV.getPitchRollYaw()
        self.camera_frame_idx, self.prev_camera_frame_idx = -1, -1

    def _advance(self):
        # print(f"\nFRAME: {self.frame_idx}")

        if self.frame_idx % round(self.sim_fps/self.camera_fps) == 0:
            self.hook_camera_frame_capture()

        # Update the leadingUAV velocity every update_vel_s*sim_fps frames
        if self.frame_idx % (self.leadingUAV_update_vel_interval_s*self.sim_fps) == 0:
            self.hook_leadingUAV_move()

        # Get a bounding box and move towards the previous detection
        # this way we also simulate the delay between the capture of the frame
        # and the output of the NN for this frame.
        if self.frame_idx % round(self.sim_fps/self.infer_freq_Hz) == 0:
            still_tracking = self.hook_net_inference()
            self.lost_lead_infer_frame_cnt = 0 if still_tracking else self.lost_lead_infer_frame_cnt+1


        # If we haven't yet decided on the movement, due to the inference frequency limitation
        # and we use a Kalman Filter, check if it is time to advance the KF.
        elif self.filter_type == "KF" and self.frame_idx % round(self.sim_fps/self.filter_freq_Hz):
            self.hook_filter_advance_only()

        # Continue the simulation for a few seconds to match
        # the desired sim_fps
        self.frame_idx += 1
        self.client.simContinueForTime(1/self.sim_fps)
        
        # Check if simulation time limit has been reached.
        if self.frame_idx == self.simulation_time_s*self.sim_fps:
            self.finalize("Time's up")
        # Check if we lost the LeadingUAV
        elif self.lost_lead_infer_frame_cnt == (self.infer_freq_Hz*self.max_time_lead_is_lost_s):
            print(f"The LeadingUAV was lost for {self.max_time_lead_is_lost_s} seconds!")
            self.finalize("LeadingUAV lost")
        # Check for collisions
        elif self.leadingUAV.simGetCollisionInfo().object_name == config.egoUAV_name:
            print(f"Collision between the two UAVs detected!")
            self.finalize("EgoUAV and LeadingUAV collision")
        elif self.leadingUAV.simGetCollisionInfo().has_collided or self.egoUAV.simGetCollisionInfo().has_collided:
            uav_collided_name = self.leadingUAV.name if self.leadingUAV.simGetCollisionInfo().has_collided else self.egoUAV.name
            status: Status_t = "LeadingUAV collision" if self.leadingUAV.simGetCollisionInfo().has_collided else "EgoUAV collision"
            print(f"The {uav_collided_name} crashed!")
            self.finalize(status)

    def advance(self):
        try:
            self._advance()
        except Exception:
            print("There was an error, writing setup file and releasing AirSim...")
            print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
            traceback.print_exc()
            self.finalize("Error")

    def hook_camera_frame_capture(self):
        fail_cnt = 0
        self.camera_frame = self.egoUAV._getImage()
        while self.camera_frame.size() != torch.Size([3,  config.img_height, config.img_width]):
            fail_cnt +=1
            self.camera_frame = self.egoUAV._getImage()
            if fail_cnt > 4:
                raise Exception("Last captured frame has unexpected size!")

        self.logger.create_frame(self.camera_frame,
                                 is_bbox_frame=(self.frame_idx % round(self.sim_fps/self.infer_freq_Hz) == 0)
        )
        self.camera_frame_idx += 1

    def hook_leadingUAV_move(self):
        self.leadingUAV.random_move(self.leadingUAV_update_vel_interval_s)
        # if self.frame_idx == 0:
            # self.leadingUAV.moveOnPathAsync(getTestPath(self.leadingUAV.simGetGroundTruthKinematics().position))

    def hook_net_inference(self) -> bool:
        """
        This function is called each time we want to evaluate a frame, in order
        to detect the LeadingUAV.

        Returns:
        A boolean that is True if the LeadingUAV was detected inside the frame.
        Else it returns False.
        """
        # Run egoUAV's detection net, save the frame with all
        # required information. Hold on to the bbox, to move towards it when the
        # next frame for evaluation is captured.
        bbox, score = self.egoUAV.net.eval(self.camera_frame, 0)
        orient = self.egoUAV.getPitchRollYaw()         

        # There is no way we have a bbox when just inserting the first frame to the logger
        if self.frame_idx != 0:
            # Perform the movement for the previous detection
            _, est_frame_info = self.egoUAV.moveToBoundingBoxAsync(self.prev_bbox, self.prev_orient, dt=(1/self.filter_freq_Hz))
        
            # Update the frame in the logger
            self.logger.update_frame(bbox=self.prev_bbox, est_frame_info=est_frame_info, camera_frame_idx=self.prev_camera_frame_idx)
            self.logger.save_frames()

        # Update
        self.prev_bbox = bbox
        self.prev_score = score
        self.prev_orient = orient
        self.prev_camera_frame_idx = self.camera_frame_idx
        return True if bbox else False

    def hook_filter_advance_only(self):
        self.egoUAV.advanceUsingFilter(dt=(1/self.filter_freq_Hz))

    def finalize(self, status: Status_t):
        if self.done == True:
            return
        self.done = True
        self.status = _map_to_status_code(status)
        # Save the last evaluated bbox
        _, est_frame_info = self.egoUAV.moveToBoundingBoxAsync(self.bbox, self.orient, dt=(1/self.filter_freq_Hz))
        self.logger.update_frame(self.bbox, est_frame_info, self.camera_frame_idx)
        # Save any leftorver frames
        self.logger.save_frames(finalize=True)
        
        # Output some statistics
        print(f"Simulation run for {self.frame_idx/self.sim_fps} seconds")
        self.exit(status)

    def exit(self, status: Status_t):
        # Free the simulation and reset the vehicles
        self.client.simPause(False)
        self.egoUAV.disable()
        self.leadingUAV.disable()
        self.client.reset()

        # Write a setup.txt file containing all the important configuration options used for
        # this run
        self.logger.write_setup(status)
        self.logger.dump_logs()
        self.logger.print_statistics()

        # Write the mp4 file
        self.logger.save_frames(finalize=True)
        self.logger.write_video()

    def export_graphs(self):
        gl = GraphLogs(frame_info=self.logger.updated_info_per_frame)
        gl.graph(self.camera_fps, graph_type="position", axis="all", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_pos.png"))
        gl.graph(self.camera_fps, graph_type="position", axis="x", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_posx.png"))
        gl.graph(self.camera_fps, graph_type="position", axis="y", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_posy.png"))
        gl.graph(self.camera_fps, graph_type="position", axis="z", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_posz.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="all", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_vel.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="x", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_velx.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="y", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_vely.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="z", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_velz.png"))

        gl.graph(self.camera_fps, graph_type="position", axis="all", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_pos.png"))
        gl.graph(self.camera_fps, graph_type="position", axis="x", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_posx.png"))
        gl.graph(self.camera_fps, graph_type="position", axis="y", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_posy.png"))
        gl.graph(self.camera_fps, graph_type="position", axis="z", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_posz.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="all", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_vel.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="x", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_velx.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="y", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_vely.png"))
        gl.graph(self.camera_fps, graph_type="velocity", axis="z", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_velz.png"))

        gl.graph(self.camera_fps, graph_type="distance", axis="all", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "dist.png"))
        gl.graph(self.camera_fps, graph_type="distance", axis="x", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "distx.png"))
        gl.graph(self.camera_fps, graph_type="distance", axis="y", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "disty.png"))
        gl.graph(self.camera_fps, graph_type="distance", axis="z", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "distz.png"))
