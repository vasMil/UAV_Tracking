from typing import Optional, List
import os
import time

import airsim
import torch
import numpy as np

from constants import EGO_UAV_NAME, IMG_HEIGHT, IMG_WIDTH, CLOCK_SPEED
from project_types import Status_t, _map_to_status_code, Movement_t, Path_version_t
from config import DefaultCoSimulatorConfig
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from models.logger import Logger, GraphLogs
from utils.simulation import getTestPath

class CoSimulator():
    def __init__(self,
                 config: DefaultCoSimulatorConfig = DefaultCoSimulatorConfig(),
                 log_folder: str = "recordings/",
                 movement: Movement_t = "Random",
                 path_version: Optional[Path_version_t] = None,
                 display_terminal_progress: bool = True
        ):
            if config.sim_fps < config.camera_fps:
                raise Exception("sim_fps cannot be less than camera_fps")
            if config.camera_fps < config.infer_freq_Hz:
                raise Exception("camera_fps cannot be less that infer_freq_Hz")
            if config.filter_type == "KF" and config.camera_fps < config.infer_freq_Hz:
                Warning("Since you allow the UAV to move (change direction) faster than the camera may record, you may experience some flickering")

            if config.filter_type == "KF":
                config.filter_freq_Hz = config.filter_freq_Hz
            else:
                config.filter_freq_Hz = config.infer_freq_Hz
            
            self.done: bool = False
            self.status: int = _map_to_status_code("Running")
            self.time_ns = 0
            
            self.movement: Movement_t = movement
            self.path_version: Optional[Path_version_t] = path_version
            if movement == "Path" and path_version is None:
                raise Exception("If movement is Path, path_version cannot be None!")
            
            self.lost_lead_infer_frame_cnt = 0

            # Create a client to communicate with the UE
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print(f"Vehicle List: {self.client.listVehicles()}\n")
            # Reset the position of the UAVs (just to make sure)
            self.client.reset()
            self.client.simPause(False)
            time.sleep(1)
            # Wait for the takeoff to complete
            self.leadingUAV = LeadingUAV(name="LeadingUAV",
                                         vel_magn=config.uav_velocity,
                                         max_vel=config.max_vel,
                                         min_vel=config.min_vel)
            self.egoUAV = EgoUAV(name=EGO_UAV_NAME,
                                 inference_freq_Hz=config.infer_freq_Hz,
                                 vel_magn=config.uav_velocity,
                                 weight_vel=config.weight_vel,
                                 filter_type=config.filter_type)
            self.egoUAV.lastAction.join()
            self.leadingUAV.lastAction.join()
            
            # Check for any early collisions that will cause the simulation to
            # exit immediately after starting stating (falsly) that the EgoUAV
            # has collided
            if self.egoUAV.simGetCollisionInfo().has_collided or\
               self.leadingUAV.simGetCollisionInfo().has_collided:
                raise Exception("Detected Collision before even starting to run the simulation. Probably a reset caused this!")

            # Create a Logger
            self.logger = Logger(egoUAV=self.egoUAV,
                                 leadingUAV=self.leadingUAV,
                                 config=config,
                                 folder=log_folder,
                                 display_terminal_progress=display_terminal_progress)

            # Define a variable that you may update inside hook_leadingUAV_move
            # with the expected path, so you may later add this path to the
            # movement plot
            self.leading_path: Optional[List[airsim.Vector3r]] = None
            self.config = config

    def start(self):
        # Initialize the control variables
        self.frame_idx = 0
        self.camera_frame = self.egoUAV._getImage()
        self.bbox, self.prev_bbox = None, None
        self.score, self.prev_score = None, None
        self.orient, self.prev_orient = self.egoUAV.getPitchRollYaw(), self.egoUAV.getPitchRollYaw()

        # Move up so you minimize shadows
        self.leadingUAV.moveByVelocityAsync(0, 0, -5, 10)
        self.egoUAV.moveByVelocityAsync(0, 0, -5, 10)
        self.egoUAV.lastAction.join()
        self.leadingUAV.lastAction.join()
        # Wait for the vehicles to stabilize
        time.sleep(20/CLOCK_SPEED)

        # Pause the simulation
        self.client.simPause(True)

    def advance(self):
        t0 = time.time_ns()

        if self.frame_idx % (self.config.sim_fps/self.config.camera_fps) == 0:
            self.hook_camera_frame_capture()

        # Update the leadingUAV velocity every update_vel_s*sim_fps frames
        if self.frame_idx % (self.config.leadingUAV_update_vel_interval_s*self.config.sim_fps) == 0:
            self.hook_leadingUAV_move()

        # Get a bounding box and move towards the previous detection
        # this way we also simulate the delay between the capture of the frame
        # and the output of the NN for this frame.
        if self.frame_idx % (self.config.sim_fps/self.config.infer_freq_Hz) == 0:
            still_tracking = self.hook_net_inference()
            self.lost_lead_infer_frame_cnt = 0 if still_tracking else self.lost_lead_infer_frame_cnt+1
        # If we haven't yet decided on the movement, due to the inference frequency limitation
        # and we use a Kalman Filter, check if it is time to advance the KF.
        elif self.config.filter_type == "KF" and self.frame_idx % (self.config.sim_fps/self.config.filter_freq_Hz):
            self.hook_filter_advance_only()

        # Continue the simulation for a few seconds to match
        # the desired sim_fps
        self.frame_idx += 1
        self.client.simContinueForTime((1/self.config.sim_fps)/CLOCK_SPEED)
        
        self.time_ns += time.time_ns() - t0

        # Check if simulation time limit has been reached.
        if self.frame_idx == self.config.simulation_time_s*self.config.sim_fps:
            self.finalize("Time's up")
        # Check if we lost the LeadingUAV
        elif self.lost_lead_infer_frame_cnt == (self.config.infer_freq_Hz*self.config.max_time_lead_is_lost_s) or\
             np.linalg.norm((
                    self.leadingUAV.simGetObjectPose().position - 
                    self.egoUAV.simGetObjectPose().position).to_numpy_array()
                ) >= 20:
            print(f"The LeadingUAV was lost for {self.config.max_time_lead_is_lost_s} seconds!")
            self.finalize("LeadingUAV lost")
        # Check for collisions
        elif self.leadingUAV.simGetCollisionInfo().object_name == EGO_UAV_NAME:
            print(f"Collision between the two UAVs detected!")
            self.finalize("EgoUAV and LeadingUAV collision")
        elif self.leadingUAV.simGetCollisionInfo().has_collided or \
             self.egoUAV.simGetCollisionInfo().has_collided:
            uav_collided_name = self.leadingUAV.name if self.leadingUAV.simGetCollisionInfo().has_collided else self.egoUAV.name
            status: Status_t = "LeadingUAV collision" if self.leadingUAV.simGetCollisionInfo().has_collided else "EgoUAV collision"
            print(f"The {uav_collided_name} crashed! At frame {self.frame_idx}")
            self.finalize(status)

    def hook_camera_frame_capture(self):
        fail_cnt = 0
        self.camera_frame = self.egoUAV._getImage()
        while self.camera_frame.size() != torch.Size([3,  IMG_HEIGHT, IMG_WIDTH]):
            fail_cnt += 1
            self.camera_frame = self.egoUAV._getImage()
            if fail_cnt > 4:
                raise Exception("Last captured frame has unexpected size!")

        self.logger.create_frame(self.camera_frame,
                                 is_bbox_frame=(self.frame_idx % (self.config.sim_fps/self.config.infer_freq_Hz) == 0)
        )

    def hook_leadingUAV_move(self):
        if self.movement == "Random":
            self.leadingUAV.random_move(self.config.leadingUAV_update_vel_interval_s)
        elif self.movement == "Path" and self.path_version is not None:
            if self.frame_idx != 0: return
            self.leading_path = getTestPath(self.leadingUAV.simGetGroundTruthKinematics().position, version=self.path_version)
            self.leadingUAV.moveOnPathAsync(self.leading_path)

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
            _, est_frame_info = self.egoUAV.moveToBoundingBoxAsync(self.prev_bbox, self.prev_orient, dt=(1/self.config.filter_freq_Hz))
            # Update the frame in the logger
            self.logger.update_frame(bbox=self.prev_bbox, est_frame_info=est_frame_info)
            self.logger.save_frames()

        # Update
        self.prev_bbox = bbox
        self.prev_score = score
        self.prev_orient = orient
        return True if bbox else False

    def hook_filter_advance_only(self):
        self.egoUAV.advanceUsingFilter(dt=(1/self.config.filter_freq_Hz))

    def export_graphs(self):
        gl = GraphLogs(frame_info=self.logger.updated_info_per_frame)
        gl.graph(self.config.camera_fps, graph_type="position", axis="all", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_pos.png"))
        gl.graph(self.config.camera_fps, graph_type="position", axis="x", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_posx.png"))
        gl.graph(self.config.camera_fps, graph_type="position", axis="y", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_posy.png"))
        gl.graph(self.config.camera_fps, graph_type="position", axis="z", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_posz.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="all", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_vel.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="x", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_velx.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="y", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_vely.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="z", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "ego_velz.png"))

        gl.graph(self.config.camera_fps, graph_type="position", axis="all", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_pos.png"))
        gl.graph(self.config.camera_fps, graph_type="position", axis="x", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_posx.png"))
        gl.graph(self.config.camera_fps, graph_type="position", axis="y", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_posy.png"))
        gl.graph(self.config.camera_fps, graph_type="position", axis="z", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_posz.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="all", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_vel.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="x", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_velx.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="y", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_vely.png"))
        gl.graph(self.config.camera_fps, graph_type="velocity", axis="z", vehicle_name="LeadingUAV", filename=os.path.join(self.logger.parent_folder, "lead_velz.png"))

        gl.graph(self.config.camera_fps, graph_type="distance", axis="all", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "dist.png"))
        gl.graph(self.config.camera_fps, graph_type="distance", axis="x", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "distx.png"))
        gl.graph(self.config.camera_fps, graph_type="distance", axis="y", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "disty.png"))
        gl.graph(self.config.camera_fps, graph_type="distance", axis="z", vehicle_name="EgoUAV", filename=os.path.join(self.logger.parent_folder, "distz.png"))

        gl.plot_movement_3d(filename=os.path.join(self.logger.parent_folder,"movement.png"), path=self.leading_path)

    def finalize(self, status: Status_t):
        if self.done == True:
            return
        self.done = True
        self.status = _map_to_status_code(status)
        # Save the last evaluated bbox
        _, est_frame_info = self.egoUAV.moveToBoundingBoxAsync(self.bbox, self.orient, dt=(1/self.config.filter_freq_Hz))
        self.logger.update_frame(bbox=self.bbox, est_frame_info=est_frame_info)
        # Save any leftorver frames
        self.logger.save_frames(finalize=True)
        
        # Output some statistics
        print(f"Simulation run for {self.frame_idx/self.config.sim_fps} seconds")
        print(f"Execution time: {self.time_ns*1e-9} seconds")
        self.exit(status)

    def exit(self, status: Status_t):
        # Update the client, since an interrupt will result to bad behaviour of the client.
        # It will raise a "RuntimeError: IOLoop is already running" when trying to invoke any of it's methods
        self.client = airsim.MultirotorClient()

        # Free the simulation and reset the vehicles
        self.egoUAV.disable()
        self.leadingUAV.disable()
        self.client.reset()
        self.client.simPause(False)
        
        # Stop the logger
        self.logger.exit(status=status)
        self.export_graphs()
        print(f"\nCoSimulator exits with status: {status}\n")
