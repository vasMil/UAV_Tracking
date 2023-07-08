from typing import Literal
import traceback
import os
import time

import airsim

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from tracking_logging.logger import Logger, GraphLogs
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
            if config.filter_type == "KF":
                self.filter_freq_Hz = filter_freq_Hz
            else:
                self.filter_freq_Hz = infer_freq_Hz
            self.done: bool = False
            self.status: int = 0

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
                                 leadingUAV_update_vel_interval_s=leadingUAV_update_vel_interval_s
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
        print(f"\nFRAME: {self.frame_idx}")

        if self.frame_idx % round(self.sim_fps/self.camera_fps) == 0:
            self.hook_camera_frame_capture()

        # Update the leadingUAV velocity every update_vel_s*sim_fps frames
        if self.frame_idx % (self.leadingUAV_update_vel_interval_s*self.sim_fps) == 0:
            self.hook_leadingUAV_move()

        # Get a bounding box and move towards the previous detection
        # this way we also simulate the delay between the capture of the frame
        # and the output of the NN for this frame.
        if self.frame_idx % round(self.sim_fps/self.infer_freq_Hz) == 0:
            self.hook_net_inference()

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
            self.finalize()

    def advance(self):
        try:
            self._advance()
        except Exception:
            print("There was an error, writing setup file and releasing AirSim...")
            print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
            traceback.print_exc()
            self.status = -1
            self.exit()

    def hook_camera_frame_capture(self):
        self.camera_frame = self.egoUAV._getImage()
        self.logger.create_frame(self.camera_frame,
                                 is_bbox_frame=(self.frame_idx % round(self.sim_fps/self.infer_freq_Hz) == 0)
        )
        self.camera_frame_idx += 1

    def hook_leadingUAV_move(self):
        # self.leadingUAV.random_move(self.leadingUAV_update_vel_interval_s)
        if self.frame_idx == 0:
            self.leadingUAV.moveOnPathAsync(getTestPath(self.leadingUAV.simGetGroundTruthKinematics().position))

    def hook_net_inference(self):
        # Run egoUAV's detection net, save the frame with all
        # required information. Hold on to the bbox, to move towards it when the
        # next frame for evaluation is captured.
        bbox, score = self.egoUAV.net.eval(self.camera_frame, 0)
        orient = self.egoUAV.getPitchRollYaw()

        # There is no way we have a bbox when just inserting the first frame to the logger
        if self.frame_idx != 0:
            # Perform the movement for the previous detection
            _, est_frame_info = self.egoUAV.moveToBoundingBoxAsync(self.prev_bbox, self.prev_orient, dt=(1/self.filter_freq_Hz))
            if self.prev_score and self.prev_score >= config.score_threshold:
                print(f"UAV detected {self.prev_score}, moving towards it...")
            else:
                print("Lost tracking!!!")
        
            # Update the frame in the logger
            self.logger.update_frame(bbox=self.prev_bbox, est_frame_info=est_frame_info, camera_frame_idx=self.prev_camera_frame_idx)
            self.logger.save_frames()

        # Update
        self.prev_bbox = bbox
        self.prev_score = score
        self.prev_orient = orient
        self.prev_camera_frame_idx = self.camera_frame_idx

    def hook_filter_advance_only(self):
        self.egoUAV.advanceUsingFilter(dt=(1/self.filter_freq_Hz))


    def finalize(self):
        # Save the last evaluated bbox
        _, est_frame_info = self.egoUAV.moveToBoundingBoxAsync(self.bbox, self.orient, dt=(1/self.filter_freq_Hz))
        self.logger.update_frame(self.bbox, est_frame_info, self.camera_frame_idx)
        # Save any leftorver frames
        self.logger.save_frames(finalize=True)

        self.exit()


    def exit(self):
        self.done = True
        # Free the simulation and reset the vehicles
        self.client.simPause(False)
        self.egoUAV.disable()
        self.leadingUAV.disable()
        self.client.reset()

        # Write a setup.txt file containing all the important configuration options used for
        # this run
        self.logger.write_setup()
        self.logger.dump_logs()

        # Write the mp4 file
        self.logger.save_frames(finalize=True)
        self.logger.write_video()


    def export_graphs(self):
        gl = GraphLogs(frame_info=self.logger.updated_info_per_frame)
        gl.graph_distance(fps=self.camera_fps,
                          filename=(os.path.join(self.logger.parent_folder, "dist_graph.png"))
        )
        gl.graph_velocities(fps=self.camera_fps,
                            filename=(os.path.join(self.logger.parent_folder, "vel_graph.png"))
        )
        gl.graph_positions(fps=self.camera_fps,
                           filename=(os.path.join(self.logger.parent_folder, "pos_graph.png"))
        )
        gl.graph_velocity_on_axis(fps=self.camera_fps,
                                  filename=(os.path.join(self.logger.parent_folder, "xvel_graph.png")),
                                  axis="x",
                                  vehicle_name="lead"
        )
        gl.graph_velocity_on_axis(fps=self.camera_fps,
                                  filename=(os.path.join(self.logger.parent_folder, "yvel_graph.png")),
                                  axis="y",
                                  vehicle_name="lead"
        )
        gl.graph_velocity_on_axis(fps=self.camera_fps,
                                  filename=(os.path.join(self.logger.parent_folder, "zvel_graph.png")),
                                  axis="z",
                                  vehicle_name="lead"
        )