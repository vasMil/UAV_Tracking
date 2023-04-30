import os
import time, datetime
import multiprocessing as mp

import airsim
import torch
from torchvision.utils import save_image

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from models.BoundingBox import BoundingBox

# Make sure move_duration exceeds sleep_duration
# otherwise in each iteration of the game loop the
# leading vehicle will "run out of moves" before the next iteration
assert(config.move_duration > config.sleep_const)

def leadingUAV_loop(exit_signal, port: int, time_interval: int):
    leadingUAV = LeadingUAV("LeadingUAV", port, config.leadingUAV_seed)
    leadingUAV.lastAction.join()
    with exit_signal.get_lock():
        exit_status = exit_signal.value # type: ignore
    while not exit_status:
        leadingUAV.random_move(time_interval)
        time.sleep(time_interval)
        with exit_signal.get_lock():
            exit_status = exit_signal.value # type: ignore
    leadingUAV.disable()

def egoUAV_loop(exit_signal, port: int):
    """
    Follows the leadingUAV, using it's NN, by predicting the bounding box
    and then moving towards it.
    """
    egoUAV = EgoUAV("EgoUAV", port)
    egoUAV.lastAction.join()
    with exit_signal.get_lock():
        exit_status = exit_signal.value # type: ignore
        
    while not exit_status:
        img = egoUAV._getImage()
        bbox, score = egoUAV.net.eval(img)
        future = egoUAV.moveToBoundingBoxAsync(bbox)
        print(f"UAV detected {score}, moving towards it..." if future else "Lost tracking!!!")
        with exit_signal.get_lock():
            exit_status = exit_signal.value # type: ignore
    egoUAV.disable()

def simple_tracking():
    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Vehicle List: {client.listVehicles()}\n")
    # Start recording
    print("\n*****************")
    print("Recording Started")
    print("*****************\n")
    client.startRecording()

    # Communication variables
    exit_signal = mp.Value('i', False)

    # Create two processes
    leadingUAV_process = mp.Process(target=leadingUAV_loop, args=(exit_signal, config.port, 2))
    egoUAV_process = mp.Process(target=egoUAV_loop, args=(exit_signal, config.port))
    leadingUAV_process.start()
    egoUAV_process.start()

    time.sleep(120)
    with exit_signal.get_lock():
        exit_signal.value = True # type: ignore
    
    leadingUAV_process.join()
    egoUAV_process.join()

    # Stop recording
    client.stopRecording()
    print("\n*****************")
    print("Recording Stopped")
    print("*****************\n")

def add_bbox_to_image(image: torch.Tensor, bbox: BoundingBox) -> None:
    x1 = max(round(bbox.x1), 0)
    x2 = min(round(bbox.x2), config.img_width-1)
    y1 = max(round(bbox.y1), 0)
    y2 = min(round(bbox.y2), config.img_height-1)
    for i, color in enumerate([255, 0, 0]):
        image[i, y1, x1:x2] = color
        image[i, y2, x1:x2] = color
        image[i, y1:y2, x1] = color
        image[i, y1:y2, x2] = color

def tracking_at_frequency(sim_fps: int = 60,
                          simulation_time_s: int = 120,
                          camera_fps: int = 30,
                          infer_freq_Hz: int = 30,
                          leadingUAV_update_vel_interval_s: int = 2
                        ) -> None:
    """
    Runs AirSim frame by frame at a framerate of sim_fps.
    This way you may simulate any given inference_frequency, even if your
    hardware is slow.

    Args:
    - sim_fps: The frames per second that the simulation runs at (cannot be 
    arbitrarily small).
    - simulation_time_s: The amount of simulation seconds should this run for.
    - camera_fps: EgoUAV's camera frame rate
    - infer_freq_Hz: The frequency at which EgoUAV's network inference operates at.
    - leadingUAV_update_vel_interval_s: How many simulation seconds between consecutive
    leadingUAV.random_move() calls.
    """
    if sim_fps < camera_fps:
        raise Exception("sim_fps cannot be less than camera_fps")
    if camera_fps < infer_freq_Hz:
        raise Exception("camera_fps cannot be less that infer_freq_Hz")

    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Vehicle List: {client.listVehicles()}\n")

    # Wait for the takeoff to complete
    leadingUAV = LeadingUAV("LeadingUAV")
    egoUAV = EgoUAV("EgoUAV")
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()

    # Pause the simulation
    client.simPause(True)

    # Create a folder for the recording
    dt = datetime.datetime.now()
    recording_path = f"recordings/{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}{dt.second}/images"
    print(recording_path)
    os.makedirs(recording_path)

    camera_frame = egoUAV._getImage()
    camera_frame_idx = 0
    frame_saved = False
    for frame_idx in range(simulation_time_s*sim_fps):
        print(f"\nFRAME: {frame_idx}")
        if frame_idx % round(sim_fps/camera_fps) == 0:
            camera_frame = egoUAV._getImage()
            frame_saved = False
            camera_frame_idx += 1

        # Update the leadingUAV velocity every update_vel_s*sim_fps frames
        if frame_idx % (leadingUAV_update_vel_interval_s*sim_fps) == 0:
            leadingUAV.random_move(leadingUAV_update_vel_interval_s)
        
        # Get a bounding box and move towards it
        if frame_idx % round(sim_fps/infer_freq_Hz) == 0:
            # Get a frame, run egoUAV's detection net and move
            # towards this position
            bbox, score = egoUAV.net.eval(camera_frame)
            if bbox and score and score >= config.score_threshold:
                future = egoUAV.moveToBoundingBoxAsync(bbox, time_interval=0.5)
                add_bbox_to_image(camera_frame, bbox)
                save_image(camera_frame, f"{recording_path}/img_EgoUAV_{time.time_ns()}.png")
                frame_saved = True
                print(f"UAV detected {score}, moving towards it..." if future else "Lost tracking!!!")
        
        # If the network did not save last camera frame, save it to preserve the framerate
        if not frame_saved and frame_idx % round(sim_fps/camera_fps) == 0:
            save_image(camera_frame, f"{recording_path}/img_EgoUAV_{time.time_ns()}.png")
            frame_saved = True

        # Restart the simulation for a few seconds to match
        # the desired sim_fps
        client.simContinueForTime(1/sim_fps)
    
    # Write a setup.txt file containing all the important configuration options used for
    # this run
    setup_path = os.path.join(recording_path,"../setup.txt")
    with open(setup_path, 'w') as f:
        f.write(f"# The upper an lower limit for the velocity on each axis of both UAVs\n"
                f"max_vx, max_vy, max_vz = {config.max_vx},  {config.max_vy},  {config.max_vz}\n"
                f"min_vx, min_vy, min_vz = {config.min_vx}, {config.min_vy}, {config.min_vz}\n"
                f"\n"
                f"# The minimum score, for which a detection is considered\n"
                f"# valid and thus is translated to EgoUAV movement.\n"
                f"score_threshold = {config.score_threshold}\n"
                f"\n"
                f"# The magnitude of the velocity vector (in 3D space)\n"
                f"uav_velocity = {config.uav_velocity}\n"
                f"\n"
                f"# Recording setup\n"
                f"sim_fps = {sim_fps}\n"
                f"simulation_time_s = {simulation_time_s}\n"
                f"camera_fps = {camera_fps}\n"
                f"infer_freq_Hz = {infer_freq_Hz}\n"
                f"leadingUAV_update_vel_interval_s = {leadingUAV_update_vel_interval_s}\n"
        )

    client.simPause(False)
    egoUAV.disable()
    leadingUAV.disable()
    client.reset()

def tracking_at_frequency2(sim_fps: int = 60,
                          simulation_time_s: int = 120,
                          camera_fps: int = 30,
                          infer_freq_Hz: int = 30,
                          leadingUAV_update_vel_interval_s: int = 2
                        ) -> None:
    """
    Runs AirSim frame by frame at a framerate of sim_fps.
    This way you may simulate any given inference_frequency, even if your
    hardware is slow.

    Args:
    - sim_fps: The frames per second that the simulation runs at (cannot be 
    arbitrarily small).
    - simulation_time_s: The amount of simulation seconds should this run for.
    - camera_fps: EgoUAV's camera frame rate
    - infer_freq_Hz: The frequency at which EgoUAV's network inference operates at.
    - leadingUAV_update_vel_interval_s: How many simulation seconds between consecutive
    leadingUAV.random_move() calls.
    """
    if sim_fps < camera_fps:
        raise Exception("sim_fps cannot be less than camera_fps")
    if camera_fps < infer_freq_Hz:
        raise Exception("camera_fps cannot be less that infer_freq_Hz")

    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Vehicle List: {client.listVehicles()}\n")

    # Wait for the takeoff to complete
    leadingUAV = LeadingUAV("LeadingUAV")
    egoUAV = EgoUAV("EgoUAV")
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()

    # Pause the simulation
    client.simPause(True)

    # Create a folder for the recording
    dt = datetime.datetime.now()
    recording_path = f"recordings/{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}{dt.second}/images"
    print(recording_path)
    os.makedirs(recording_path)

    camera_frame = egoUAV._getImage()
    camera_frame_idx = 0
    frame_saved = False
    for frame_idx in range(simulation_time_s*sim_fps):
        print(f"\nFRAME: {frame_idx}")
        if frame_idx % round(sim_fps/camera_fps) == 0:
            camera_frame = egoUAV._getImage()
            frame_saved = False
            camera_frame_idx += 1

        # Update the leadingUAV velocity every update_vel_s*sim_fps frames
        if frame_idx % (leadingUAV_update_vel_interval_s*sim_fps) == 0:
            leadingUAV.random_move(leadingUAV_update_vel_interval_s)
        
        # Get a bounding box and move towards it
        if frame_idx % round(sim_fps/infer_freq_Hz) == 0:
            # Get a frame, run egoUAV's detection net and move
            # towards this position
            bbox, score = egoUAV.net.eval(camera_frame)
            if bbox and score and score >= config.score_threshold:
                future = egoUAV.moveToBoundingBoxAsync(bbox)
                add_bbox_to_image(camera_frame, bbox)
                save_image(camera_frame, f"{recording_path}/img_EgoUAV_{time.time_ns()}.png")
                frame_saved = True
                print(f"UAV detected {score}, moving towards it..." if future else "Lost tracking!!!")
        
        # If the network did not save last camera frame, save it to preserve the framerate
        if not frame_saved and frame_idx % round(sim_fps/camera_fps) == 0:
            save_image(camera_frame, f"{recording_path}/img_EgoUAV_{time.time_ns()}.png")
            frame_saved = True

        # Restart the simulation for a few seconds to match
        # the desired sim_fps
        client.simContinueForTime(1/sim_fps)
    
    # Write a setup.txt file containing all the important configuration options used for
    # this run
    setup_path = os.path.join(recording_path,"../setup.txt")
    with open(setup_path, 'w') as f:
        f.write(f"# The upper an lower limit for the velocity on each axis of both UAVs\n"
                f"max_vx, max_vy, max_vz = {config.max_vx},  {config.max_vy},  {config.max_vz}\n"
                f"min_vx, min_vy, min_vz = {config.min_vx}, {config.min_vy}, {config.min_vz}\n"
                f"\n"
                f"# The minimum score, for which a detection is considered\n"
                f"# valid and thus is translated to EgoUAV movement.\n"
                f"score_threshold = {config.score_threshold}\n"
                f"\n"
                f"# The magnitude of the velocity vector (in 3D space)\n"
                f"uav_velocity = {config.uav_velocity}\n"
                f"\n"
                f"# Recording setup\n"
                f"sim_fps = {sim_fps}\n"
                f"simulation_time_s = {simulation_time_s}\n"
                f"camera_fps = {camera_fps}\n"
                f"infer_freq_Hz = {infer_freq_Hz}\n"
                f"leadingUAV_update_vel_interval_s = {leadingUAV_update_vel_interval_s}\n"
        )

    client.simPause(False)
    egoUAV.disable()
    leadingUAV.disable()
    client.reset()

if __name__ == '__main__':
    print("VELOCITY")
    tracking_at_frequency(simulation_time_s=60, infer_freq_Hz=30)
