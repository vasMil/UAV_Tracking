import os
import time, datetime

import airsim
from torchvision.utils import save_image

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from utils.image import add_bbox_to_image, add_angle_info_to_image
from utils.simulation import sim_calculate_angle

# Make sure move_duration exceeds sleep_duration
# otherwise in each iteration of the game loop the
# leading vehicle will "run out of moves" before the next iteration
assert(config.move_duration > config.sleep_const)

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

    # Create a folder for the recording
    dt = datetime.datetime.now()
    recording_path = f"recordings/{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}{dt.second}/images"
    print(recording_path)
    os.makedirs(recording_path)

    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Vehicle List: {client.listVehicles()}\n")

    # Wait for the takeoff to complete
    leadingUAV = LeadingUAV("LeadingUAV")
    egoUAV = EgoUAV("EgoUAV")
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()

    try:
        # Pause the simulation
        client.simPause(True)

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
                    # Calculate the ground truth angle
                    sim_angle = sim_calculate_angle(egoUAV, leadingUAV)
                    # Perform the movement
                    future, estim_angle = egoUAV.moveToBoundingBoxAsync(bbox, time_interval=(0.5))
                    # Add info on the camera frame
                    camera_frame = add_bbox_to_image(camera_frame, bbox)
                    camera_frame = add_angle_info_to_image(camera_frame, estim_angle, sim_angle)
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
    except Exception as e:
        import traceback
        print("There was an error, writing setup file and releasing AirSim...")
        print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
        traceback.print_exc()
    finally:
        # Write a setup.txt file containing all the important configuration options used for
        # this run
        setup_path = os.path.join(recording_path,"../setup.txt")
        with open(setup_path, 'w') as f:
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
                    f"weight_pos_x, weight_pos_y, weight_pos_z = {config.weight_pos_x}, {config.weight_pos_y}, {config.weight_pos_z}\n"
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
    tracking_at_frequency(simulation_time_s=60, infer_freq_Hz=10)
