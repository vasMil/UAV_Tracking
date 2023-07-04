import traceback
import os
import time

import airsim

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from tracking_logging.logger import Logger, GraphLogs
from utils.simulation import getTestPath

def tracking_at_frequency() -> Logger:
    """
    Runs AirSim step by step at a framerate of sim_fps.
    This way you may simulate any given inference_frequency, even if your
    hardware is slow.
    
    This function can be configured from constants set in GlobalConfig.

    Returns:
    A Logger instance that logged info at camera_fps
    """
    if config.sim_fps < config.camera_fps:
        raise Exception("sim_fps cannot be less than camera_fps")
    if config.camera_fps < config.infer_freq_Hz:
        raise Exception("camera_fps cannot be less that infer_freq_Hz")
    if config.filter_type == "KF" and config.camera_fps < config.infer_freq_Hz:
        Warning("Since you allow the UAV to move (change direction) faster than the camera may record, you may experience some flickering")

    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Vehicle List: {client.listVehicles()}\n")
    # Reset the position of the UAVs (just to make sure)
    client.reset()
    # Wait for the takeoff to complete
    leadingUAV = LeadingUAV("LeadingUAV")
    egoUAV = EgoUAV("EgoUAV", filter_type=config.filter_type)
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()
    # Move up so you minimize shadows
    leadingUAV.moveByVelocityAsync(0, 0, -5, 10)
    egoUAV.moveByVelocityAsync(0, 0, -5, 10)
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()
    # Wait for the vehicles to stabilize
    time.sleep(10)

    # Create a Logger
    logger = Logger(egoUAV=egoUAV,
                    leadingUAV=leadingUAV,
                    sim_fps=config.sim_fps,
                    simulation_time_s=config.simulation_time_s,
                    camera_fps=config.camera_fps,
                    infer_freq_Hz=config.infer_freq_Hz,
                    leadingUAV_update_vel_interval_s=config.leadingUAV_update_vel_interval_s
    )

    try:
        # Pause the simulation
        client.simPause(True)

        camera_frame = egoUAV._getImage()
        bbox, prev_bbox = None, None
        score, prev_score = None, None
        orient, prev_orient = egoUAV.getPitchRollYaw(), egoUAV.getPitchRollYaw()
        camera_frame_idx, prev_camera_frame_idx = -1, -1

        if config.filter_type == "KF":
            filter_freq_Hz = config.filter_freq_Hz
        else:
            filter_freq_Hz = config.infer_freq_Hz

        for frame_idx in range(config.simulation_time_s*config.sim_fps):
            print(f"\nFRAME: {frame_idx}")

            if frame_idx % round(config.sim_fps/config.camera_fps) == 0:
                camera_frame = egoUAV._getImage()
                logger.create_frame(camera_frame,
                                    is_bbox_frame=(frame_idx % round(config.sim_fps/config.infer_freq_Hz) == 0)
                )
                camera_frame_idx += 1

            # Update the leadingUAV velocity every update_vel_s*sim_fps frames
            if frame_idx % (config.leadingUAV_update_vel_interval_s*config.sim_fps) == 0:
                # leadingUAV.random_move(config.leadingUAV_update_vel_interval_s)
                if frame_idx == 0:
                    leadingUAV.moveOnPathAsync(getTestPath(leadingUAV.simGetGroundTruthKinematics().position))

            # Get a bounding box and move towards the previous detection
            # this way we also simulate the delay between the capture of the frame
            # and the output of the NN for this frame.
            if frame_idx % round(config.sim_fps/config.infer_freq_Hz) == 0:
                # Run egoUAV's detection net, save the frame with all
                # required information. Hold on to the bbox, to move towards it when the
                # next frame for evaluation is captured.
                bbox, score = egoUAV.net.eval(camera_frame, 0)
                orient = egoUAV.getPitchRollYaw()

                # There is no way we have a bbox when just inserting the first frame to the logger
                if frame_idx != 0:
                    # Perform the movement for the previous detection
                    _, est_frame_info = egoUAV.moveToBoundingBoxAsync(prev_bbox, prev_orient, dt=(1/filter_freq_Hz))
                    if prev_score and prev_score >= config.score_threshold:
                        print(f"UAV detected {prev_score}, moving towards it...")
                    else:
                        print("Lost tracking!!!")
                
                    # Update the frame in the logger
                    logger.update_frame(bbox=prev_bbox, est_frame_info=est_frame_info, camera_frame_idx=prev_camera_frame_idx)
                    logger.save_frames()

                # Update
                prev_bbox = bbox
                prev_score = score
                prev_orient = orient
                prev_camera_frame_idx = camera_frame_idx

            # If we haven't yet decided on the movement, due to the inference frequency limitation
            # and we use a Kalman Filter, check if it is time to advance the KF.
            elif config.filter_type == "KF" and frame_idx % round(config.sim_fps/filter_freq_Hz):
                egoUAV.advanceUsingFilter(dt=(1/filter_freq_Hz))

            # Continue the simulation for a few seconds to match
            # the desired sim_fps
            client.simContinueForTime(1/config.sim_fps)

        # Save the last evaluated bbox
        _, est_frame_info = egoUAV.moveToBoundingBoxAsync(bbox, orient, dt=(1/filter_freq_Hz))
        logger.update_frame(bbox, est_frame_info, camera_frame_idx)
        # Save any leftorver frames
        logger.save_frames(finalize=True)

    except Exception:
        print("There was an error, writing setup file and releasing AirSim...")
        print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
        traceback.print_exc()

    finally:
        # Write a setup.txt file containing all the important configuration options used for
        # this run
        logger.write_setup()
        logger.dump_logs()

        # Write the mp4 file
        logger.save_frames(finalize=True)
        logger.write_video()

        client.simPause(False)
        egoUAV.disable()
        leadingUAV.disable()
        client.reset()

        return logger

if __name__ == '__main__':
    # Generate data
    # from gendata import generate_training_data
    # generate_training_data(
    #     csv_file="/home/airsim_user/UAV_Tracking/data/empty_map/shadows/empty_map_positions.csv",
    #     root_dir="/home/airsim_user/UAV_Tracking/data/empty_map/shadows/",
    #     num_samples=500
    # )

    # Train the NNs
    # from nets.DetectionNets import Detection_SSD
    # ssd = Detection_SSD(
    #     root_train_dir="/home/airsim_user/UAV_Tracking/data/empty_map/train",
    #     json_train_labels="/home/airsim_user/UAV_Tracking/data/empty_map/train/empty_map.json",
    #     root_test_dir="/home/airsim_user/UAV_Tracking/data/empty_map/test",
    #     json_test_labels="/home/airsim_user/UAV_Tracking/data/empty_map/test/empty_map.json"
    # )
    # torch.backends.cudnn.benchmark = True
    # for i in range(0, 30)*10:
    #     ssd.train(num_epochs=10)
    #     ssd.save(f"nets/checkpoints/ssd{i+10-1}.checkpoint")

    # Run the simulation
    logger = tracking_at_frequency()
    gl = GraphLogs(frame_info=logger.updated_info_per_frame)
    gl.graph_distance(fps=config.camera_fps, filename=(os.path.join(logger.parent_folder, "dist_graph.png")))
    gl.graph_velocities(fps=config.camera_fps, filename=(os.path.join(logger.parent_folder, "vel_graph.png")))
    gl.graph_positions(fps=config.camera_fps, filename=(os.path.join(logger.parent_folder, "pos_graph.png")))
    gl.graph_velocity_on_axis(fps=config.camera_fps, filename=(os.path.join(logger.parent_folder, "xvel_graph.png")), axis="x", vehicle_name="lead")
    gl.graph_velocity_on_axis(fps=config.camera_fps, filename=(os.path.join(logger.parent_folder, "yvel_graph.png")), axis="y", vehicle_name="lead")
    gl.graph_velocity_on_axis(fps=config.camera_fps, filename=(os.path.join(logger.parent_folder, "zvel_graph.png")), axis="z", vehicle_name="lead")

    # Run inference frequency benchmark
    # from nets.DetectionNets import Detection_SSD, Detection_FasterRCNN
    # ssd = Detection_SSD(root_test_dir="/home/airsim_user/UAV_Tracking/data/empty_map/test",
    # 		          json_test_labels="/home/airsim_user/UAV_Tracking/data/empty_map/test/empty_map.json")
    # # ssd.get_inference_frequency(num_tests=10, warmup=2, cudnn_benchmark=True)
    # ssd.load("nets/checkpoints/ssd300.checkpoint")
    # ssd.plot_losses()
    # rcnn = Detection_FasterRCNN(root_test_dir="/home/airsim_user/UAV_Tracking/data/empty_map/test",
    #        	                 json_test_labels="/home/airsim_user/UAV_Tracking/data/empty_map/test/empty_map.json")
    # rcnn.load("nets/checkpoints/rcnn120.checkpoint")
    # rcnn.plot_losses()
    # rcnn.get_inference_frequency(num_tests=10, warmup=2, cudnn_benchmark=True)

    # Test Kalman utility functions
    # from utils.kalman_filter import estimate_process_noise, estimate_measurement_noise
    # from nets.DetectionNets import Detection_SSD
    # print(estimate_measurement_noise(network=Detection_SSD(), num_samples=10000))

    # Test distance estimation
    # import math
    # import numpy as np
    # from utils.operations import vector_transformation
    # egoUAV = EgoUAV("EgoUAV")
    # leadingUAV = LeadingUAV("LeadingUAV")
    # leadingUAV.lastAction.join()
    # egoUAV.lastAction.join()

    # # img = egoUAV._getImage()
    # # save_image(img, "temp.png")
    # # bbox, _ = egoUAV.net.eval(img)
    # # estim_dist = egoUAV.get_distance_from_bbox(bbox)
    # # true_dist = np.expand_dims((leadingUAV.simGetObjectPose().position - egoUAV.simGetObjectPose().position).to_numpy_array(), axis=1)
    # # error = estim_dist - true_dist

    # # print(f"\n estim_dist \n {estim_dist}")
    # # print(f"\n true_dist \n {true_dist}")
    # # print(f"\n error \n {error}")
    # # Place the leadingUAV
    # ego_z = egoUAV.getMultirotorState().kinematics_estimated.position.z_val
    # leadingUAV.moveToPositionAsync(-3.5, 10, ego_z, velocity=5).join()
    # # Rotate the EgoUAV to an angle so the leadingUAV is visible
    # egoUAV.rotateToYawAsync(90).join()
    # # Try to move towards the LeadingUAV
    # bbox, score = egoUAV.net.eval(egoUAV._getImage())
    # # Calculate the magnitude of the distance between the two UAVs
    # if not bbox:
    #     egoUAV.disable()
    #     leadingUAV.disable()
    #     egoUAV.client.reset()
    #     raise Exception("No bbox found")
    # # Calculate the expected distance
    # expected_dist = np.array([[0], [10], [0]])
    # # Estimate the distance
    # estimated_dist = egoUAV.get_distance_from_bbox(bbox)
    # print(egoUAV.getPitchRollYaw())
    # estimated_dist = vector_transformation(*(egoUAV.getPitchRollYaw()), vec=estimated_dist) # type: ignore
    # # estimated_dist = math.sqrt(estimated_dist[0]**2 + estimated_dist[1]**2 + estimated_dist[2]**2) # type: ignore
    # print(f"expected_dist: {expected_dist}")
    # print(f"estimated_dist: {estimated_dist}")
    # egoUAV.disable()
    # leadingUAV.disable()
    # egoUAV.client.reset()
