import traceback

import airsim
import numpy as np

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from tracking_logging.logger import Logger, GraphLogs
from utils.simulation import getSquarePathAroundPoint

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

    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Vehicle List: {client.listVehicles()}\n")
    # Reset the position of the UAVs (just to make sure)
    client.reset()
    # Wait for the takeoff to complete
    leadingUAV = LeadingUAV("LeadingUAV")
    egoUAV = EgoUAV("EgoUAV", filter_type="None")
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()
    # Move up so you minimize shadows
    leadingUAV.moveByVelocityAsync(0, 0, -5, 10)
    egoUAV.moveByVelocityAsync(0, 0, -5, 10)
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()
    # Move to a location with different signs for position values
    pos = egoUAV.getMultirotorState().kinematics_estimated.position.z_val
    egoUAV.moveToPositionAsync(-5, -5, pos).join()
    leadingUAV.moveToPositionAsync(-4.5, -5, pos).join()
    # Wait for the vehicles to stabilize
    import time
    time.sleep(10)
    # Setup the leadingUAV to move on a square path arount the egoUAV
    pos = egoUAV.getMultirotorState().kinematics_estimated.position
    leadingUAV.moveOnPathAsync(getSquarePathAroundPoint(pos.x_val, pos.y_val, pos.z_val, coord_frame_offset=leadingUAV.sim_global_coord_frame_origin, square_width=8))
    
    # Create a Logger
    logger = Logger(egoUAV,
                    leadingUAV,
                    sim_fps=sim_fps,
                    simulation_time_s=simulation_time_s,
                    camera_fps=camera_fps,
                    infer_freq_Hz=infer_freq_Hz,
                    leadingUAV_update_vel_interval_s=leadingUAV_update_vel_interval_s
                )

    try:
        # Pause the simulation
        client.simPause(True)

        camera_frame = egoUAV._getImage()
        bbox, prev_bbox = None, None
        score, prev_score = None, None
        orient, prev_orient = egoUAV.getPitchRollYaw(), egoUAV.getPitchRollYaw()
        for frame_idx in range(simulation_time_s*sim_fps):
            print(f"\nFRAME: {frame_idx}")
            logger.step(prev_bbox != None)

            if frame_idx % round(sim_fps/camera_fps) == 0:
                camera_frame = egoUAV._getImage()

            # Update the leadingUAV velocity every update_vel_s*sim_fps frames
            # if frame_idx % (leadingUAV_update_vel_interval_s*sim_fps) == 0:
                # leadingUAV.random_move(leadingUAV_update_vel_interval_s)

            # Get a bounding box and move towards the previous detection
            # this way we also simulate the delay between the capture of the frame
            # and the output of the NN for this frame.
            if frame_idx % round(sim_fps/infer_freq_Hz) == 0:
                # Run egoUAV's detection net, save the frame with all
                # required information. Hold on to the bbox, to move towards it when the
                # next frame for evaluation is captured.
                bbox, score = egoUAV.net.eval(camera_frame, 0)
                orient = egoUAV.getPitchRollYaw()

                # Perform the movement for the previous detection
                egoUAV.moveToBoundingBoxAsync(prev_bbox, prev_orient, dt=(1/infer_freq_Hz))
                if prev_score and prev_score >= config.score_threshold:
                    print(f"UAV detected {prev_score}, moving towards it...")
                else:
                    print("Lost tracking!!!")

                # Update
                prev_bbox = bbox
                prev_score = score
                prev_orient = orient
            else:
                bbox, score = None, None

            if frame_idx % round(sim_fps/camera_fps) == 0:
                logger.save_frame(camera_frame, bbox, orient)

            # Restart the simulation for a few seconds to match
            # the desired sim_fps
            client.simContinueForTime(1/sim_fps)

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
        logger.write_video()

        client.simPause(False)
        egoUAV.disable()
        leadingUAV.disable()
        client.reset()

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
    tracking_at_frequency(simulation_time_s=10, infer_freq_Hz=30)
    # gl = GraphLogs(pickle_file="../proodos/keypoint_presentation/ssd10_log.pkl")
    # gl.graph_distance(sim_fps=60)
    # gl = GraphLogs(pickle_file="../proodos/keypoint_presentation/ssd30_log.pkl")
    # gl.graph_distance(sim_fps=60)
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

    # Test the Kalman filter
    # import numpy as np
    # from controller.KalmanFilter import KalmanFilter
    # X_init = np.zeros([6, 1]); X_init[0][0] = 3.5
    # kf = KalmanFilter(X_init, np.zeros([6, 6]), np.zeros([6, 6]), np.zeros([3, 3]))
    # print(kf.step(np.expand_dims(np.pad(np.array([4, 0, 0]), (0,3)), axis=1), dt=0.1))
    # print(kf.step(np.expand_dims(np.pad(np.array([5, 0, 0]), (0,3)), axis=1), dt=0.2))
    # print(kf.step(np.expand_dims(np.pad(np.array([5.6, 0, 0]), (0,3)), axis=1), dt=0.1))
    # print(kf.step(np.expand_dims(np.pad(np.array([6.2, 0, 0]), (0,3)), axis=1), dt=0.1))

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

    # import math
    # import numpy as np
    # from utils.operations import rotate_to_yaw
    # vel = rotate_to_yaw(math.radians(360), np.array([[math.sqrt(8)], [0], [0]]))
    # print(vel)
