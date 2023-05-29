import traceback

import airsim

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from tracking_logging.logger import Logger, GraphLogs

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
    egoUAV = EgoUAV("EgoUAV", filter="None")
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()

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
        for frame_idx in range(simulation_time_s*sim_fps):
            print(f"\nFRAME: {frame_idx}")
            logger.step(prev_bbox != None)

            if frame_idx % round(sim_fps/camera_fps) == 0:
                camera_frame = egoUAV._getImage()

            # Update the leadingUAV velocity every update_vel_s*sim_fps frames
            if frame_idx % (leadingUAV_update_vel_interval_s*sim_fps) == 0:
                leadingUAV.random_move(leadingUAV_update_vel_interval_s)

            # Get a bounding box and move towards the previous detection
            # this way we also simulate the delay between the capture of the frame
            # and the output of the NN for this frame.
            if frame_idx % round(sim_fps/infer_freq_Hz) == 0:
                # Run egoUAV's detection net, save the frame with all
                # required inforamtion. Hold on to the bbox, to move towards it when the
                # next frame for evaluation is captured.
                bbox, score = egoUAV.net.eval(camera_frame)

                # Perform the movement for the previous detection
                egoUAV.moveToBoundingBoxAsync(prev_bbox, dt=(1/infer_freq_Hz))
                if prev_score and prev_score >= config.score_threshold:
                    print(f"UAV detected {prev_score}, moving towards it...")
                else:
                    print("Lost tracking!!!")

                # Update
                prev_bbox = bbox
                prev_score = score
            else:
                bbox, score = None, None

            if frame_idx % round(sim_fps/camera_fps) == 0:
                logger.save_frame(camera_frame, bbox)

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
    #     csv_file="/home/airsim_user/UAV_Tracking/data/empty_map/train/empty_map_positions.csv",
    #     root_dir="/home/airsim_user/UAV_Tracking/data/empty_map/train/",
    #     num_samples=100
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
    tracking_at_frequency(simulation_time_s=60, infer_freq_Hz=10)
    # gl = GraphLogs(pickle_file="recordings/diogenis_ssd30_vel5/log.pkl")
    # gl.graph_distance(sim_fps=60)

    # Run inference frequency benchmark
    # from nets.DetectionNets import Detection_SSD, Detection_FasterRCNN
    # ssd = Detection_SSD(root_test_dir="/home/airsim_user/UAV_Tracking/data/empty_map/test",
    # 		          json_test_labels="/home/airsim_user/UAV_Tracking/data/empty_map/test/empty_map.json")
    # ssd.get_inference_frequency(num_tests=10, warmup=2, cudnn_benchmark=True)
    # ssd.load("nets/checkpoints/ssd300.checkpoint")
    # ssd.plot_losses()
    # rcnn = Detection_FasterRCNN(root_test_dir="/home/airsim_user/UAV_Tracking/data/empty_map/test",
    #        	                 json_test_labels="/home/airsim_user/UAV_Tracking/data/empty_map/test/empty_map.json")
    # rcnn.load("nets/checkpoints/rcnn100.checkpoint")
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
    # print(estimate_measurement_noise(network=Detection_SSD()))
    # from utils.kalman_filter import estimate_process_noise, estimate_measurement_noise
    # from nets.DetectionNets import Detection_SSD
    # print(estimate_measurement_noise(network=Detection_SSD()))
