from models.CoSimulator import CoSimulator

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
    co_sim = CoSimulator()
    co_sim.start()
    while not co_sim.done and co_sim.status == 0:
        co_sim.advance()
    
    co_sim.export_graphs()

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
