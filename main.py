import traceback

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

    # Run the simulation
    co_sim = CoSimulator()
    try:
        co_sim.start()
        while not co_sim.done and co_sim.status == 0:
            co_sim.advance()
    except Exception:
        print("There was an error, writing setup file and releasing AirSim...")
        print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
        traceback.print_exc()
        co_sim.finalize("Error")
    finally:
        co_sim.finalize("Time's up")
        co_sim.export_graphs()
