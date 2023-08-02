

if __name__ == '__main__':
    # # Generate (and clean) training data
    # from nets.DetectionNets import Detection_FasterRCNN
    # from gendata import generate_data_using_segmentation, clean_generated_frames_and_bboxes
    # generate_data_using_segmentation(3000, frames_root_path="data/empty_map/test/", json_path="data/empty_map/test/bboxes.json")
    # rcnn = Detection_FasterRCNN()
    # rcnn.load("nets/checkpoints/rcnn100.checkpoint")
    # clean_generated_frames_and_bboxes(frames_root_path="data/empty_map/test/", json_path="data/empty_map/test/bboxes.json", net=rcnn)

    # Train the NNs
    import torch.backends.cudnn
    from nets.DetectionNets import Detection_SSD
    ssd = Detection_SSD(root_train_dir="data/empty_map/train", json_train_labels="data/empty_map/train/bboxes.json",
                        root_test_dir="data/empty_map/test", json_test_labels="data/empty_map/test/bboxes.json")
    
    torch.backends.cudnn.benchmark = True
    save_step = 50
    map_step = 10
    step = map_step
    for i in range(0, 300, step):
        ssd.train(num_epochs=step)
        if i != 0 and i % map_step == 0:
            ssd.calculate_metrics()
            ssd.plot_mAPs("nets/checkpoints/ssd/ssd_mAP_50.png", "map_50")
            ssd.plot_mAPs("nets/checkpoints/ssd/ssd_mAP_75.png", "map_75")
        if i != 0 and i % save_step == 0:
            ssd.save(f"nets/checkpoints/ssd/ssd{i+step}.checkpoint")
            ssd.plot_losses("nets/checkpoints/ssd/ssd_losses.png")
            if i == 150:
                save_step = 10


    # # Experiments
    # import traceback
    # import shutil

    # from config import DefaultCoSimulatorConfig
    # from models.CoSimulator import CoSimulator
    # from utils.data import plot_for_path
    # num_tests = 1
    # best_freq = 30
    # for v in range(5, 11):
        # for i in range(num_tests):
            # config = DefaultCoSimulatorConfig(sim_fps=20, camera_fps=20, infer_freq_Hz=20, filter_freq_Hz=20)
            # co_sim = CoSimulator(config=config, log_folder="recordings/", movement="Path", path_version="v0", display_terminal_progress=False)
            # try:
            #     co_sim.start()
            #     while not co_sim.done and co_sim.status == 0:
            #         co_sim.advance()
            # except Exception:
            #     co_sim.finalize("Error")
            #     print("There was an error, writing setup file and releasing AirSim...")
            #     print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
            #     traceback.print_exc()
            # finally:
            #     co_sim.finalize("Time's up")
            #     shutil.rmtree(co_sim.logger.images_path)

    # plot_for_path("recordings/freq_tests/path_v0", "dist_5.png", "time_5.png", "v0", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v0", "dist_30Hz.png", "time_30Hz.png", "v0", "infer_freq_Hz", 30)

    # plot_for_path("recordings/freq_tests/path_v0_lessShadows", "dist_5.png", "time_5.png", "v0", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v0_lessShadows", "dist_30Hz.png", "time_30Hz.png", "v0", "infer_freq_Hz", 30)

    # plot_for_path("recordings/freq_tests/path_v1", "dist_5.png", "time_5.png", "v1", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v1", "dist_30Hz.png", "time_30Hz.png", "v1", "infer_freq_Hz", 30)

    # plot_for_path("recordings/freq_tests/path_v2", "dist_5.png", "time_5.png", "v2", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v2", "dist_30Hz.png", "time_30Hz.png", "v2", "infer_freq_Hz", 65)
