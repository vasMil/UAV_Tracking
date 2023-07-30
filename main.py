import traceback
import shutil
import time

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator
from utils.data import plot_for_path

if __name__ == '__main__':
    # plot_for_path("recordings/freq_tests/path_v0", "dist_5.png", "time_5.png", "v0", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v0", "dist_30Hz.png", "time_30Hz.png", "v0", "infer_freq_Hz", 30)

    # plot_for_path("recordings/freq_tests/path_v0_lessShadows", "dist_5.png", "time_5.png", "v0", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v0_lessShadows", "dist_30Hz.png", "time_30Hz.png", "v0", "infer_freq_Hz", 30)

    # plot_for_path("recordings/freq_tests/path_v1", "dist_5.png", "time_5.png", "v1", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v1", "dist_30Hz.png", "time_30Hz.png", "v1", "infer_freq_Hz", 30)

    # plot_for_path("recordings/freq_tests/path_v2", "dist_5.png", "time_5.png", "v2", "uav_velocity", 5)
    # plot_for_path("recordings/vel_tests/path_v2", "dist_30Hz.png", "time_30Hz.png", "v2", "infer_freq_Hz", 65)
    # num_tests = 1
    # best_freq = 30
    # for v in range(5, 11):
    #     for i in range(num_tests):
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
        # shutil.rmtree(co_sim.logger.images_path)
    # from nets.DetectionNets import Detection_SSD
    # ssd = Detection_SSD(root_test_dir="data/empty_map/test/", json_test_labels="data/empty_map/test/empty_map.json")
    # ssd.get_inference_frequency(100,10)
    from gendata import generate_frames_and_bboxes, clean_generated_frames_and_bboxes, generate_data_using_segmentation
    # generate_frames_and_bboxes(10000, frames_root_path="data/empty_map_new/train/", json_path="data/empty_map_new/train/bboxes.json")
    # from nets.DetectionNets import Detection_FasterRCNN
    # rcnn = Detection_FasterRCNN()
    # clean_generated_frames_and_bboxes(frames_root_path="data/empty_map_new/train/",
    #                                   json_path="data/empty_map_new/train/bboxes.json",
    #                                   net=rcnn,
    #                                   last_checked_image=4203)
    generate_data_using_segmentation(10000, frames_root_path="data/empty_map_new/train/", json_path="data/empty_map_new/train/bboxes.json")

    # import os
    # path = "data/empty_map_new/train"
    # # List all files and directories in the specified directory
    # files_in_directory = os.listdir(path)
    # clean_filenames = []
    # # Iterate over each item in the directory
    # for item in files_in_directory:
    #     if os.path.isfile(os.path.join(path, item)) and item != "bboxes.json":
    #         clean_filenames.append(item)

    # i = 9999
    # clean_filenames.sort()
    # for filename in reversed(clean_filenames):
    #     if int(filename.split(".")[0]) != i:
    #         print(i, filename)
    #         break
    #     i -= 1