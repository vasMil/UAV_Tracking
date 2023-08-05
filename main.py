import traceback
import shutil

import numpy as np

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator
from utils.recordings.plots import plot_for_path, plot_success_rate
from utils.path import plot_path

if __name__ == '__main__':
    plot_path(np.array([3.5, 0, -1.6]), "v0", "recordings/freq_tests/path_v0/path_v0.png")
    num_runs = 10
    for f in range(5, 36, 1):
        run_cnt = 0
        while run_cnt < num_runs:
            config = DefaultCoSimulatorConfig(sim_fps=f,
                                                camera_fps=f,
                                                infer_freq_Hz=f,
                                                filter_freq_Hz=f)
            co_sim = CoSimulator(config=config, log_folder="recordings/freq_tests/path_v0",
                                movement="Path",
                                path_version="v0",
                                display_terminal_progress=True,
                                keep_frames=False,
                                get_video=False)
            try:
                co_sim.start()
                while not co_sim.done and co_sim.status == 0:
                    co_sim.advance()
            except Exception:
                co_sim.finalize("Error")
                print("There was an error, writing setup file and releasing AirSim...")
                print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
                traceback.print_exc()
                shutil.rmtree(co_sim.logger.parent_folder)
                run_cnt -= 1
            finally:
                co_sim.finalize("Time's up")
                run_cnt += 1
    
    # plot_for_path("recordings/freq_tests/path_v0", "dist_5.png", "time_5.png", "v0", "uav_velocity", 5)
    # # plot_for_path("recordings/vel_tests/path_v0", f"dist_{best_freq}Hz.png", f"time_{best_freq}Hz.png", "v0", "infer_freq_Hz", best_freq)

    # # plot_for_path("recordings/freq_tests/path_v0_lessShadows", "dist_5.png", "time_5.png", "v0", "uav_velocity", 5)
    # # plot_for_path("recordings/vel_tests/path_v0_lessShadows", "dist_30Hz.png", "time_30Hz.png", "v0", "infer_freq_Hz", 30)

    # # plot_for_path("recordings/freq_tests/path_v1", "dist_5.png", "time_5.png", "v1", "uav_velocity", 5)
    # # plot_for_path("recordings/vel_tests/path_v1", "dist_30Hz.png", "time_30Hz.png", "v1", "infer_freq_Hz", 30)

    # # plot_for_path("recordings/freq_tests/path_v2", "dist_5.png", "time_5.png", "v2", "uav_velocity", 5)
    # # plot_for_path("recordings/vel_tests/path_v2", "dist_30Hz.png", "time_30Hz.png", "v2", "infer_freq_Hz", 65)

    # plot_success_rate("recordings/freq_tests/path_v0",
    #                   "recordings/freq_tests/path_v0/success_rate.png",
    #                   path_version="v0",
    #                   constant_key="uav_velocity",
    #                   constant_value=5)
