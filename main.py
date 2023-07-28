import traceback
import shutil

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
    num_tests = 1
    best_freq = 30
    for v in range(5, 11):
        for i in range(num_tests):
            config = DefaultCoSimulatorConfig(sim_fps=best_freq,
                                              camera_fps=best_freq,
                                              infer_freq_Hz=best_freq,
                                              filter_freq_Hz=best_freq,
                                              uav_velocity=v)
            co_sim = CoSimulator(config=config, log_folder="recordings/vel_tests/path_v1/", movement="Path", path_version="v1")
            try:
                co_sim.start()
                while not co_sim.done and co_sim.status == 0:
                    co_sim.advance()
            except Exception:
                co_sim.finalize("Error")
                print("There was an error, writing setup file and releasing AirSim...")
                print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
                traceback.print_exc()
            finally:
                co_sim.finalize("Time's up")
                shutil.rmtree(co_sim.logger.images_path)
