import traceback
import shutil

from tqdm import tqdm

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator
from utils.recordings.plots import plots_for_path, plot_success_rate

if __name__ == '__main__':
    num_runs = 5
    for velocity in tqdm([6, 7, 8], "Velocity"):
        for f in tqdm(range(5, 36, 1), "Frequency"):
            run_cnt = 0
            while run_cnt < num_runs:
                config = DefaultCoSimulatorConfig(sim_fps=f,
                                                  camera_fps=f,
                                                  infer_freq_Hz=f,
                                                  filter_freq_Hz=f,
                                                  uav_velocity=velocity)
                co_sim = CoSimulator(config=config, log_folder=f"recordings/freq_tests/path_v1/vel{velocity}_freq5_to_35Hz_5runs",
                                     movement="Path",
                                     path_version="v1",
                                     display_terminal_progress=False,
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
 
        plots_for_path(folder_path=f"recordings/freq_tests/path_v1/vel{velocity}_freq5_to_35Hz_5runs",
                    dist_filename=f"dist_{velocity}.png",
                    time_filename=f"time_{velocity}.png",
                    constant_key="uav_velocity",
                    constant_value=velocity,
                    mode="binary",
                    path_version="v1",
                    nn_name="SSD"
        )
        plot_success_rate(folder_path=f"recordings/freq_tests/path_v1/vel{velocity}_freq5_to_35Hz_5runs",
                        out_filename=f"recordings/freq_tests/path_v1/vel{velocity}_freq5_to_35Hz_5runs/success_rate.png",
                        path_version="v1",
                        constant_key="uav_velocity",
                        constant_value=velocity
        )
