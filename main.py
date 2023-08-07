import traceback
import shutil

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator
from utils.recordings.plots import plots_for_path, plot_success_rate

if __name__ == '__main__':
    # num_runs = 10
    f = 30
    # for f in range(5, 36, 1):
        # run_cnt = 0
        # while run_cnt < num_runs:
    config = DefaultCoSimulatorConfig(sim_fps=f,
                                        camera_fps=f,
                                        infer_freq_Hz=f,
                                        filter_freq_Hz=f)
    co_sim = CoSimulator(config=config, log_folder="recordings/",
                        movement="Path",
                        path_version="v1",
                        display_terminal_progress=True,
                        keep_frames=False,
                        get_video=True)
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
        # run_cnt -= 1
    finally:
        co_sim.finalize("Time's up")
        # run_cnt += 1


    # plots_for_path(folder_path="recordings/freq_tests/path_v1",
    #                dist_filename="dist_5.png",
    #                time_filename="time_5.png",
    #                constant_key="uav_velocity",
    #                constant_value=5,
    #                mode="all",
    #                path_version="v1",
    #                nn_name="SSD"
    # )
    # plot_success_rate(folder_path="recordings/freq_tests/path_v1",
    #                   out_filename="recordings/freq_tests/path_v1/success_rate.png",
    #                   path_version="v1",
    #                   constant_key="uav_velocity",
    #                   constant_value=5
    # )
