from typing import get_args
import os
import traceback
import shutil

from models.CoSimulator import CoSimulator
from config import DefaultCoSimulatorConfig
from pruning import load_pruning_report, get_pruning_pareto
from utils.recordings.plots import plots_for_path, plot_success_rate
from project_types import Path_version_t

NUM_RUNS = 5
VELOCITY = 5
MIN_FREQ = 5
MAX_FREQ = 35

if __name__ == '__main__':
    pareto_parent_folder = "nets/checkpoints/pareto"
    pruned_parent_folder = "nets/checkpoints/pruning/ssd_pretrained/finetuning/"
    stats = load_pruning_report(os.path.join(pruned_parent_folder, "report.json"))
    pareto_stats = get_pruning_pareto(stats)
    for stat in pareto_stats:
        pareto_folder = os.path.join(pareto_parent_folder, stat["model_id"])
        pareto_model_path = os.path.join(pareto_folder, "model.pth")
        pareto_checkpoint_path = os.path.join(pareto_folder, "checkpoint.pth")
        logging_model_folder = os.path.join("recordings/", stat["model_id"], "freq_tests")
        os.makedirs(logging_model_folder)

        for path in ["path_v0", "path_v1", "path_v2"]:
            logging_parent_folder = os.path.join(logging_model_folder,
                                                 path,
                                                 f"vel{VELOCITY}_freq{MIN_FREQ}_to_{MAX_FREQ}Hz_{NUM_RUNS}runs"
            )
            os.makedirs(logging_parent_folder)

            path_version = path.split("_")[1]
            if path_version not in get_args(Path_version_t):
                raise ValueError(f"{path_version} is not a valid Path_version_t!")

            for f in range(MIN_FREQ, MAX_FREQ+1, 1):
                run_cnt = 0
                while run_cnt < NUM_RUNS:
                    config = DefaultCoSimulatorConfig(sim_fps=f,
                                                        camera_fps=f,
                                                        infer_freq_Hz=f,
                                                        filter_freq_Hz=f,
                                                        uav_velocity=VELOCITY,
                                                        model_id=stat["model_id"],
                                                        model_path=pareto_model_path,
                                                        checkpoint_path=pareto_checkpoint_path)
                    co_sim = CoSimulator(config=config,
                                            log_folder=logging_parent_folder,
                                            movement="Path",
                                            path_version=path_version, # type: ignore
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

            plots_for_path(folder_path=logging_parent_folder,
                           dist_filename=f"dist_{VELOCITY}.png",
                           time_filename=f"time_{VELOCITY}.png",
                           constant_key="uav_velocity",
                           constant_value=VELOCITY,
                           mode="binary",
                           path_version=path_version, # type: ignore
                           nn_name=f"Pruned_SSD_{stat['model_id']}"
            )
            plot_success_rate(folder_path=logging_parent_folder,
                              out_filename=os.path.join(logging_parent_folder, "success_rate.png"),
                              path_version=path_version, # type: ignore
                              constant_key="uav_velocity",
                              constant_value=VELOCITY
            )
