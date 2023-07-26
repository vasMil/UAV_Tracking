import traceback
import shutil

import numpy as np
import matplotlib.pyplot as plt

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator

if __name__ == '__main__':
    num_tests = 3
    x = np.arange(1,31)
    y = np.zeros(x.shape)
    t = np.zeros(x.shape)
    for fi, f in enumerate(range(1,31)):
        for i in range(num_tests):
            config = DefaultCoSimulatorConfig(sim_fps=f,
                                              camera_fps=f,
                                              infer_freq_Hz=f,
                                              filter_freq_Hz=f)
            co_sim = CoSimulator(config=config)
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
            
            y[fi] += co_sim.logger.get_statistics()["avg_true_dist"]
            t[fi] += co_sim.logger.frame_cnt/co_sim.config.camera_fps
            shutil.rmtree(co_sim.logger.images_path)
        y[fi] /= num_tests
        t[fi] /= num_tests
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("SSD - Inference Frequency (Hz)")
    ax.set_ylabel("Average True Distance (m)")
    fig.savefig("recordings/path_v0/path_v0_dist_for_freq.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(x, t)
    ax.set_xlabel("SSD - Inference Frequency (Hz)")
    ax.set_ylabel("Simulation Time (s)")
    fig.savefig("recordings/path_v0/path_v0_simtime_for_freq.png")
    plt.close(fig)
