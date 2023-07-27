import traceback
import shutil

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator
from utils.data import plot_for_path

if __name__ == '__main__':
    plot_for_path("recordings/path_v0", "dist.png", "time.png", "v0")
    plot_for_path("recordings/path_v1", "dist.png", "time.png", "v1")
    plot_for_path("recordings/path_v2", "dist.png", "time.png", "v2")
    # num_tests = 3
    # for fi, f in enumerate(range(1,31)):
    #     for i in range(num_tests):
    #         config = DefaultCoSimulatorConfig(sim_fps=f,
    #                                           camera_fps=f,
    #                                           infer_freq_Hz=f,
    #                                           filter_freq_Hz=f)
    #         co_sim = CoSimulator(config=config)
    #         try:
    #             co_sim.start()
    #             while not co_sim.done and co_sim.status == 0:
    #                 co_sim.advance()
    #         except Exception:
    #             co_sim.finalize("Error")
    #             print("There was an error, writing setup file and releasing AirSim...")
    #             print("\n" + "*"*10 + " THE ERROR MESSAGE " + "*"*10)
    #             traceback.print_exc()
    #         finally:
    #             co_sim.finalize("Time's up")
    #             shutil.rmtree(co_sim.logger.images_path)
