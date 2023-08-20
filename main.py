import traceback
import shutil

from config import DefaultCoSimulatorConfig
from models.CoSimulator import CoSimulator

if __name__ == '__main__':
    f = 30
    config = DefaultCoSimulatorConfig(sim_fps=f,
                                      camera_fps=f,
                                      infer_freq_Hz=f,
                                      filter_freq_Hz=f)
    co_sim = CoSimulator(config=config,
                         log_folder="recordings/",
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
    finally:
        co_sim.finalize("Time's up")
