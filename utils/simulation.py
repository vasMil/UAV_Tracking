from typing import List, Tuple
import math

import numpy as np

from models.UAV import UAV


def sim_calculate_angle(uav_source: UAV, uav_target: UAV) -> float:
    source_pos = np.expand_dims(
        uav_source.simGetObjectPose().position.to_numpy_array(),
        axis=1
    )
    # source_pos = vector_transformation(*(uav_source.getPitchRollYaw()), vec=source_pos, to=True)
    target_pos = np.expand_dims(
        uav_target.simGetObjectPose().position.to_numpy_array(),
        axis=1
    )
    # target_pos = vector_transformation(*(uav_source.getPitchRollYaw()), vec=target_pos, to=True)
    dist = target_pos - source_pos

    rad = 0. if dist[0] == 0 else math.atan(dist[1]/dist[0])
    deg = math.degrees(rad)
    if deg > 0 and dist[1] < 0:
        deg -= 180
    elif deg < 0 and dist[1] > 0:
        deg += 180
    return deg

def compare_runs(root_paths: List[str]):
    from models.FrameInfo import FrameInfo
    from project_types import ExtendedCoSimulatorConfig_t
    from utils.recordings.helpers import folder_to_info

    print("\nLoading data...")
    configs: List[ExtendedCoSimulatorConfig_t] = []
    infos: List[List[FrameInfo]] = []
    for root_path in root_paths:
            info, config, _ = folder_to_info(root_path)
            configs.append(config)
            infos.append(info)

    print("\nChecking for any configuration missmatch")
    for key in configs[0].keys():
        if key == "frame_count" and any(config[key] != configs[0][key] for config in configs[1:]):
            # raise Exception("frame_count missmatch")
            print(f"{key} missmatch")
            continue
        if any(config[key] != configs[0][key] for config in configs[1:]):
            print(f"{key} missmatch")
    
    min_len = min([len(info) for info in infos])
    for i, info in enumerate(infos):
        infos[i] = info[:min_len]
    
    print("\nSearching for any missmatch in the information collected for each run")
    for key in ["sim_lead_pos", "sim_ego_pos", "sim_lead_vel", "sim_ego_vel", "sim_angle_deg", "bbox_score", "extra_timestamp"]:
        if key == "extra_timestamp":
            intervals = []
            for info_list in infos:
                timestamps = [(frame_info[key] 
                               - info_list[0][key]
                               ) for frame_info in info_list]

                intervals.append(timestamps)
            for interval_list in intervals[1:]:
                for i, interval in enumerate(interval_list):
                    if interval != intervals[0][i]:
                        print(f"Interval {i} missmatch")
                        break
            continue

        for info_list in infos:
            for i, info in enumerate(info_list):
                if info[key] != infos[0][i][key]:
                    print(f"{key} {i} missmatch")
                    break

    print("\nMSEs for the first two recordings:")
    infos0 = infos[0]
    infos1 = infos[1]
    for key in ["sim_lead_pos", "sim_lead_vel", "sim_ego_pos", "sim_lead_pos", "est_lead_pos"]:
        mse = 0
        for i, f_info in enumerate(infos0):
            mse += np.linalg.norm(np.array(f_info[key]) - np.array(infos1[i][key]))**2
        mse /= len(infos0)
        print(f"mse({key}) = {mse}")

def get_noisy_measurements(path_to_prev_exec: str,
                           target_snr: float,
                           meas_key: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
    - path_to_prev_exec: The root folder, where a previous excecution
    of the same leadingUAV path we want to add noise to is.
    - target_snr: The desirned Singal to Noise Ratio.

    Returns:
    - The signal without any noise.
    - The signal with AWGN.
    - The times (in seconds), at which each point was sampled (Starting from 0).
    - The achieved SNR.
    """
    from utils.recordings.helpers import folder_to_info

    info, _, _ = folder_to_info(path_to_prev_exec)
    sig = np.array([f_info[meas_key] for f_info in info]).T.astype(np.float64)
    sig_pow = (np.abs(sig)**2).sum(axis=1) / len(sig)
    # Add white Gaussian Noise to your signal
    target_snr = 100
    mean = 0
    std = np.sqrt(sig_pow/target_snr)
    noise = np.zeros(sig.shape)
    for i in range(sig.shape[0]):
        noise[i,:] += np.random.normal(mean, std[i], size=sig.shape[1])
    noisy_sig = sig + noise

    snr = np.divide(sig_pow, (np.abs(noise)**2).sum(axis=1)/noise.shape[1])
    times = np.array([(f_info["extra_timestamp"] - info[0]["extra_timestamp"])*1e9 for f_info in info])
    
    return (sig, noisy_sig, times, snr,)

def plot_clean_and_noisy_signals(sig: np.ndarray,
                                 noisy_sig: np.ndarray,
                                 times: np.ndarray,
                                 outpath: str,
                                 ylabel: str
    ):
    import matplotlib.pyplot as plt

    fig, ((ax0,ax1,ax2), (ax3,ax4,ax5)) = plt.subplots(2,3, figsize=(20,10))

    for i, axes in enumerate([(ax0,ax3), (ax1,ax4), (ax2,ax5)]):
        ax_up: plt.Axes = axes[0]
        ax_down: plt.Axes = axes[1]
        ax_up.set_title(f"Dimension {i} - Column")
        ax_up.plot(times, sig[i,:], label="Ground Truth")
        ax_down.plot(times, noisy_sig[i,:], label="Noisy")
        ax_up.set_xlabel("Time (s)")
        ax_down.set_xlabel("Time (s)")
        ax_up.set_ylabel(ylabel)
        ax_down.set_ylabel(ylabel)
    fig.savefig(outpath)
    plt.close(fig)
