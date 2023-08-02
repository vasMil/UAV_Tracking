from typing import List, Tuple
import math

import numpy as np
import airsim

from models.UAV import UAV
from project_types import Path_version_t


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

def createPathFromPoints(path: List[airsim.Vector3r]) -> List[airsim.Vector3r]:
    """
    Given a list of points. Sum each point with the previous, in order to get
    the actual points on the coordinate system and allow for a smooth movement
    on the resulted path.
    """
    for i, _ in enumerate(path):
        if i == 0: continue
        path[i] += path[i-1]

    return path

def getSpiralPath(radius: float,
                  height_limit: float,
                  num_points: int,
                  rotational_velocity_z: float
    ) -> List[airsim.Vector3r]:
    path = []

    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        # Calculate the rotational movement around the z-axis
        z_angle = rotational_velocity_z * i / num_points
        rotation_matrix = np.array([[math.cos(z_angle), -math.sin(z_angle)], 
                                    [math.sin(z_angle), math.cos(z_angle)]])
        x, y = np.dot(rotation_matrix, [x, y])

        z = height_limit * i / num_points
        path.append(airsim.Vector3r(x, y, -z))

    for i in reversed(range(num_points)):
        if i == 0: continue
        path[i] -= path[i-1]

    return path

def getSinusoidalPath(num_points: int = 1000,
                      x_length: float = 50.,
                      y_amplitude: float = 10.,
                      z_amplitude: float = 10.,
                      rotational_velocity_y: float = 2*math.pi,
                      rotational_velocity_z: float = 2*math.pi
    ) -> List[airsim.Vector3r]:
    path = []
    for i in range(num_points):
        angle_y = rotational_velocity_y * i / num_points
        angle_z = rotational_velocity_z * i / num_points
        x = x_length * i / num_points
        y = y_amplitude * math.cos(angle_y)
        z = z_amplitude * math.sin(angle_z)
        path.append(airsim.Vector3r(x, y, -z))

    for i in reversed(range(num_points)):
        if i == 0: continue
        path[i] -= path[i-1]

    return path

def getTestPath(start_pos: airsim.Vector3r, version: Path_version_t = "v2") -> List[airsim.Vector3r]:
    """
    Get a predefined path, given the starting position of the vehicle.
    """
    path_v0 = [
        start_pos,
        airsim.Vector3r(10, 0, 0),    # Test x axis - moving forward
        airsim.Vector3r(2, 10, 0),    # Test y axis - moving fast right
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(2, -10, 0),   # Test y axis - moving fast left
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, 10, 0),    # Test y axis - moving faster right
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
        airsim.Vector3r(0, -10, 0),   # Test y axis - moving faster left
        airsim.Vector3r(20, 0, 0),    # Move forward so you will not crush on the EgoUAV
    ]
    path_v0 += getSpiralPath(radius=4, height_limit=5, num_points=100, rotational_velocity_z=4*math.pi)

    path_v1 = [start_pos, airsim.Vector3r(20, 0, 0)]
    path_v1 += getSinusoidalPath(rotational_velocity_y=1*math.pi, rotational_velocity_z=0)
    path_v1 += [airsim.Vector3r(20, 0, 0)]
    path_v1 += getSinusoidalPath(rotational_velocity_y=2*math.pi, rotational_velocity_z=0)
    path_v1 += [airsim.Vector3r(20, 0, 0)]
    path_v1 += getSinusoidalPath(rotational_velocity_y=3*math.pi, rotational_velocity_z=0)
    path_v1 += [airsim.Vector3r(20, 0, 0)]

    path_v2 = [start_pos, airsim.Vector3r(20, 0, 0)]
    path_v2 += getSinusoidalPath(rotational_velocity_y=1*math.pi, rotational_velocity_z=1*math.pi)
    path_v2 += [airsim.Vector3r(20, 0, 0)]
    path_v2 += getSinusoidalPath(rotational_velocity_y=2*math.pi, rotational_velocity_z=2*math.pi)
    path_v2 += [airsim.Vector3r(20, 0, 0)]
    path_v2 += getSinusoidalPath(rotational_velocity_y=3*math.pi, rotational_velocity_z=3*math.pi)
    path_v2 += [airsim.Vector3r(20, 0, 0)]

    str8line = [start_pos, airsim.Vector3r(500, 0, 0)]

    return createPathFromPoints(path_v0 if version == "v0" else\
                                path_v1 if version == "v1" else\
                                path_v2 if version == "v2" else
                                str8line
            )


def compare_runs(root_paths: List[str]):
    from models.FrameInfo import FrameInfo
    from project_types import ExtendedCoSimulatorConfig_t
    from utils.data import load_run_data

    print("\nLoading data...")
    configs: List[ExtendedCoSimulatorConfig_t] = []
    infos: List[List[FrameInfo]] = []
    for root_path in root_paths:
            config, info = load_run_data(root_path)
            configs.append(config)
            infos.append(info)

    print("\nChecking for any configuration missmatch")
    for key in configs[0].keys():
        if key == "frame_count" and any(config[key] != configs[0][key] for config in configs[1:]):
            raise Exception("frame_count missmatch")    
        if any(config[key] != configs[0][key] for config in configs[1:]):
            print(f"{key} missmatch")
    
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
    from utils.data import load_run_data

    _, info = load_run_data(path_to_prev_exec)
    sig = np.array([f_info[meas_key] for f_info in info]).T
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
