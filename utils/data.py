from typing import List, Tuple, Any, get_args, Literal, Union
import os
import json

import pickle
import airsim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from constants import FILENAME_LEADING_ZEROS, STATUS_COLORS
from models.BoundingBox import BoundingBoxFactory
from models.FrameInfo import FrameInfo
from project_types import Status_t, Path_version_t, ExtendedCoSimulatorConfig_t,\
    map_status_to_color, map_to_binary_status, map_to_status_code, map_from_status_code

def get_folders_in_path(path: str) -> List[str]:
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(os.path.join(path, entry.name))
    return folders

def rename_images(root_path: str, json_file: str) -> None:
    root_path = "data/empty_map/train/"
    bboxes = BoundingBoxFactory(json_file).bboxes
    
    for bbox in bboxes:
        img_path = os.path.join(root_path, bbox.img_name) # type: ignore
        dest_path = os.path.join(root_path, "temp_" + bbox.img_name) # type: ignore
        os.rename(img_path, dest_path)

    for i, bbox in enumerate(bboxes):
        img_path = os.path.join(root_path, "temp_" + bbox.img_name) # type: ignore
        new_img_name = str(i).zfill(FILENAME_LEADING_ZEROS) + ".png"
        dest_path = os.path.join(root_path, new_img_name) # type: ignore
        os.rename(img_path, dest_path)
        bbox.img_name = new_img_name

    with open(json_file, 'w') as f:
        json.dump([bbox.__dict__() for bbox in bboxes], f)

def folder_to_info(path: str) -> Tuple[List[FrameInfo], ExtendedCoSimulatorConfig_t, Status_t]:
    pkl_file = os.path.join(path, "log.pkl")
    json_file = os.path.join(path, "config.json")
    with open(pkl_file, 'rb') as f:
        frames_info: List[FrameInfo] = pickle.load(file=f)

    with open(json_file, 'r') as f:
        extended_config: ExtendedCoSimulatorConfig_t = json.load(fp=f)
    
    status: Status_t = extended_config["status"]
    return (frames_info, extended_config, status)

def update_movement_plots(path: str, start_pos: airsim.Vector3r, path_version: Path_version_t):
    from models.logger import GraphLogs
    from utils.simulation import getTestPath
    folders = get_folders_in_path(path)
    for folder in folders:
        movement_file = os.path.join(folder, "movement.png")
        frames_info, _, _ = folder_to_info(folder)
        os.remove(movement_file)
        gl = GraphLogs(frames_info)
        gl.plot_movement_3d(movement_file, getTestPath(start_pos=start_pos, version=path_version))

def _plot_for_path(fig: Any, ax: plt.Axes, x: np.ndarray, y: np.ndarray, c: List[str]):
    # Calculate the average y values (between runs)
    unique_x = np.unique(x)
    min_y = np.zeros(len(unique_x))
    mean_y = np.zeros(len(unique_x))
    max_y = np.zeros(len(unique_x))
    for i, freq in enumerate(unique_x):
        mask = (x == freq)
        dists_for_freq = y[mask]
        min_y[i] = dists_for_freq.min()
        mean_y[i] = dists_for_freq.mean()
        max_y[i] = dists_for_freq.max()

    ax.plot(unique_x, mean_y, linewidth=1, color="blue")
    ax.errorbar(unique_x, mean_y, yerr=np.stack([mean_y - min_y, max_y - mean_y]), linewidth=1, capsize=0, color="blue")
    # Scatter with colorbar depicting the status
    ax.scatter(x=x, y=y, c=c)
    cmap = ListedColormap(STATUS_COLORS) # type: ignore
    cbar = fig.colorbar(mappable=None, cmap=cmap)
    cbar.set_ticks(((np.arange(len(STATUS_COLORS)) + 0.5)/len(STATUS_COLORS)).tolist())
    cbar.set_ticklabels([ustat for ustat in get_args(Status_t)]) # type: ignore

    # Other options
    ax.set_xlabel("SSD - Inference Frequency (Hz)")
    ax.set_ylabel("Average True Distance (m)")

def plot_for_path(folder_path: str,
                  dist_filename: str,
                  time_filename: str,
                  path_version: Path_version_t,
                  constant_key: Literal["uav_velocity", "infer_freq_Hz"],
                  constant_value: Union[int, float]
    ) -> None:
    folders = get_folders_in_path(folder_path)
    n = len(folders)
    x = np.zeros(n)
    dists = np.zeros(n)
    times = np.zeros(n)
    status_colors: List[str] = []
    if constant_key == "uav_velocity":
        x_key = "infer_freq_Hz"
        x_label = "Inference Frequency (Hz)"
    else:
        x_key = "uav_velocity"
        x_label = "UAV Velocity (m/s)"

    # Load data from runs one by one and update your statistics
    i = 0
    for folder in folders:
        infos, config, status = folder_to_info(folder)
        if config[constant_key] != constant_value:
            continue
        x[i] = config[x_key]
        times[i] = config["frame_count"]/config["camera_fps"]
        dists[i] = config["avg_true_dist"]
        status_colors.append(map_status_to_color(status))
        i += 1
    x = x[0:i]
    times = times[0:i]
    dists = dists[0:i]

    # Plot ground truth distances
    fig, ax = plt.subplots()
    fig.set_figwidth(14)
    _plot_for_path(fig=fig, ax=ax, x=x, y=dists, c=status_colors)
    ax.set_title(f"Path {path_version}")
    ax.set_xlabel(f"SSD - {x_label}")
    ax.set_ylabel("Average Ground Truth Distance (m)")
    fig.savefig(os.path.join(folder_path, dist_filename))
    plt.close(fig)

    # Plot simulation time
    fig, ax = plt.subplots()
    fig.set_figwidth(14)
    _plot_for_path(fig=fig, ax=ax, x=x, y=times, c=status_colors)
    ax.set_title(f"Path {path_version}")
    ax.set_xlabel(f"SSD - {x_label}")
    ax.set_ylabel("Simulation Time (s)")
    fig.savefig(os.path.join(folder_path, time_filename))
    plt.close(fig)

def plot_success_rate(folder_path: str,
                      out_filename: str,
                      path_version: Path_version_t,
                      constant_key: Literal["uav_velocity", "infer_freq_Hz"],
                      constant_value: Union[int, float]
    ):
    folders = get_folders_in_path(folder_path)
    n = len(folders)
    x = np.zeros(n)
    status_codes = np.zeros(n, dtype=np.int64)

    if constant_key == "uav_velocity":
        x_key = "infer_freq_Hz"
        x_label = "Inference Frequency (Hz)"
    else:
        x_key = "uav_velocity"
        x_label = "UAV Velocity (m/s)"

    # Load data from runs one by one and update your statistics
    i = 0
    for folder in folders:
        _, config, status = folder_to_info(folder)
        if config[constant_key] != constant_value:
            continue
        x[i] = config[x_key]
        status_codes[i] = map_to_status_code(status)
        i += 1
    x = x[0:i]

    unq_x = np.unique(x)
    unq_x_succ_rate = np.zeros(len(unq_x))
    for i, x_val in enumerate(unq_x):
        mask = (x == x_val)
        x_val_status_codes = status_codes[mask]
        for x_val_status_code in x_val_status_codes:
            if map_to_binary_status(
                map_from_status_code(
                    x_val_status_code
                )) == "Success":
                unq_x_succ_rate[i] += 1
        unq_x_succ_rate[i] /= len(x_val_status_codes)
    
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(unq_x, unq_x_succ_rate, color="green", marker='o', linestyle="-")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Success rate")
    ax.set_title(f"SSD - Path {path_version}")
    fig.savefig(out_filename)
    plt.close(fig)      
