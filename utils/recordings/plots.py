from typing import List, Any, get_args, Literal, Union, Callable
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.recordings.helpers import get_folders_in_path, extract_recording_data
from constants import STATUS_COLORS, BINARY_STATUS_COLORS
from project_types import Status_t, Binary_status_t, Path_version_t, map_status_to_color, map_binary_status_to_color

def _plots_for_path(fig: Any,
                    ax: plt.Axes,
                    x: np.ndarray,
                    y: np.ndarray,
                    statuses: Union[List[Union[Status_t, int]], List[Union[Binary_status_t, bool]]]
    ):
    if x.shape != y.shape or max(x.shape) != len(statuses):
        raise Exception("x, y and statuses must be of the same size!")
    if len(statuses) == 0:
        raise Exception("The arrays (or List) should have at least one element, else there is nothing to plot!")
    status_t = type(statuses[0])
    mapper_func = None
    COLORS = None
    color_labels = None
    if status_t == Status_t or status_t == int:
        mapper_func = map_status_to_color
        COLORS = STATUS_COLORS
        color_labels = get_args(Status_t)
    elif status_t == Binary_status_t or status_t == bool:
        mapper_func = map_binary_status_to_color
        COLORS = BINARY_STATUS_COLORS
        color_labels = get_args(Binary_status_t)
    else:
        raise Exception(f"Type {status_t} is invalid as status type!")
    color_list = [mapper_func(status) for status in statuses] # type: ignore
    # Calculate the average y values (between runs)
    unq_x = np.unique(x)
    min_y = np.zeros(len(unq_x))
    mean_y = np.zeros(len(unq_x))
    max_y = np.zeros(len(unq_x))
    for i, freq in enumerate(unq_x):
        mask = (x == freq)
        unq_y = y[mask]
        min_y[i] = unq_y.min()
        mean_y[i] = unq_y.mean()
        max_y[i] = unq_y.max()
    ax.plot(unq_x, mean_y, linewidth=1, color="black")
    # np.maximum is used to "fix" any arithmetic errors that occure
    # when subtracting min, max, and mean arrays. Sometimes these
    # subtractions result to values like: -7.10542736e-15
    ax.errorbar(unq_x,
                mean_y,
                yerr=np.stack([np.maximum(mean_y - min_y, 0), np.maximum(max_y - mean_y, 0)]),
                linewidth=1, capsize=0, color="black"
    )
    # Scatter with colorbar depicting the status
    ax.scatter(x=x, y=y, c=color_list) # type: ignore
    cmap = ListedColormap(COLORS) # type: ignore
    cbar = fig.colorbar(mappable=None, cmap=cmap)
    cbar.set_ticks(((np.arange(len(COLORS)) + 0.5)/len(COLORS)).tolist())
    cbar.set_ticklabels([ustat for ustat in color_labels]) # type: ignore

def plots_for_path(folder_path: str,
                   dist_filename: str,
                   time_filename: str,
                   constant_key: Literal["uav_velocity", "infer_freq_Hz"],
                   constant_value: Union[int, float],
                   mode: Literal["all", "binary"],
                   path_version: Path_version_t,
                   nn_name: str
    ) -> None:
    folders = get_folders_in_path(folder_path)
    n = len(folders)
    x = np.zeros(n)
    dists = np.zeros(n)
    times = np.zeros(n)
    if constant_key == "uav_velocity":
        x_label = "Inference Frequency (Hz)"
    else:
        x_label = "UAV Velocity (m/s)"

    x, statuses, dists, times = extract_recording_data(recordings_path=folder_path,
                                                       constant_key=constant_key,
                                                       constant_value=constant_value,
                                                       mode=mode
                                )

    # Plot ground truth distances
    fig, ax = plt.subplots()
    fig.set_figwidth(14)
    _plots_for_path(fig=fig, ax=ax, x=x, y=dists, statuses=statuses.tolist())
    ax.set_title(f"{nn_name} - Path {path_version}")
    ax.set_xlabel(f"{x_label}")
    ax.set_ylabel("Average True Distance between UAVs (m)")
    fig.savefig(os.path.join(folder_path, dist_filename))
    plt.close(fig)

    # Plot simulation time
    fig, ax = plt.subplots()
    fig.set_figwidth(14)
    _plots_for_path(fig=fig, ax=ax, x=x, y=times, statuses=statuses.tolist())
    ax.set_title(f"{nn_name} - Path {path_version}")
    ax.set_xlabel(f"{x_label}")
    ax.set_ylabel("Simulation Time (s)")
    fig.savefig(os.path.join(folder_path, time_filename))
    plt.close(fig)

def plot_success_rate(folder_path: str,
                      out_filename: str,
                      path_version: Path_version_t,
                      constant_key: Literal["uav_velocity", "infer_freq_Hz"],
                      constant_value: Union[int, float]
    ):
    if constant_key == "uav_velocity":
        x_key = "infer_freq_Hz"
        x_label = "Inference Frequency (Hz)"
    else:
        x_key = "uav_velocity"
        x_label = "UAV Velocity (m/s)"
    unq_x, unq_succ_rate, _, _ = extract_recording_data(folder_path, constant_key, constant_value, mode="mean")

    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(unq_x, unq_succ_rate, color="green", marker='o', linestyle="-")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Success rate")
    ax.set_title(f"SSD - Path {path_version}")
    fig.savefig(out_filename)
    plt.close(fig)
