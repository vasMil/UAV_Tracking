from typing import List, Union
import math

import numpy as np
import airsim

from project_types import Path_version_t

def accumulatePoints(path: List[np.ndarray]) -> List[np.ndarray]:
    """
    Given a list of points. Sum each point with the previous, in order to get
    the actual points on the 3D coordinate system.
    """
    for i, _ in enumerate(path):
        if i == 0: continue
        path[i] += path[i-1]

    return path

def get_points_on_spiral(radius: float,
                         height_limit: float,
                         num_points: int,
                         rotational_velocity_z: float
    ) -> List[np.ndarray]:
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
        path.append(np.array([x, y, -z]))

    for i in reversed(range(num_points)):
        if i == 0: continue
        path[i] -= path[i-1]

    return path

def get_points_on_sinusoid(num_points: int = 1000,
                           x_length: float = 50.,
                           y_amplitude: float = 10.,
                           z_amplitude: float = 10.,
                           rotational_velocity_y: float = 2*math.pi,
                           rotational_velocity_z: float = 2*math.pi,
                           init_phase_y: float = 0.,
                           init_phase_z: float = 0.
    ) -> List[np.ndarray]:
    path = []
    for i in range(num_points):
        angle_y = rotational_velocity_y * i / num_points
        angle_z = rotational_velocity_z * i / num_points
        x = x_length * i / num_points
        y = y_amplitude * math.sin(angle_y + init_phase_y)
        z = z_amplitude * math.sin(angle_z + init_phase_z)
        path.append(np.array([x, y, -z]))

    for i in reversed(range(num_points)):
        if i == 0: continue
        path[i] -= path[i-1]

    return path

def create_np_path(start_pos: np.ndarray,
                   version: Path_version_t = "v2"
    ) -> List[np.ndarray]:
    path_v0 = [
        start_pos,
        np.array([10, 0,   0]),    # Test x axis - moving forward
        np.array([2,  10,  0]),    # Test y axis - moving fast right
        np.array([20, 0,   0]),    # Move forward so you will not crush on the EgoUAV
        np.array([2,  -10, 0]),    # Test y axis - moving fast left
        np.array([20, 0,   0]),    # Move forward so you will not crush on the EgoUAV
        np.array([0,  10,  0]),    # Test y axis - moving faster right
        np.array([20, 0,   0]),    # Move forward so you will not crush on the EgoUAV
        np.array([0,  -10, 0]),    # Test y axis - moving faster left
        np.array([20, 0,   0]),    # Move forward so you will not crush on the EgoUAV
    ]
    path_v0 += get_points_on_spiral(radius=4, height_limit=5, num_points=100, rotational_velocity_z=4*math.pi)
    path_v0 += [np.array([20, 0, 0])]

    path_v1 = [start_pos, np.array([20, 0, 0])]
    path_v1 += get_points_on_sinusoid(rotational_velocity_y=2*math.pi,
                                 rotational_velocity_z=0)
    path_v1 += get_points_on_sinusoid(rotational_velocity_y=4*math.pi,
                                 rotational_velocity_z=0)
    path_v1 += get_points_on_sinusoid(rotational_velocity_y=6*math.pi,
                                 rotational_velocity_z=0)

    path_v2 = [start_pos, np.array([20, 0, 0])]
    path_v2 += get_points_on_sinusoid(rotational_velocity_y=1*math.pi, rotational_velocity_z=1*math.pi)
    path_v2 += [np.array([20, 0, 0])]
    path_v2 += get_points_on_sinusoid(rotational_velocity_y=2*math.pi, rotational_velocity_z=2*math.pi)
    path_v2 += [np.array([20, 0, 0])]
    path_v2 += get_points_on_sinusoid(rotational_velocity_y=3*math.pi, rotational_velocity_z=3*math.pi)
    path_v2 += [np.array([20, 0, 0])]

    return accumulatePoints(path_v0 if version == "v0" else\
                            path_v1 if version == "v1" else\
                            path_v2)

def get_path(start_pos: airsim.Vector3r,
             version: Path_version_t = "v2"
    ) -> List[airsim.Vector3r]:
    """
    Get a predefined path, given the starting position of the vehicle.
    """
    path = create_np_path(start_pos=start_pos.to_numpy_array().squeeze(),
                          version=version)
    return [airsim.Vector3r(*x) for x in path]

def plot_path(start_pos: Union[np.ndarray, airsim.Vector3r],
              version: Path_version_t
    ) -> None:
    import matplotlib.pyplot as plt

    if isinstance(start_pos, airsim.Vector3r):
        start_pos = start_pos.to_numpy_array().squeeze()
    path = create_np_path(start_pos, version=version)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.array([point[0] for point in path])
    y = np.array([point[1] for point in path])
    z = np.array([point[2] for point in path])
        
    ax.plot(xs=x, ys=y, zs=z)

    ax_list = [x, y, z]
    for i, ticks_func in enumerate(["set_xlim3d", "set_ylim3d", "set_zlim3d"]):
        vals = ax_list[i]
        max, min = vals.max(), vals.min()
        if max - min < 5:
            avg = (max + min)/2
            half_range = 10 / 2
            getattr(ax, ticks_func)(avg-half_range, avg+half_range)

    ax.scatter(x[0], y[0], z[0], color="green")
    ax.text(x[0], y[0], z[0], 'Start', horizontalalignment='right', color="green")
    ax.scatter(x[-1], y[-1], z[-1], color="red")
    ax.text(x[-1], y[-1], z[-1], 'Finish', horizontalalignment='left', color="red")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-120, roll=0)
    ax.invert_yaxis()
    ax.invert_zaxis()
    plt.show()
