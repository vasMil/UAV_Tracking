from typing import List, Tuple
import math

import numpy as np
import airsim

from project_types import Path_version_t

def smooth_path(path: List[np.ndarray], kernel_size: Tuple[int, int, int] = (1,3,1)) -> List[np.ndarray]:
    res = np.zeros([3, len(path)])
    for dim in range(3):
        stacked_dim = np.array([])
        for point in path:
            stacked_dim = np.hstack([stacked_dim, point[dim]])
        kernel = np.ones([kernel_size[dim]]) * (1/kernel_size[dim])
        for _ in range(len(path) % kernel_size[dim]):
            stacked_dim = np.hstack([stacked_dim, stacked_dim[-1]])
        for i in range(int(len(path) - kernel_size[dim] + 1)):
            stacked_dim[i] = np.multiply(stacked_dim[i:i+kernel_size[dim]], kernel).sum()
        res[dim] = stacked_dim[:len(path)]
    return [i.squeeze() for i in np.hsplit(res, len(path))]

def accumulate_points(path: List[np.ndarray]) -> List[np.ndarray]:
    """
    Given a list of points. Sum each point with the previous, in order to get
    the actual points on the 3D coordinate system.
    """
    res = []
    for i, _ in enumerate(path):
        if i == 0:
            res.append(path[i])
            continue
        res.append(res[-1] + path[i])

    return res

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

def create_linear_samples(points: List[np.ndarray], samples: int) -> List[np.ndarray]:
    res = []
    for point in points:
        step = point/samples
        for _ in range(samples):
            res.append(step)
    return res

def create_np_path(start_pos: np.ndarray,
                   version: Path_version_t = "v2"
    ) -> List[np.ndarray]:
    path_v0 = [start_pos]
    path_v0 += create_linear_samples([
        np.array([10., 0.,   0.]),
        np.array([2.,  10.,  0.]),
        np.array([20., 0.,   0.]),
        np.array([2.,  -10., 0.]),
        np.array([20., 0.,   0.]),
        np.array([0.,  10.,  0.]),
        np.array([20., 0.,   0.]),
        np.array([0.,  -10., 0.]),
        np.array([20., 0.,   0.]),
    ], 100)
    path_v0 += get_points_on_spiral(radius=4, height_limit=5, num_points=100, rotational_velocity_z=4*math.pi)
    path_v0 += create_linear_samples([np.array([20., 0., 0.])], 100)

    path_v1 = [start_pos]
    path_v1 += create_linear_samples([np.array([20., 0., 0.])], 100)
    path_v1 += get_points_on_sinusoid(rotational_velocity_y=2*math.pi,
                                 rotational_velocity_z=0)
    path_v1 += get_points_on_sinusoid(rotational_velocity_y=4*math.pi,
                                 rotational_velocity_z=0)
    path_v1 += get_points_on_sinusoid(rotational_velocity_y=6*math.pi,
                                 rotational_velocity_z=0)

    path_v2 =  [start_pos] 
    path_v2 += create_linear_samples([np.array([20., 0., 0.])], 100)
    path_v2 += get_points_on_sinusoid(rotational_velocity_y=4*math.pi, rotational_velocity_z=2*math.pi)
    path_v2 += get_points_on_sinusoid(rotational_velocity_y=4*math.pi, rotational_velocity_z=4*math.pi)
    path_v2 += get_points_on_sinusoid(rotational_velocity_y=2*math.pi, rotational_velocity_z=4*math.pi)
    path_v2 += create_linear_samples([np.array([20., 0., 0.])], 100)

    str8line = [start_pos] + create_linear_samples([np.array([20., 0., 0.])], 100)
    rightline = [start_pos] + create_linear_samples([np.array([0., 20., 0.])], 100)
    upline = [start_pos] + create_linear_samples([np.array([0., 0., -20.])], 100)
    return accumulate_points(path_v0 if version == "v0" else\
                             path_v1 if version == "v1" else\
                             path_v2 if version == "v2" else\
                             str8line if version == "str8line" else
                             rightline if version == "rightline" else
                             upline)

def get_path(start_pos: airsim.Vector3r,
             version: Path_version_t = "v2"
    ) -> List[airsim.Vector3r]:
    """
    Get a predefined path, given the starting position of the vehicle.
    """
    path = create_np_path(start_pos=start_pos.to_numpy_array().squeeze(),
                          version=version)
    return [airsim.Vector3r(x[0].item(), x[1].item(), x[2].item()) for x in path]

def plot_path(start_pos: np.ndarray,
              version: Path_version_t,
              filename: str
    ) -> None:
    import matplotlib.pyplot as plt

    path = create_np_path(start_pos, version=version)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.invert_yaxis()
    ax.invert_zaxis()
    x = np.array([point[0] for point in path])
    y = np.array([point[1] for point in path])
    z = np.array([point[2] for point in path])
        
    ax.plot3D(xs=x, ys=y, zs=z)
    ax.scatter3D(xs=x, ys=y, zs=z) # type: ignore
    
    ax.scatter3D(x[0], y[0], z[0], color="green")
    ax.text(x[0], y[0], z[0], 'Start', horizontalalignment='right', color="green")
    ax.scatter3D(x[-1], y[-1], z[-1], color="red")
    ax.text(x[-1], y[-1], z[-1], 'Finish', horizontalalignment='left', color="red")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-120, roll=0)
    fig.savefig(filename)
    plt.close(fig)
