import time

import numpy as np
import airsim
from tqdm import tqdm

from models.UAV import UAV
from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from nets.DetectionNetBench import DetectionNetBench
from gendata import create_sample
from GlobalConfig import GlobalConfig as config

def construct_measurement_vector(kinematics: airsim.KinematicsState) -> np.ndarray:
    pos = np.expand_dims(kinematics.position.to_numpy_array(), axis=1)
    vel = np.expand_dims(kinematics.linear_velocity.to_numpy_array(), axis=1)
    return np.vstack([pos, vel])

def complex_process_noise_estim(num_samples: int = 100) -> np.ndarray:
    uav = LeadingUAV(config.leadingUAV_name)
    uav.lastAction.join()

    # Allocate space for the 2D matrix that will have the
    # random variables as lines and each column will be an
    # observation
    observ_matrix = np.zeros([6, num_samples])

    prev_kin = uav.simGetGroundTruthKinematics()
    prev_meas_vec = construct_measurement_vector(prev_kin).squeeze()
    for obs_idx in tqdm(range(num_samples)):
        uav.random_move(config.leadingUAV_update_vel_interval_s)
        time.sleep(1/config.infer_freq_Hz)

        cur_kin = uav.simGetGroundTruthKinematics()
        cur_meas_vec = construct_measurement_vector(cur_kin).squeeze()
        
        observ_matrix[:, obs_idx] = cur_meas_vec - prev_meas_vec

        # Update
        prev_kin = cur_kin
        prev_meas_vec = cur_meas_vec


    # Use numpy to calculate the covariance matrix for all
    # random variables
    process_noise = np.cov(observ_matrix, bias=True)

    # Restore the initial simulation state
    uav.disable()
    uav.client.reset()
    return process_noise


def estimate_process_noise(num_samples: int = 100, set_wind: bool = False) -> np.ndarray:
    """
    Estimates the process covariance matrix, by setting the UAV to
    hover and measuring (ground truth, since we are in a simulation)
    it's position and velocity.
    """
    # Initialize the UAV to measure
    uav = UAV(config.leadingUAV_name)
    uav.lastAction.join()
    if set_wind:
        uav.client.simSetWind(airsim.Vector3r(1, -1, 0.2))

    # Allocate space for the 2D matrix that will have the
    # random variables as lines and each column will be an
    # observation
    observ_matrix = np.zeros([6, num_samples])

    # Use ground truth measurements, provided by the simulation
    # to create observation vectors.
    for obs_idx in tqdm(range(num_samples)):
        kinematics = uav.simGetGroundTruthKinematics()
        observ_matrix[:, obs_idx] = construct_measurement_vector(kinematics).squeeze()
        time.sleep(1/config.infer_freq_Hz)

    # Use numpy to calculate the covariance matrix for all
    # random variables
    process_noise = np.cov(observ_matrix, bias=True)

    # Restore the initial simulation state
    uav.disable()
    uav.client.reset()
    return process_noise

def estimate_measurement_noise(network: DetectionNetBench, num_samples: int = 100) -> np.ndarray:
    """
    Estimates the measurement noise covariance matrix (3x3) required for designing the
    KalmanFilter. The steps below summarize the process. Because of the inference step
    this may take a while to complete.

    1. Moves EgoUAV at a random position inside the map.
    2. Moves LeadingUAV inside EgoUAV's FOV, so it may detect it.
    3. Utilizes the provided network in order to estimate the offset of the LeadingUAV
    from EgoUAV's camera and offsets this estimation (on each axis) in order to match
    EgoUAV's center.
    4. Uses the API to get the ground truth position of the EgoUAV.
    5. Sums the results of 3 and 4. The result is an estimation for LeadingUAV's position.
    It is important to note that this estimation will be in EgoUAV's coordinate frame.
    6. Using the ground truth information provided by the simulation, we may have the exact
    position of the LeadingUAV and thus the true error of the estimation. Note that the ground
    truth is in the (global) world coordinate frame and thus in order to calculate the error we
    need to map 5 and 6 to the same coordinate frame.

    Using the above steps we may create a 2D matrix that will contain the errors of each observation
    for each one of the 3 axis (x, y, z). Then utilizing numpy's numpy.cov function, we may extract
    the covariance matrix for the measurement noise.

    In order for step 1 to work you have to set
    `"PhysicsEngineName":"ExternalPhysicsEngine"`
    in the settings file.
    """
    # Allocate a 2D matrix in order to store the errors (observations).
    # One row per random variable and a column per observation.
    observ_matrix = np.zeros([3, num_samples])

    # Create instances for the two UAVs
    egoUAV = EgoUAV(config.egoUAV_name, genmode=True)
    leadingUAV = LeadingUAV(config.leadingUAV_name, genmode=True)

    # Get the distance between the origins of the two coordinate frames defined
    # for each of the UAVs. (Reminder: This origin point is defined by where the
    # UAV spawns at)
    lead_ego_origin_dist = np.expand_dims((
        leadingUAV.sim_global_coord_frame_origin
        - egoUAV.sim_global_coord_frame_origin
        ).to_numpy_array(), axis=1)

    sample_idx = 0
    pbar = tqdm(total=num_samples)
    while sample_idx < num_samples:
        # Utilize create_sample from gendata.py to execute steps 1 and 2.
        # Have the EgoUAV at a height, so no shadows can be detected.
        img, _ = create_sample(egoUAV, leadingUAV, ego_box_lims_x=(-50, -60))
        # Evaluate the image returned
        bbox, _ = network.eval(img)
        if not bbox:
            # The image did not contain the LeadingUAV, restart the process.
            continue

        # Use the API to get the ground truth position of the EgoUAV.
        # Note: The kalman filter we will be utilizing only cares about the measurement
        # noise due to the error of the bbox and it's conversion to a distance on each axis.
        # The AirSim API notes: https://microsoft.github.io/AirSim/simple_flight/#state-estimation
        # that the position returned by the estimation function is in reality the ground truth position.
        # If we where to actually estimate the position of the EgoUAV a separate KF would be required.
        ego_pos_estim = np.expand_dims(
            egoUAV.simGetGroundTruthKinematics().position.to_numpy_array(),
            axis=1
        )
        # Estimate leadingUAV's pos
        # Since there is a bbox there get_distance_from_bbox will return an offset
        lead_pos_estim = ego_pos_estim + egoUAV.get_distance_from_bbox(bbox) # type: ignore
        
        # Calculate the error for each axis
        lead_pos_gt = np.expand_dims(leadingUAV.simGetObjectPose().position.to_numpy_array(), axis=1)
        lead_pos_gt += lead_ego_origin_dist

        # Store the error into the matrix
        observ_matrix[:, sample_idx] = abs(lead_pos_gt - lead_pos_estim).squeeze()

        # Update control variable
        sample_idx += 1
        pbar.update(n=1)

    pbar.close()
    # Calculate the covariance
    return np.cov(observ_matrix)
