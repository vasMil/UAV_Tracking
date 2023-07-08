import math
from typing import Optional, Literal

class GlobalConfig:
    """ Contains all the configuration options used in this project """
    # The port number airsim if configured to listen to
    port = 41451
    # The seed used for the LeadingUAV random movements
    leadingUAV_seed: Optional[int] = 10

    # The magnitude of the velocity vector (in 3D space)
    uav_velocity: float = 5.

    # The minimum score, for which a detection is considered
    # valid and thus is translated to EgoUAV movement.
    score_threshold: float = 0.1

    # The upper an lower limit for the velocity on each axis of both UAVs
    max_vx, max_vy, max_vz = 5.,  5,  5
    min_vx, min_vy, min_vz = 1., -5, -5

    # How much time should each move airsim api call last for
    # (ex. moveByVelocityAsync, duration argument)
    move_duration: float = 4

    # Number of steps the game loop should take
    game_loop_steps: int = 10

    # Minimum acceptable error
    eps = 1e-5

    # Pawn size
    pawn_size_x = 0.98
    pawn_size_y = 0.98
    pawn_size_z = 0.29

    # Camera settings, change this if you change the defaults in setting.json (or vice versa)
    img_height = 144
    img_width = 256
    aspect_ratio = img_width / img_height  # (=16:9)

    # Found by using simGetCameraInfo()
    horiz_fov = math.radians(89.90362548828125)
    # How to calculate the vertical FOV:
    # https://github.com/microsoft/AirSim/issues/902
    vert_fov = math.radians((img_height / img_width) * 89.90362548828125)
    camera_offset_x = 0.4599999785423279
    # The focal length of our camera has already been calculated as
    # F = (P * D) / W, where
    # P is the width of the object in pixels,
    # D is the actual distance of the camera, from that object,
    # W is the width of the object in meters
    # We placed the two UAVs the one infront of the other, with 3.5
    # meters between their centers. Thus the actual distance between the
    # camera and the back of the leadingUAV is 3.5 - camera_offset_x - pawn_size_x/2.
    focal_length_x = 46 * (3.5 - camera_offset_x - pawn_size_x/2) / pawn_size_y
    focal_length_y = 13 * (3.5 - camera_offset_x - pawn_size_x/2) / pawn_size_z

    # Data generation
    # Bounds for random distance between the two UAVs
    min_dist = 1.5
    max_dist = 10
    # Box to allow random movement of the egoUAV in
    rand_move_box_x = (-10, 10)
    rand_move_box_y = (-10, 10)
    rand_move_box_z = (-10, -1) # Min dist from the ground is -1 (i.e. 1m above ground level)

    # The sample images are stored as (image_index).(fromat), where the image_index is a number
    # in [0,9999] thus we add leading zeros to have a uniform representation of the indexes
    filename_leading_zeros = 4

    # The position calculated inside create_sample is slightly different
    # by the one calculated inside generate_training_data,
    # because when capturing those the UAVs are not completely stationary.
    # If you want to minimize this allowed threshold you will have to
    # increase wait_stationarity.
    measurement_threshold = (0.1, 0.1, 0.1)

    # Detection NN constants
    num_epochs = 25
    default_batch_size = 4
    num_workers = 0
    sgd_learning_rate = 0.001
    sgd_momentum = 0.9
    sgd_weight_decay = 0.0005
    scheduler_milestones = [100]
    scheduler_gamma = 0.1

    # Pytorch Profiler constants
    profile = False
    prof_wait = 1
    prof_warmup = 1
    prof_active = 3
    prof_repeat = 1

    # Controller constants - converting bbox to velocity
    # Weights are added for y and z coords. This is helpful since the
    # UAV is not going to reach the target position, using a constant
    # velocity in this small time interval (1/inference_freq_Hz).
    # The most important aspect of the tracking is to preserve the
    # LeadingUAV inside your FOV. Thus we require, at the end
    # of each command, to have the LeadingUAV as close to the center
    # of our FOV as possible. In order to achieve this we need to allocate
    # most of velocity's magnitude towards the z axis. The same applies for
    # the y axis, but this weight may be less, since we also take advantage
    # of the yaw_mode, in order to rotate our camera and look at all times
    # at the LeadingUAV.
    weight_vel_x, weight_vel_y, weight_vel_z = 1, 1, 4

    # Logger settings
    sim_fps = 60
    simulation_time_s = 60
    camera_fps = 30
    infer_freq_Hz = 30
    filter_freq_Hz = 30
    filter_type: Literal["None", "KF"] = "KF"
    leadingUAV_update_vel_interval_s = 2
