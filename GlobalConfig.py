import math
from typing import Optional

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
    score_threshold: float = 0.5

    # The upper an lower limit for the velocity on each axis of both UAVs
    max_vx, max_vy, max_vz = 4.,  2.,  0
    min_vx, min_vy, min_vz = 0., -2.,  0

    # How much time should each move airsim api call last for
    # (ex. moveByVelocityAsync, duration argument)
    move_duration: float = 4

    # When using a single client to communicate with AirSim for multiple UAVs
    # calling .join() on an Async function will result to all UAVs waiting on that Future.
    # Thus we need a sleep constant to wait on, as the other UAVs continue with their movements and
    # the current drone applies the Async movement, this way we do not overwhelm the server port.
    # AirSim discussions:
    # - https://github.com/microsoft/AirSim/issues/2971
    # - https://github.com/microsoft/AirSim/issues/2974
    # As a reminder: (source: https://microsoft.github.io/AirSim/apis/)
    # If you start another command then it automatically cancels the previous task and 
    # starts new command.
    # This allows to use a pattern, where your code continuously does the sensing, 
    # computes a new trajectory to follow and issues that path to vehicle in AirSim.
    sleep_const: float = 2

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
    # The probability with which an empty image will be generated
    gen_empty_img_prob = 0.05
    # Bounds for random distance between the two UAVs
    min_dist = 3
    max_dist = 15
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
