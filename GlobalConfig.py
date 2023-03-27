from typing import Optional

class GlobalConfig:
    """ Contains all the configuration options used in this project """
    # The seed used for the LeadingUAV random movements
    leadingUAV_seed: Optional[int] = 1

    # The magnitude of the velocity vector (in 3D space)
    leading_velocity: float = 4.

    # The upper an lower limit for the velocity on each axis of both UAVs
    max_vx, max_vy, max_vz = 4.,  1.,  0.2
    min_vx, min_vy, min_vz = 0., -1., -0.2

    # How much time should each move airsim api call last for 
    # (ex. moveByVelocityAsync, duration argument)
    move_duration: float = 4

    # When using a single client to communicate with AirSim for multiple UAVs
    # calling .join() on an Async function will result to all UAVs to wait on that Future.
    # Thus we need a sleep constant to wait on, as the other UAVs continue with their movements and
    # the current drone applies the Async movement, this way we do not overwhelm the server port.
    # TODO: Maybe in the future we need to use two clients, one for each UAV,
    # each interacting with AirSim using
    # different ports, thus .join() should be allowed.
    # AirSim discussions:
    # - https://github.com/microsoft/AirSim/issues/2971
    # - https://github.com/microsoft/AirSim/issues/2974
    # As a reminder: (source: https://microsoft.github.io/AirSim/apis/)
    # If you start another command then it automatically cancels the previous task and 
    # starts new command.
    # This allows to use pattern where your coded continuously does the sensing, 
    # computes a new trajectory to follow and issues that path to vehicle in AirSim.
    sleep_const: float = 2

    # Number of steps the game loop should take
    game_loop_steps: int = 10

    # Minimum acceptable error
    eps = 1e-8

    # Camera setting, change this if you change the defaults in setting.json (or vice versa)
    img_height = 256
    img_width = 144
    aspect_ratio = img_height / img_width # (=16:9)

    # Pawn size
    pawn_size_x = 0.98
    pawn_size_y = 0.98
    pawn_size_z = 0.29

    # Data generation
    # Seconds to wait for the UAVs to stop moving before capturing the image
    wait_stationarity = 7
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
