from typing import Optional

class GlobalConfig:
    """ Contains all the configuration options used in this project """
    # The seed used for the LeadingUAV random movements
    leadingUAV_seed: Optional[int] = 1
    
    # Decide if the movements of the LeadingUAV should be 
    # dependant to the current direction of the vehicle (smooth movement)
    leadingUAV_smooth: bool = False

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
    # TODO: Maybe in the future we need to use two clients, one for each UAV, each interacting with AirSim using
    # different ports, thus .join() should be allowed.
    # AirSim discussions: 
    # - https://github.com/microsoft/AirSim/issues/2971
    # - https://github.com/microsoft/AirSim/issues/2974
    # As a reminder: (source: https://microsoft.github.io/AirSim/apis/)
    # If you start another command then it automatically cancels the previous task and starts new command. 
    # This allows to use pattern where your coded continuously does the sensing, 
    # computes a new trajectory to follow and issues that path to vehicle in AirSim.
    sleep_const: float = 2

    # Number of steps the game loop should take
    game_loop_steps: int = 50

    # Minimum acceptable error
    eps = 1e-8