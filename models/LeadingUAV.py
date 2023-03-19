from typing import Optional, Tuple
import math

import airsim
from msgpackrpc.future import Future
from numpy.random import RandomState
import torch

from models.UAV import UAV
from GlobalConfig import GlobalConfig as config

class LeadingUAV(UAV):
    """ 
    Defines the behaviour of the leading UAV - the one being chased.
    As a first approach it should move randomly, without having to worry about
    obstacles in the AirSim map.
    """
    def __init__(
            self, 
            client: airsim.MultirotorClient, 
            name: str, 
            seed: Optional[int] = None
        ) -> None:
        super().__init__(client, name)
        # Takeoff
        self.lastAction = client.takeoffAsync(vehicle_name=name)
        # Should not allow the UAV to go below a certain height, since it may collide with the ground.
        # In this case we won't allow it to go lower than the position at which it is placed after takeoff.
        self.min_z = client.simGetGroundTruthKinematics(vehicle_name=self.name).position.z_val
        # Configure random state
        self._randomState = RandomState(seed)


    def random_move(self) -> Tuple[Future, torch.Tensor] :
        """
        Moves the UAV using random velocity values for the x and y axis and a height value for the z axis
        - vx: Should always be a positive value, we will not allow the leading UAV to go backwards
        - vy: Should allow for movement with max velocity both to the left and to the right
        - vz : Positive (NED) velocity moves the drone down, negative moves the drone up.
        Should be careful not to let z value be less than self.min_z
        
        Since the velocity is a vector in 3D space, it should have a constant magnitude
        equal to config.leading_velocity and a changing direction 
        (with the exeption that x axis should be bounded to only positive values in order to only allow t forward movement).
        In order to achieve this (i.e. constant magnitude and variable direction) we will randomize the contribution
        of each axis (x, y, z) to the final velocity vector. This contribution should be normalized so that the square root
        of the sum of the dimension values squared (i.e. the magnitude) is equal to 1.
        After that we may multiply with config.leading_velocity in order to get the final velocity vector.
        """
        # Create the random contribution vector
        cx = self._randomState.uniform(config.min_vx, config.max_vx)
        cy = self._randomState.uniform(config.min_vy, config.max_vy)
        cz = self._randomState.uniform(config.min_vz, config.max_vz)
        # Correct for movement that will lead the drone to get at a lower height 
        # than the one specified by self.min_z (we test for greater than or equal since we are using NED coordinates)
        if (self.client.simGetGroundTruthKinematics(vehicle_name=self.name).position.z_val >= self.min_z and 
                cz > 0):
            cz *= -1
        contrib_vec = torch.tensor([cx, cy, cz], dtype=float)
        # Normalize the contribution vector
        contrib_vec.div_(contrib_vec.pow(2).sum().sqrt().item())
        # Calculate the velocity vector
        velocity_vec = config.leading_velocity * contrib_vec
        (vx, vy, vz) = velocity_vec.tolist()
        # Make sure that the magnitude of the velocity is equal to config.leading_velocity
        assert(abs(velocity_vec.pow(2).sum().sqrt() - config.leading_velocity) < config.eps)
        self.lastAction = self.client.moveByVelocityAsync(vx, vy, vz, config.move_duration, vehicle_name=self.name)
        print(f"{self.name} moveByVelocityAsync: vx = {vx}, vy = {vy}, vz = {vz}")
        return (self.lastAction, velocity_vec)
    