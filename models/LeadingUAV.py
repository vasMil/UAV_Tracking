from typing import Optional, Tuple
import random

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
            name: str,
            port: int = 41451,
            seed: Optional[int] = None
        ) -> None:
        super().__init__(name, port)
        # Configure random state
        self._randomState = RandomState(seed)


    def random_move(self) -> Tuple[Future, torch.Tensor]:
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
        contrib_vec = torch.tensor([cx, cy, cz], dtype=torch.float)
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
    
    def sim_move_within_FOV(self, uav: UAV, print_offset: bool = False) -> Tuple[Future, airsim.Vector3r]:
        """
        Move the leadingUAV at a random position, that is within the bounds
        of uav's Field Of View (FOV).
        """
        # Decide (randomly) on the offset of the leadingUAV in relationship with the egoUAV
        # on the x axis
        random_dist_x = random.uniform(config.min_dist, config.max_dist)
        offset = airsim.Vector3r(random_dist_x, 0, 0)
        
        # Using the FOV (90deg both horizontally and vertically) of the camera, 
        # and simple trigonometry we know that the maximum visible distance (both horizontally and vertically)
        # from the center of the frame is equal to the (relative) distance between the leadingUAV and the egoUAV.
        # Then I need to subtract the offset of the camera position on the uav (located at the front center).
        max_dist_y = abs(random_dist_x - config.pawn_size_x/2)
        max_dist_z = abs(random_dist_x - config.pawn_size_x/2)

        # Determine the max and min values for the other two axes (y, z),
        # so that the leadingUAV lies inside the uav's view.
        # 
        # As a reminder the size of the pawn is (98x98x29 cm). (source: https://github.com/microsoft/AirSim/issues/2059)
        # The pawn_size fixes are required because of the camera position on the egoUAV in
        # relationship with the "last" pixel of the leading vehicle we require to be visible.
        # In order to simplify the process we consider the leading vehicle as it's bounding box.
        # 1) For the y axis, when the leadingUAV is on the left in egoUAV's view, the bottom left corner of
        # the leadingUAV's bounding box is the "last" pixel (top view of the bounding box).
        # 2) The "last" pixel, when the leadingUAV is on the right, is the one at the bottom right (top view of the bounding box).
        # 3) For the z axis, when the leadingUAV is higher than the egoUAV, the "last" pixel is
        # the top right (side view of the bounding box).
        # 4) The "last" pixel, when the leadingUAV is lower, is the bottom right (side view of the bounding box).
        # 
        # We also need to take into account the height of the Multirotor, when deciding on the offset on the z axis.
        #
        # We do not consider the lookahead error introduced by the carrot algorithm, used
        # in moveToPositionAsync. (source: https://github.com/microsoft/AirSim/issues/4293)
        # 
        # Although both the horizontal and the vertical FOV is 90degrees,
        # the scaling of the image (aspect ratio) limits the vertical dimension.
        # The division with the aspect_ratio of the camera would not be necessary if
        # it had an aspect ratio of 1:1.
        max_y_offset = max_dist_y - (config.pawn_size_x/2 + config.pawn_size_y/2)
        min_y_offset = - max_dist_y + (config.pawn_size_x/2 + config.pawn_size_y/2)

        min_z_offset = - max_dist_z/config.aspect_ratio + (config.pawn_size_z/2 + config.pawn_size_x/2)
        max_z_offset = max_dist_z/config.aspect_ratio - (config.pawn_size_z/2 + config.pawn_size_x/2)

        # Calculate random offsets for y and z axis, using the bounds above
        # on the coordinate frame whose center is the uav
        offset.y_val = random.uniform(min_y_offset, max_y_offset)
        offset.z_val = random.uniform(min_z_offset, max_z_offset)
        
        # Change the coordinate frame, so the center is the leadingUAV
        lead_ego_dist_coord_frame_offset = self.sim_global_coord_frame_origin - uav.sim_global_coord_frame_origin
        lead_local_pos = uav.simGetGroundTruthKinematics().position + offset - lead_ego_dist_coord_frame_offset

        # Adjust the z axis so the leading drone does not collide with the ground
        if lead_local_pos.z_val > self.min_z:
            lead_local_pos.z_val = self.min_z
            offset.z_val = self.min_z - uav.simGetGroundTruthKinematics().position.z_val

        # offset = airsim.Vector3r(14.524488722836576, -3.9391929802995307, -6.169142644866273)
        # lead_ego_dist_coord_frame_offset = self.sim_global_coord_frame_origin - uav.sim_global_coord_frame_origin
        # lead_local_pos = uav.simGetGroundTruthKinematics().position + offset - lead_ego_dist_coord_frame_offset

        if print_offset:
            print(f"Target offset: (x={offset.x_val}, y={offset.y_val}, z={offset.z_val})")

        # Move the leadingUAV to those global coordinates
        self.lastAction = self.moveToPositionAsync(*lead_local_pos)
        return self.lastAction, offset
