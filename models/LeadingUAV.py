from typing import Optional, Tuple
import random
import math

import airsim
from msgpackrpc.future import Future
from numpy.random import RandomState
import torch

from constants import LEADING_UAV_NAME, EPS,\
    PAWN_SIZE_X, PAWN_SIZE_Y, PAWN_SIZE_Z,\
    ASPECT_RATIO,\
    MIN_DIST, MAX_DIST
from models.UAV import UAV

class LeadingUAV(UAV):
    """ 
    Defines the behaviour of the leading UAV - the one being chased.
    As a first approach it should move randomly, without having to worry about
    obstacles in the AirSim map.
    """
    def __init__(
            self,
            name: str = LEADING_UAV_NAME,
            vel_magn: float = 0,
            max_vel: Tuple[float, float, float] = (0, 0, 0,),
            min_vel: Tuple[float, float, float] = (0, 0, 0,),
            port: int = 41451,
            seed: Optional[int] = None,
            genmode: bool = False
        ) -> None:
        super().__init__(name=name,
                         vel_magn=vel_magn,
                         port=port,
                         genmode=genmode)
        self.vel_magn = vel_magn
        self.max_vel = max_vel
        self.min_vel = min_vel
        # Configure random state
        self._randomState = RandomState(seed)


    def random_move(self, command_time: float, silence: bool = True) -> Tuple[Future, torch.Tensor]:
        """
        Moves the UAV using random velocity values for the x and y axis and a
        height value for the z axis.
        - vx: Should always be a positive value, we will not allow the
        leading UAV to go backwards
        - vy: Should allow for movement with max velocity both to the
        left and to the right
        - vz : Positive (NED) velocity moves the drone down, negative
        moves the drone up.

        Args:
        - command_time: How long to run this move for
        Using this information we can ensure that the z value cannot
        be less than self.min_z

        Returns:
        A Tuple containing:
        - The Future returned by the AirSim API, which is also available
        at self.lastAction
        - A torch.Tensor with the random velocity applied.

        More Info:
        Since the velocity is a vector in 3D space, it should have a constant
        magnitude equal to config.uav_velocity and a changing direction,
        bounded by the constants [min_vx, max_vx], [min_vy, max_vy],
        [min_vz, max_vz] defined in config.
        In order to achieve this (i.e. constant magnitude and variable
        direction) we will randomize the contribution of each axis (x, y, z)
        to the final velocity vector.
        This contribution should be normalized so that the square root
        of the sum of the dimension values squared (i.e. the magnitude)
        is equal to 1.
        After that we may multiply with config.uav_velocity in order to get the
        final velocity vector.

        This is an overcontrained problem - decide the BOUNDED velocity on each
        axis randomly, such that the final 3D vector has a magnitude of
        config.uav_velocity. Consider the case where the first 2 axis randomly
        decide 0m/s. The velocity for the third axis will not be random and
        may also exceed it's bounds (ex. config.uav_velocity = 4,
        but the upper bound for the velocity on this axis is 1).
        """
        # Create the random contribution vector
        cx = self._randomState.uniform(self.min_vel[0], self.max_vel[0])
        cy = self._randomState.uniform(self.min_vel[1], self.max_vel[1])
        cz = self._randomState.uniform(self.min_vel[2], self.max_vel[2])
        contrib_vec = torch.tensor([cx, cy, cz], dtype=torch.float)

        # Normalize the contribution vector
        contrib_vec.div_(contrib_vec.pow(2).sum().sqrt().item())
        # Calculate the velocity vector
        velocity_vec = self.vel_magn * contrib_vec
        (vx, vy, vz) = velocity_vec.tolist()

        # Correct for movement that will lead the drone to get at a lower height
        # than the one specified by self.min_z
        # (we test for greater than or equalsince we are using NED coordinates)
        if (self.simGetGroundTruthKinematics()
                .position.z_val
                + command_time*vz >= self.min_z
        ):
            vz *= -1
        
        # Make sure that the magnitude of the velocity is equal to config.leading_velocity
        assert(abs(velocity_vec.pow(2).sum().sqrt() - self.vel_magn) < EPS)
        self.lastAction = self.moveByVelocityAsync(vx, vy, vz,
                                                   command_time,
                                                   yaw_mode=airsim.YawMode(False, math.degrees(math.atan(vy/vx)))
                                                )
        if not silence:
            print(f"{self.name} moveByVelocityAsync: vx = {vx}, vy = {vy}, vz = {vz}")
        return (self.lastAction, velocity_vec)
    
    def _get_FOV_bounds(self, x_dist_m: float) -> Tuple[float, float, float, float]:
        """
        Given the distance between the centers of the two UAVs, returns the relative
        distances between the two that would allow this-(LeadingUAV) to be within EgoUAVs
        FOV.

        Args:
        - x_dist_m: The distance on x axis between the two UAVs (in meters)

        Returns:
        A tuple in the form of (min_y, max_y, min_z, max_z)
        """
        # Using the FOV (90deg both horizontally and vertically) of the camera, 
        # and simple trigonometry we know that the maximum visible distance (both horizontally and vertically)
        # from the center of the frame is equal to the (relative) distance between the leadingUAV and the egoUAV.
        # Then I need to subtract the offset of the camera position on the uav (located at the front center).
        max_dist_y = abs(x_dist_m - PAWN_SIZE_X/2)
        max_dist_z = abs(x_dist_m - PAWN_SIZE_X/2)

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
        max_y_offset = max_dist_y - (PAWN_SIZE_X/2 + PAWN_SIZE_Y/2)
        min_y_offset = - max_dist_y + (PAWN_SIZE_X/2 + PAWN_SIZE_Y/2)

        min_z_offset = - max_dist_z/ASPECT_RATIO + (PAWN_SIZE_Z/2 + PAWN_SIZE_X/2)
        max_z_offset = max_dist_z/ASPECT_RATIO - (PAWN_SIZE_Z/2 + PAWN_SIZE_X/2)
        return (min_y_offset, max_y_offset, min_z_offset, max_z_offset,)

    def sim_move_within_FOV(self,
                            uav: UAV,
                            execute: bool = True,
                            print_offset: bool = False
                        ) -> Tuple[Optional[Future], airsim.Vector3r]:
        """
        Move the LeadingUAV at a random position, that is within the bounds
        of uav's Field Of View (FOV).

        Args:
        - uav: The uav's FOV in which to move
        - execute: Whether to execute the command, or just return the coords

        Returns:
        If execute is True, it returns the Future and the target offset between the two UAVs
        If execute is False, None and the offset from uav's center to move at.
        """
        # Decide (randomly) on the offset of the leadingUAV in relationship with the egoUAV
        # on the x axis
        random_dist_x = random.uniform(MIN_DIST, MAX_DIST)
        offset = airsim.Vector3r(random_dist_x, 0, 0)

        (min_y_offset, max_y_offset, min_z_offset, max_z_offset) = self._get_FOV_bounds(random_dist_x)

        # Calculate random offsets for y and z axis, using the bounds above
        # on the coordinate frame whose center is the uav
        offset.y_val = random.uniform(min_y_offset, max_y_offset)
        offset.z_val = random.uniform(min_z_offset, max_z_offset)
        
        # Change the coordinate frame, so the center is the leadingUAV
        lead_local_pos = uav.simGetGroundTruthKinematics().position \
            - uav.sim_global_coord_frame_origin \
            + offset \
            - self.sim_global_coord_frame_origin

        # Adjust the z axis so the leading drone does not collide with the ground
        if lead_local_pos.z_val > self.min_z:
            lead_local_pos.z_val = self.min_z
            offset.z_val = self.min_z - uav.simGetGroundTruthKinematics().position.z_val

        if print_offset:
            print(f"Target offset: (x={offset.x_val}, y={offset.y_val}, z={offset.z_val})")

        # Move the leadingUAV to those global coordinates
        if execute:
            lead_local_pos = (lead_local_pos.x_val, lead_local_pos.y_val, lead_local_pos.z_val)
            self.lastAction = self.moveToPositionAsync(*lead_local_pos)
            return self.lastAction, offset
        else:
            return None, offset

    def sim_move_out_FOV(self,
                         uav: UAV,
                         execute: bool = True
                        ) -> Tuple[Optional[Future], airsim.Vector3r]:
        """
        Moves the LeadingUAV out of uav's FOV,

        Args:
        - uav: The uav's FOV out of which to move
        - execute: Whether to execute the command, or just return the coords

        Returns:
        If execute is True, it returns the Future and the target offset between the two UAVs
        If execute is False, None and the target position for LeadingUAV, in it's
        coordinate frame.
        """
        # Excess offset, to add to max visible distances, in order to move out of
        # uav's FOV.
        EXCESS_OFFSET = 10

        # Decide (randomly) on the offset of the leadingUAV in relationship with the egoUAV
        # on the x axis
        random_dist_x = random.uniform(3.5, 5)
        offset = airsim.Vector3r(random_dist_x, 0, 0)

        # Calculate the offsets that will move you out of uav's FOV
        (_, max_y_offset, _, max_z_offset,) = self._get_FOV_bounds(random_dist_x)
        offset.y_val = max_y_offset + EXCESS_OFFSET
        offset.z_val = max_z_offset + EXCESS_OFFSET

        # Change the coordinate frame, so the center is the leadingUAV
        lead_ego_dist_coord_frame_offset = self.sim_global_coord_frame_origin - uav.sim_global_coord_frame_origin
        lead_local_pos = uav.simGetGroundTruthKinematics().position + offset - lead_ego_dist_coord_frame_offset
        print(lead_local_pos)
        if execute:
            lead_local_pos = (lead_local_pos.x_val, lead_local_pos.y_val, lead_local_pos.z_val)
            self.lastAction = self.moveToPositionAsync(*lead_local_pos)
            return self.lastAction, offset
        else:
            return None, lead_local_pos
