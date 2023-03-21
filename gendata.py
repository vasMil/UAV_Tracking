import random
import time

import airsim
import matplotlib.pyplot as plt

from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from GlobalConfig import GlobalConfig as config


def generate_training_data(num_samples: int = 1):
    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Create the vehicles and perform the takeoff
    leadingUAV = LeadingUAV(client, "LeadingUAV", config.leadingUAV_seed)
    egoUAV = EgoUAV(client, "EgoUAV")
    egoUAV.lastAction.join()
    leadingUAV.lastAction.join()

    # Calculate the distance between the two UAVs
    init_lead_ego_dist = client.simGetObjectPose(object_name="LeadingUAV").position - client.simGetObjectPose(object_name="EgoUAV").position
    # Both UAVs takeoff and do nothing else, thus the distance on the z axis is only error
    init_lead_ego_dist.z_val = 0

    # Set the original positions of the two UAVs as their home
    ego_home_vec3r = egoUAV.simGetGroundTruthKinematics().position
    lead_home_vec3r = leadingUAV.simGetGroundTruthKinematics().position

    s: int = 0
    while s < num_samples:
        success = create_sample(egoUAV, leadingUAV, init_lead_ego_dist)
        if success:
            s+=1
        else:
            # Return home and retry
            egoUAV.moveToPositionAsync(*ego_home_vec3r)
            leadingUAV.moveToPositionAsync(*lead_home_vec3r)

    # Reset the location of all Multirotors
    client.reset()
    # Do not forget to disable all Multirotors
    leadingUAV.disable()
    egoUAV.disable()


def create_sample(egoUAV: EgoUAV, leadingUAV: LeadingUAV, init_lead_ego_dist: airsim.Vector3r) -> bool:
    # Move the ego vehicle at a random location
    egoUAV.moveToPositionAsync(
        random.uniform(*config.rand_move_box_x),
        random.uniform(*config.rand_move_box_y),
        random.uniform(*config.rand_move_box_z)
    ).join()

    if(egoUAV.hasCollided()): 
        print("Collision detected, may need to restart!")
        return False

    # Decide (randomly) on the offset of the leadingUAV in relationship with the egoUAV
    # on the x axis
    random_dist_from_ego_x = random.uniform(config.min_dist, config.max_dist)
    offset = airsim.Vector3r(random_dist_from_ego_x, 0, 0)
    
    # Using the FOV (90deg both horizontally and vertically) of the camera, 
    # and simple trigonometry we know that the maximum visible distance (both horizontally and vertically)
    # from the center of the frame is equal to the (relative) distance between the leadingUAV and the egoUAV.
    # Then I need to subtract the offset of the camera position on the leadingUAV (located at the front center).
    max_dist_y = abs(random_dist_from_ego_x - config.pawn_size_x/2)
    max_dist_z = abs(random_dist_from_ego_x - config.pawn_size_x/2)

    # Determine the max and min values for the other two axes (y, z),
    # so that the leadingUAV lies inside the egoUAV's view.
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
    max_global_y = max_dist_y - (config.pawn_size_x/2 + config.pawn_size_y/2)
    min_global_y = - max_dist_y + (config.pawn_size_x/2 + config.pawn_size_y/2)

    min_global_z = - max_dist_z/config.aspect_ratio + (config.pawn_size_z/2 + config.pawn_size_x/2)
    max_global_z = max_dist_z/config.aspect_ratio - (config.pawn_size_z/2 + config.pawn_size_x/2)

    # Calculate random offsets for y and z axis, using the bounds above
    # on the coordinate frame whose center is the egoUAV
    offset.y_val = random.uniform(min_global_y, max_global_y)
    offset.z_val = random.uniform(min_global_z, max_global_z)
    
    # Change the coordinate frame, so the center is the leadingUAV
    lead_local_pos = egoUAV.simGetGroundTruthKinematics().position + offset - init_lead_ego_dist

    # Adjust the z axis so the leading drone does not collide with the ground
    if lead_local_pos.z_val > leadingUAV.min_z:
        lead_local_pos.z_val = leadingUAV.min_z

    # Move the leadingUAV to those global coordinates
    leadingUAV.moveToPositionAsync(*lead_local_pos).join()
    if(leadingUAV.hasCollided()): 
        print("Collision detected, may need to restart!")
        return False

    # Wait for the leadingUAV to stop moving
    time.sleep(config.wait_stationarity)
    plt.imshow(egoUAV._getImage(view_mode=True))
    plt.show()

    # Save the image

    # Convert the global coordinates to egoUAV local coordinates
    # so you may use it as the ground truth label of the image
    
    # Save the label
    return True
