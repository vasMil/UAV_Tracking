import os
import random
import time
from typing import Optional, Tuple

import airsim
import torch
import pandas as pd
from torchvision.utils import save_image

from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from GlobalConfig import GlobalConfig as config


def generate_training_data(
        csv_file: str,
        root_dir: str, 
        num_samples: int = 10
    ) -> None:

    """
    Creates a client that connects to the UAVs, using their names "LeadingUAV" and "EgoUAV",
    it generates images of the egoUAV's perspective, with the leadingUAV in them.
    It then saves those images into the specified folder (root_dir) and appends to a csv file
    the name of the image and the "ground truth" offset of the leadingUAV from the egoUAV.
    
    The csv should exist and have the first row as:
    filename, x_offset, y_offset, z_offset
    """
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

    # Load the csv file into a DataFrame
    csv_df = pd.read_csv(csv_file)
    # Get the index of the last saved img in the dataset and increment it by one
    # to get the index of the sample image you will produce.
    sample_idx = getLastImageIdx(csv_df) + 1
    s: int = 0
    while s < num_samples:
        img, offset = create_sample(egoUAV, leadingUAV, init_lead_ego_dist)
        if img is not None and offset is not None:
            # Save the image and update the csv
            img_filename = str(sample_idx + s).zfill(config.filename_leading_zeros) + ".png"
            save_image(img, os.path.join(root_dir, img_filename))
            new_row_df = pd.DataFrame([[img_filename, *offset]], columns=csv_df.columns)
            csv_df = pd.concat([csv_df, new_row_df], ignore_index=True)
            csv_df.to_csv(csv_file, index=False)
            
            # Make sure that the label is correct
            next_ego_pos = (egoUAV.simGetGroundTruthKinematics().position.to_numpy_array() + offset)
            curr_lead_pos = leadingUAV.simGetGroundTruthKinematics().position.to_numpy_array()
            # Canonicalize the curr_lead_pos so it is egoUAV's coordinate frame
            curr_lead_pos = curr_lead_pos - init_lead_ego_dist.to_numpy_array()
            # Make sure that the two positions match (there should be a small error, 
            # because when we called simGetObjectPose the UAVs where not completely still, thus
            # they moved slightly in the time gap between that measurement and when we called simGetGroundTruthKinematics).
            # Thus I need to make sure that this error does not exceed a certain threshold.

            assert(all(curr_lead_pos - next_ego_pos < config.measurement_threshold))
            
            s += 1
        else:
            # Return home and retry
            print("returning back home")
            egoUAV.moveToPositionAsync(*ego_home_vec3r).join()
            leadingUAV.moveToPositionAsync(*lead_home_vec3r).join()

    # Reset the location of all Multirotors
    client.reset()
    # Do not forget to disable all Multirotors
    leadingUAV.disable()
    egoUAV.disable()


def create_sample(
        egoUAV: EgoUAV, 
        leadingUAV: LeadingUAV, 
        init_lead_ego_dist: airsim.Vector3r,
    ) -> Tuple[Optional[torch.Tensor], Optional[list]]:
    """
    Given the two UAVs and their initial offset
    return a tuple consisting of an image (torch.Tensor), which
    is from the egoUAV's perspective and contains the leadingUAV.
    The second element in the tuple should be another Tuple that contains
    the relative offset of the leadingUAV with respect to the egoUAV's position.

    TODO: (make sure you move the x dimension offset backwards by half the pawn's depth, 
    because the current offset is considering the egoUAV's camera location as its (0,0,0) point).
    """
    # Move the ego vehicle at a random location
    egoUAV.moveToPositionAsync(
        random.uniform(*config.rand_move_box_x),
        random.uniform(*config.rand_move_box_y),
        random.uniform(*config.rand_move_box_z)
    ).join()

    if(egoUAV.hasCollided()): 
        print("Collision detected, may need to restart!")
        return (None, None)

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
    max_y_offset = max_dist_y - (config.pawn_size_x/2 + config.pawn_size_y/2)
    min_y_offset = - max_dist_y + (config.pawn_size_x/2 + config.pawn_size_y/2)

    min_z_offset = - max_dist_z/config.aspect_ratio + (config.pawn_size_z/2 + config.pawn_size_x/2)
    max_z_offset = max_dist_z/config.aspect_ratio - (config.pawn_size_z/2 + config.pawn_size_x/2)

    # Calculate random offsets for y and z axis, using the bounds above
    # on the coordinate frame whose center is the egoUAV
    offset.y_val = random.uniform(min_y_offset, max_y_offset)
    offset.z_val = random.uniform(min_z_offset, max_z_offset)
    
    # Change the coordinate frame, so the center is the leadingUAV
    lead_local_pos = egoUAV.simGetGroundTruthKinematics().position + offset - init_lead_ego_dist

    # Adjust the z axis so the leading drone does not collide with the ground
    if lead_local_pos.z_val > leadingUAV.min_z:
        lead_local_pos.z_val = leadingUAV.min_z

    # Move the leadingUAV to those global coordinates
    leadingUAV.moveToPositionAsync(*lead_local_pos).join()
    if(leadingUAV.hasCollided()): 
        print("Collision detected, may need to restart!")
        return (None, None)

    # Wait for the leadingUAV to stop moving
    time.sleep(config.wait_stationarity)
    
    # Capture the image
    img = egoUAV._getImage(view_mode=False)

    # Get the ground truth global offset (in the world frame) coordinates that will move the egoUAV
    # at the position of leadingUAV.
    # Adding this offset to the egoUAV's position (defined in coordinate frame with it's origin (0,0,0) being where the egoUAV spawned)
    # should result to the target_position the egoUAV needs to move to next.
    global_offset = leadingUAV.simGetObjectPose().position - egoUAV.simGetObjectPose().position
    
    # Return the sample
    return (img, [*global_offset])


def getLastImageIdx(csv_df: pd.DataFrame) -> int:
    if csv_df.empty:
        return -1
    img_name = str(csv_df.iloc[-1, 0])
    return int(img_name.split(sep=".")[0])
