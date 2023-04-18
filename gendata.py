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
    leadingUAV = LeadingUAV("LeadingUAV", seed=config.leadingUAV_seed)
    egoUAV = EgoUAV("EgoUAV")
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
        img, offset = create_sample(egoUAV, leadingUAV)
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

    leadingUAV.sim_move_within_FOV(egoUAV)

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
