import os
import random
from typing import Tuple

import airsim
import torch
import pandas as pd
from torchvision.utils import save_image

from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from GlobalConfig import GlobalConfig as config


def create_sample(
        egoUAV: EgoUAV, 
        leadingUAV: LeadingUAV, 
    ) -> Tuple[torch.Tensor, airsim.Vector3r]:
    """
    Given the two UAVs and their initial offset
    return a tuple consisting of an image (torch.Tensor), which
    is from the egoUAV's perspective and contains the leadingUAV.
    The second element in the tuple should be another Tuple that contains
    the relative offset of the leadingUAV with respect to the egoUAV's position.
    This offset is the distance between the centers of the two UAVs.
    """
    reset_quarernion = airsim.to_quaternion(pitch=0, roll=0, yaw=0)
    while True:
        # The location to move the ego vehicle at
        ego_pos = airsim.Vector3r(
            random.uniform(*config.rand_move_box_x),
            random.uniform(*config.rand_move_box_y),
            random.uniform(*config.rand_move_box_z)
        )

        # Reset the orientation of the egoUAV before making any further changes
        ego_pose = airsim.Pose(position_val=ego_pos, orientation_val=reset_quarernion)
        egoUAV.simSetVehiclePose(ego_pose)
        # Update to the random orientation
        ego_pose.orientation = airsim.to_quaternion(pitch=random.uniform(-0.2, 0.2), roll=random.uniform(-1,1), yaw=0)
        egoUAV.simSetVehiclePose(ego_pose)
        
        _, offset = leadingUAV.sim_move_within_FOV(egoUAV, execute=False)
        lead_pos = offset + ego_pos

        # Reset the orientation of the egoUAV before making any further changes
        lead_pose = airsim.Pose(position_val=lead_pos, orientation_val=reset_quarernion)
        leadingUAV.simSetVehiclePose(lead_pose)
        # Update to the random orientation
        lead_pose.orientation = airsim.to_quaternion(pitch=random.uniform(-0.1, -0.1), roll=random.uniform(-1, 1), yaw=random.uniform(-0.2, 0.2))
        leadingUAV.simSetVehiclePose(lead_pose)

        if egoUAV.simTestLineOfSightToPoint(leadingUAV.simGetGroundTruthEnvironment().geo_point):
            # Capture the image
            img = egoUAV._getImage(view_mode=False)

            # Get the ground truth global offset (in the world frame) coordinates that will move the egoUAV
            # at the position of leadingUAV.
            # Adding this offset to the egoUAV's position (defined in coordinate frame with it's origin (0,0,0) being where the egoUAV spawned)
            # should result to the target_position the egoUAV needs to move to next.
            global_offset = leadingUAV.simGetObjectPose().position - egoUAV.simGetObjectPose().position
            
            # Return the sample
            return (img, global_offset)


def getLastImageIdx(csv_df: pd.DataFrame) -> int:
    if csv_df.empty:
        return -1
    img_name = str(csv_df.iloc[-1, 0])
    return int(img_name.split(sep=".")[0])


def generate_training_data(
        csv_file: str,
        root_dir: str, 
        num_samples: int = 10
    ) -> None:

    """
    It generates images of the egoUAV's perspective, with the leadingUAV in them.
    The leadingUAV may have different poses (random).
    It saves those images into the specified folder (root_dir) and appends to a csv file
    the name of the image and the "ground truth" offset of the leadingUAV from the egoUAV.
    
    The csv should exist and have the first row as:
    filename, x_offset, y_offset, z_offset

    In order for this function to work you have to set
    `"PhysicsEngineName":"ExternalPhysicsEngine"`
    in the settings file.
    """
    # Create a client to communicate with the UE
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Create the vehicles and perform the takeoff
    leadingUAV = LeadingUAV("LeadingUAV", seed=config.leadingUAV_seed, genmode=True)
    egoUAV = EgoUAV("EgoUAV", genmode=True)

    # Calculate the distance between the two UAVs
    init_lead_ego_dist = client.simGetObjectPose(object_name="LeadingUAV").position - client.simGetObjectPose(object_name="EgoUAV").position
    # Both UAVs takeoff and do nothing else, thus the distance on the z axis is only error
    init_lead_ego_dist.z_val = 0

    # Load the csv file into a DataFrame
    csv_df = pd.read_csv(csv_file)

    # Get the index of the last saved img in the dataset and increment it by one
    # to get the index of the sample image you will produce.
    sample_idx = getLastImageIdx(csv_df) + 1
    for s in range(num_samples):
        img, offset = create_sample(egoUAV, leadingUAV)
        # Save the image
        img_filename = str(sample_idx + s).zfill(config.filename_leading_zeros) + ".png"
        save_image(img, os.path.join(root_dir, img_filename))
        # Update the df
        new_row_df = pd.DataFrame([[img_filename, *offset]], columns=csv_df.columns)
        csv_df = pd.concat([csv_df, new_row_df], ignore_index=True)
        csv_df.to_csv(csv_file, index=False)

    # Reset the location of all Multirotors
    client.reset()
    # Do not forget to disable all Multirotors
    leadingUAV.disable()
    egoUAV.disable()
