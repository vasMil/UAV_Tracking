from typing import Tuple, List
import os
import shutil
import random
import math

import airsim
import torch
from torchvision.utils import save_image
import json
import numpy as np

from constants import RAND_MOVE_BOX_X, RAND_MOVE_BOX_Y, RAND_MOVE_BOX_Z,\
    FILENAME_LEADING_ZEROS, LEADING_UAV_NAME
from project_types import Bbox_dict_t
from models.EgoUAV import EgoUAV
from models.LeadingUAV import LeadingUAV
from models.BoundingBox import BoundingBox, BoundingBoxFactory
from utils.image import add_bbox_to_image, load_png_as_tensor
from nets.DetectionNetBench import DetectionNetBench

def create_sample(
        egoUAV: EgoUAV,
        leadingUAV: LeadingUAV,
        ego_box_lims_x: Tuple[float, float] = RAND_MOVE_BOX_X,
        ego_box_lims_y: Tuple[float, float] = RAND_MOVE_BOX_Y,
        ego_box_lims_z: Tuple[float, float] = RAND_MOVE_BOX_Z
    ) -> Tuple[torch.Tensor, airsim.Vector3r]:
    """
    Given two UAVs return a tuple consisting of an image (torch.Tensor) and the offset.\
    This image is egoUAV's perspective and contains the leadingUAV.\
    The second element of the tuple is the offset of the leadingUAV with respect to the\
    egoUAV's position.\
    (This offset is the distance between the centers of the two UAVs.)
    """
    reset_quarernion = airsim.to_quaternion(pitch=0, roll=0, yaw=0)
    while True:
        # The location to move the ego vehicle at
        ego_pos = airsim.Vector3r(
            random.uniform(*ego_box_lims_x),
            random.uniform(*ego_box_lims_y),
            random.uniform(*ego_box_lims_z)
        )

        # Reset the orientation of the egoUAV before making any further changes
        ego_pose = airsim.Pose(position_val=ego_pos, orientation_val=reset_quarernion)
        egoUAV.simSetVehiclePose(ego_pose)
        # Update to the random orientation
        # TODO: Allow yaw variation. In order to position the LeadingUAV, you will have to
        # project the position returned by sim_move_within_FOV to the coordinate frame defined
        # by the orientation of EgoUAV's camera (i.e. rotate to EgoUAV's yaw angle).
        # You may be better off doing this inside sim_move_within_FOV.
        ego_pose.orientation = airsim.to_quaternion(pitch=random.uniform(-0.2, 0.2), roll=random.uniform(-1,1), yaw=0)
        egoUAV.simSetVehiclePose(ego_pose)

        _, offset = leadingUAV.sim_move_within_FOV(egoUAV, execute=False)
        lead_pos = offset + ego_pos - leadingUAV.sim_global_coord_frame_origin

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

def clean_generated_frames_and_bboxes(frames_root_path: str,
                                      json_path: str,
                                      net: DetectionNetBench,
                                      last_checked_image: int = 0):
    """
    Loads each generated image and the corresponding bbox evaluates the frame
    using the provided net. If each point of the two bboxes differ more than
    10 pixels at each dimension then the image with both the bboxes is copied
    to a temp folder and I am prompted to evaluate it and decide upon the following
    options:
    1. Delete the frame entirely
    2. Keep the bbox recorded by airsim's mesh
    3. Keep the bbox returned by the net
    """
    def handle_user_input(has_option_r: bool) -> int:
        print("Options:")
        print("d -> to delete the frame entirely")
        if has_option_r:
            print("r -> to keep the bbox proposed from the net (red)")
        print("b -> to keep the bbox proposed from airsim (black)")
        print("back -> go to previous image")
        print("stop -> break out of the loop")
        while True:
            user_inp = input("Your choise: ")
            if user_inp == "d":
                return 0
            elif user_inp == "r" and has_option_r:
                return 1
            elif user_inp == "b":
                return 2
            elif user_inp == "back":
                return 3
            elif user_inp == "stop":
                return 4
            else:
                print("Invalid input. Try again.")
            
    temp_path = os.path.join(frames_root_path, "temp/")
    temp_frame_path = os.path.join(temp_path, "temp.png")
    os.mkdir(temp_path)
    bboxes = BoundingBoxFactory(json_path).bboxes
    i = last_checked_image
    try:
        while i < len(bboxes):
            bbox = bboxes[i]
            if bbox.img_name:
                img_path = os.path.join(frames_root_path, bbox.img_name)
            else:
                raise Exception("Missing info (img_name) for bbox object!")
            
            img = load_png_as_tensor(img_path)
            net_bbox, _ = net.eval(img, threshold=0)

            if net_bbox is None or\
                abs(net_bbox.x1 - bbox.x1) > 10 or abs(net_bbox.y1 - bbox.y1) > 10 or\
                abs(net_bbox.x2 - bbox.x2) > 10 or abs(net_bbox.y2 - bbox.y2) > 10:
                print(f"\n\nIteration: {i}/{len(bboxes)}")
                print(f"Image: {bbox.img_name}")
                frame = img
                if net_bbox:
                    frame = add_bbox_to_image(img, net_bbox, (1, 0, 0))
                frame = add_bbox_to_image(frame, bbox, (0, 0, 0))
                save_image(frame, temp_frame_path)
                status = handle_user_input(has_option_r=(net_bbox is not None))
                if status == 0:
                    bboxes.pop(i)
                    os.remove(img_path)
                    i -= 1
                elif status == 1:
                    temp = bboxes[i]
                    bboxes[i] = net_bbox # type: ignore
                    bboxes[i].img_name = temp.img_name
                    bboxes[i].img_height = temp.img_height
                    bboxes[i].img_width = temp.img_width
                elif status == 3:
                    i -= 2
                elif status == 4:
                    print("Stopping...")
                    print(f"Iteration {i}")
                    break
            
            i += 1
    except Exception as e:
        print(f"Error at iteration: {i}")
        raise e
    finally:
        with open(json_path, 'w') as f:
            json.dump(fp=f, obj=[bbox.__dict__() for bbox in bboxes])
        shutil.rmtree(temp_path)

def generate_data_using_segmentation(num_samples: int,
                                     frames_root_path: str,
                                     json_path: str
    ) -> None:
    # If there are already data on that path do not overwrite, append
    bboxes: List[Bbox_dict_t] = []
    if os.path.exists(json_path):
        bboxes = [bbox.__dict__() for bbox in BoundingBoxFactory(json_path).bboxes]
    
    last_img_idx = 0
    for file in os.listdir(frames_root_path):
        file_path = os.path.join(frames_root_path, file)
        if os.path.isfile(file_path):
            file_idx = int(file.split('.')[0])
            last_img_idx = file_idx if last_img_idx < file_idx else last_img_idx
            
    client = airsim.MultirotorClient()
    leadingUAV = LeadingUAV(genmode=True)
    egoUAV = EgoUAV(genmode=True)

    # Configure the mesh
    egoUAV.simSetDetectionFilterRadius(radius_cm=10. * 100.)
    egoUAV.simAddDetectionFilterMeshName(mesh_name=LEADING_UAV_NAME)

    # Configure the segmentation camera:
    # https://microsoft.github.io/AirSim/image_apis/#segmentation
    client.simSetSegmentationObjectID(LEADING_UAV_NAME, 0, True)

    subfolder_path = os.path.join(frames_root_path, "painted")
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)

    cnt: int = 0
    while cnt < num_samples:
        img_name = str(last_img_idx+cnt).zfill(FILENAME_LEADING_ZEROS) + ".png"
        frame, _ = create_sample(egoUAV, leadingUAV)
        img_segme = egoUAV._getImage(img_type=airsim.ImageType.Segmentation) # type: ignore
        detections = egoUAV.simGetDetections()
        if len(detections) != 1:
            continue
        detection = detections[0]
        x_min, x_max = math.ceil(detection.box2D.min.x_val), math.ceil(detection.box2D.max.x_val)
        y_min, y_max = math.ceil(detection.box2D.min.y_val), math.ceil(detection.box2D.max.y_val)

        # Refine the bbox estimation returned by the mesh,
        # using the segmentation camera
        x = []
        y = []
        for h in range(y_min, y_max):
            for w in range(x_min, x_max):
                if np.linalg.norm(img_segme[:, h, w]) == 0:
                    x.append(w)
                    y.append(h)
        
        if len(x) < 8:
            continue

        bbox = BoundingBox(x1=min(x),
                           y1=min(y),
                           x2=max(x),
                           y2=max(y),
                           label=1,
                           img_name=img_name,
                           img_height=frame.shape[1],
                           img_width=frame.shape[2])
        

        bboxes.append(bbox.__dict__())
        save_image(frame, os.path.join(frames_root_path, img_name))
        save_image(add_bbox_to_image(frame, bbox), os.path.join(subfolder_path, img_name))
        cnt += 1
    
    egoUAV.simClearDetectionMeshNames(camera_name="0")
    leadingUAV.disable()
    egoUAV.disable()
    egoUAV.client.reset()
    with open(json_path, 'w') as f:
        json.dump(fp=f, obj=bboxes)
