import math
from typing import Optional, Literal, Tuple

import airsim
from msgpackrpc.future import Future
import numpy as np
import torch
from torchvision import transforms as T

from models.UAV import UAV
from project_types import Filter_t, Motion_model_t
from constants import EGO_UAV_NAME, PORT,\
    FOCAL_LENGTH_X, FOCAL_LENGTH_Y,\
    PAWN_SIZE_X, PAWN_SIZE_Y, PAWN_SIZE_Z,\
    IMG_HEIGHT, IMG_WIDTH,\
    HORIZ_FOV, VERT_FOV,\
    CAMERA_OFFSET_X,\
    SCORE_THRESHOLD
from models.BoundingBox import BoundingBox
from nets.DetectionNets import Detection_FasterRCNN
from nets.DetectionNets import Detection_SSD
from controller.Controller import Controller
from controller.KalmanFilter import KalmanFilter
from models.FrameInfo import EstimatedFrameInfo

class EgoUAV(UAV):
    def __init__(self,
                 name: str = EGO_UAV_NAME,
                 inference_freq_Hz: int = 1,
                 vel_magn: float = 0,
                 filter_type: Filter_t = "KF",
                 motion_model: Motion_model_t = "CA",
                 use_pepper_filter: bool = True,
                 weight_vel: Tuple[float, float, float] = (1, 1, 1,),
                 port: int = PORT,
                 genmode: bool = False
            ) -> None:
        super().__init__(name, vel_magn, port, genmode=genmode)
        # Initialize the NN
        if genmode:
            return
        # self.net = Detection_FasterRCNN()
        # self.net.load("nets/checkpoints/rcnn100.checkpoint")
        self.net = Detection_SSD()
        self.net.load("nets/checkpoints/ssd/ssd250.checkpoint")
        self.controller = Controller(vel_magn=vel_magn,
                                     dt=(1/inference_freq_Hz),
                                     weight_vel=weight_vel,
                                     filter_type=filter_type,
                                     motion_model=motion_model,
                                     use_pepper_filter=use_pepper_filter)

    def _getImage(self,
                  view_mode: bool = False,
                  img_type: airsim.ImageType = airsim.ImageType.Scene # type: ignore
        ) -> torch.Tensor:
        """
        Returns an RGB image as a tensor, of the EgoUAV.

        If view_mode is False:
        - The tensor returned will be of type float and in shape CxHxW
        Else:
        - The tensor returned will be of type uint8 and in shape HxWxC
        """
        # Respond is of type list[airsim.ImageResponse],
        # since the request is only for a single image,
        # I may extract only the first element of the list
        resp = self.client.simGetImages(
                [airsim.ImageRequest(0, img_type, False, False)],
                vehicle_name=self.name
            )[0]
        # Convert the string to a 1D numpy array (dtype = uint8)
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        # Reshape it into the proper tensor dimensions
        img = np.array(np.reshape(img, [resp.height, resp.width, 3]))
        # Convert the numpy array to a pytorch tensor
        if not view_mode:
            # Convert PIL image to a tensor, since it is not for viewing
            img = T.ToTensor()(img)
        else:
            img = torch.from_numpy(img)
        return img

    def _get_y_distance(self, x_distance: float, bbox: BoundingBox, mode: Literal["focal", "mpp", "fix_focal", "fix_mpp"]) -> float:
        """
        This function will calculate the distance of the object, on the y axis.
        We introduce this function, since we have not yet decided, which is the best method
        to use in order to derive the y axis distance of the object.
        The best performing mode is: "mpp".

        The modes:
        - focal, it is proposed at https://github.com/microsoft/AirSim/issues/1907.
                 it uses the focal length of the camera in order to retrieve the distance

        - mpp, stands for meters per pixel. I think it is logical to consider that if we devide the
               distance the image captures along the horizontal axis, in meters, with the width in pixels
               of the image, the result would be a number with units m/px. This may then be multiplied with
               the distance between the center of the image and the bbox's center, in pixels.
               This way we "map" the pixel distance to meters.

        - fix_focal, performs a fix to the center of the bounding box and the utilizes the focal_length
                     formula. I think this fix should be useful, when the object is not perpendicular to
                     the image, the height of the bounding box, which is used to find the y distance is
                     larger than what it should be.

        - fix_mpp, performs the same fix as above and uses mpp to derive the y distance.
        """
        if mode.split('_')[0] == "fix":
            mode2 = mode.split('_')[1]
            # Perform a small fix:
            # This will help, when the UAV is not perpendicular
            # to the image, thus the width of the bounding box is larger than the expected one.
            # That is because more of the front of the UAV (3D object) is visible.
            # If the leadingUAV is on the left of the egoUAV, then the UAV's back is located at the left
            # side of the bounding box. Since we may calculate the expected width of the bounding box, for
            # a given offset.x_val, we can estimate the center of the bounding box, if the UAV was
            # perpendicular to the image, by offseting the x1 coordinate of the predicted bounding box
            # by expected_width/2.
            # The same logic may be applied when the leadingUAV is on the right of egoUAV.
            expected_bbox_width_px = FOCAL_LENGTH_X * PAWN_SIZE_X / x_distance

            if bbox.x2 < IMG_WIDTH/2: # The leadingUAV is at the left of the egoUAV
                bbox_fixed_center_y = bbox.x1 + expected_bbox_width_px/2
            elif bbox.x1 > IMG_WIDTH/2:
                bbox_fixed_center_y = bbox.x2 - expected_bbox_width_px/2
            else:
                bbox_fixed_center_y = bbox.x_center

            y_box_displacement = bbox_fixed_center_y - IMG_WIDTH/2
        else:
            mode2 = mode
            y_box_displacement = bbox.x_center - IMG_WIDTH/2

        if mode2 == "focal":
            return y_box_displacement * x_distance / FOCAL_LENGTH_X

        if mode2 == "mpp":
            img_width_meters = 2 * x_distance / math.tan((math.pi - HORIZ_FOV)/2)
            return y_box_displacement * (img_width_meters / IMG_WIDTH)

        raise ValueError("_get_z_distance: Invalid mode!")

    def _get_z_distance(self, x_distance: float, bbox: BoundingBox, mode: Literal["focal", "mpp", "fix_focal", "fix_mpp"]) -> float:
        """
        This function will calculate the distance of the object, on the z axis.
        We introduce this function, since we have not yet decided, which is the best method
        to use in order to derive the z axis distance of the object.
        The best performing mode is: "focal_length".

        The modes:
        - focal, it is proposed at https://github.com/microsoft/AirSim/issues/1907.
                        it uses the focal length of the camera in order to retrieve the distance

        - mpp, stands for meters per pixel. I think it is logical to consider that if we devide the
            distance the image captures along the vertical axis, in meters, with the height in pixels
            of the image, the result would be a number with units m/px. This may then be multiplied with
            the distance between the center of the image and the bbox's center, in pixels.
            This way we "map" the pixel distance to meters.

        - fix_focal, performs a fix to the center of the bounding box and the utilizes the focal_length
                    formula. I think this fix should be useful, when the object is not perpendicular to
                    the image, the height of the bounding box, which is used to find the z distance is
                    larger than what it should be.

        - fix_mpp, performs the same fix as above and uses mpp to derive the z distance.
        """
        if mode.split('_')[0] == "fix":
            mode2 = mode.split('_')[1]
            # Perform a small fix: (This seems to have a negative effect on the error)
            # This will help, when the UAV is not perpendicular
            # to the image, thus the height of the bounding box is larger than the expected one.
            # That is because more of the front of the UAV (3D object) is visible.
            # If the egoUAV is lower than the leadingUAV, then the UAV's back is located at the top of
            # the bounding box. Since we may calculate the expected height of the bounding box, for
            # a given offset.x_val, we can estimate the center of the bounding box, if the UAV was
            # perpendicular to the image, by offseting the y1 coordinate of the predicted bounding box
            # by expected_height/2.
            # The same logic may be applied to cases aswell.
            expected_bbox_height_px = FOCAL_LENGTH_Y* PAWN_SIZE_Z / x_distance

            if bbox.y2 < IMG_HEIGHT/2: # The leadingUAV is higher than the egoUAV
                bbox_fixed_center_z = bbox.y1 + expected_bbox_height_px/2
            elif bbox.y1 > IMG_HEIGHT/2:
                bbox_fixed_center_z = bbox.y2 - expected_bbox_height_px/2
            else:
                bbox_fixed_center_z = bbox.y_center

            z_box_displacement = bbox_fixed_center_z - IMG_HEIGHT/2
        else:
            mode2 = mode
            z_box_displacement = bbox.y_center - IMG_HEIGHT/2

        if mode2 == "focal":
            return z_box_displacement * x_distance / FOCAL_LENGTH_Y

        if mode2 == "mpp":
            img_height_meters = 2 * x_distance / math.tan((math.pi - VERT_FOV)/2)
            return z_box_displacement * (img_height_meters / IMG_HEIGHT)

        raise ValueError("_get_z_distance: Invalid mode!")

    def get_distance_from_bbox(self, bbox: Optional[BoundingBox]) -> Optional[np.ndarray]:
        """
        Args:
        bbox: The bbox predicted using a NN, on EgoUAV's image, or None.

        Returns:
        A (3x1) column vector containing the estimated distance on the (x, y, z) axis,
        between the EgoUAV and the target.
        If the bbox is None, all values will be zero.
        """
        if not bbox:
            return None
        dist = np.zeros([3,1])
        x_offset = FOCAL_LENGTH_X * PAWN_SIZE_Y / bbox.width
        dist[0] = x_offset
        dist[1] = self._get_y_distance(x_offset, bbox, "focal")
        dist[2] = self._get_z_distance(x_offset, bbox, "focal")

        # The distance on the x axis so far, is from EgoUAV's camera,
        # to the back side of the LeadingUAV. We require this distance to be
        # from one center to the other.
        dist[0] += (CAMERA_OFFSET_X + PAWN_SIZE_X/2)
        return dist

    def moveToBoundingBoxAsync(self,
                               bbox: Optional[BoundingBox],
                               orient: Tuple[float, float, float],
                               dt: float
                            ) -> Tuple[Future, EstimatedFrameInfo]:
        """
        Given a BoundingBox object, calculate its relative distance
        (offset) on the x axis, using the focal length.
        Then using some trigonomerty determine the offset on the
        y and z axis.
        Lastly, use a Controller to decide upon the target velocity of
        the EgoUAV.

        Args:
        - bbox: The BoundingBox object to move towards | or None
        (If None and there is a Kalman filter present target's position
        will be estimated, else EgoUAV will preserve it's velocity and direction)
        - orient: The orientation of the EgoUAV, when the image frame was captured.
        (This allows us to get the offset back to the original axis, from the camera's
        axis)
        - dt: The amount of (simulation) time that has passed from previous advance 
        on the controller.

        Returns:
        A Tuple with:
        - The Future returned by AirSim
        - An EstimatedFrameInfo object, that contains estimations of target's
        velocity and target's position etc. (If available)
        """
        if (not bbox) or (bbox.score and bbox.score < SCORE_THRESHOLD):
            offset = None
        else:
            if bbox.score is None:
                Warning("Score in BoundingBox object is None, any score_threashold will be ignored!")
            offset = self.get_distance_from_bbox(bbox)

        pitch_roll_yaw_deg = np.array(orient)
        current_pos = self.getMultirotorState().kinematics_estimated.position
        current_pos = np.expand_dims(current_pos.to_numpy_array(), axis=1)
        velocity, yaw_deg, est_frame_info = self.controller.step(offset=offset,
                                                                 pitch_roll_yaw_deg=pitch_roll_yaw_deg,
                                                                 ego_pos=current_pos)
        self.lastAction = self.moveByVelocityAsync(*(velocity.squeeze()),
                                                   duration=dt,
                                                   yaw_mode=airsim.YawMode(False, yaw_deg)
        )
        est_frame_info["still_tracking"] = (bbox is not None)
        return self.lastAction, est_frame_info

    def advanceUsingFilter(self, dt: float) -> Tuple[Future, EstimatedFrameInfo]:
        """
        Use this function to predict the position of the target.
        It advances the controller without requiring any kind of measurement.

        If the controller utilizes a Kalman Filter, this will be a prediction,
        otherwise (i.e. there is no filter), the EgoUAV will continue with the same
        yaw_angle and the same velocity.

        Args:
        - dt: The amount of (simulation) time that has passed from previous advance 
        on the controller.

        Returns:
        A Tuple with:
        - The Future returned by AirSim
        - An EstimatedFrameInfo object, that contains estimations of target's
        velocity and target's position etc. (If available)
        """
        if not isinstance(self.controller.filter, KalmanFilter):
            raise Exception(f"There is no KF to support advanceUsingFilter: self.controller.filter is of type {type(self.controller.filter)}")
        current_pos = np.expand_dims(self.getMultirotorState()
                                     .kinematics_estimated
                                     .position
                                     .to_numpy_array(),
                                     axis=1
        )

        velocity, yaw_deg, est_frame_info = self.controller.step(offset=None,
                                                                 pitch_roll_yaw_deg=np.array([0, 0, 0]),
                                                                 ego_pos=current_pos
                                                            )

        self.lastAction = self.moveByVelocityAsync(*(velocity.squeeze()),
                                                   duration=dt,
                                                   yaw_mode=airsim.YawMode(False, yaw_deg)
        )
        
        return self.lastAction, est_frame_info
