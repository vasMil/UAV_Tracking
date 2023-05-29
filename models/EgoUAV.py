import math
from typing import Optional, Literal, Tuple

import airsim
from msgpackrpc.future import Future
import numpy as np
import torch
from torchvision import transforms as T

from models.UAV import UAV
from GlobalConfig import GlobalConfig as config
from models.BoundingBox import BoundingBox
from nets.DetectionNets import Detection_FasterRCNN
from nets.DetectionNets import Detection_SSD
from controller.Controller import Controller
from controller.KalmanFilter import KalmanFilter

class EgoUAV(UAV):
    def __init__(self,
                 name: str,
                 filter: Literal["None", "KF"] = "None",
                 port: int = 41451,
                 genmode: bool = False
            ) -> None:
        super().__init__(name, port, genmode=genmode)
        # Initialize the NN
        if genmode:
            return
        # self.net = Detection_FasterRCNN()
        # self.net.load("nets/checkpoints/rcnn100.checkpoint")
        self.net = Detection_SSD()
        self.net.load("nets/checkpoints/ssd300.checkpoint")
        self.controller = Controller(self)
        if filter == "None":
            self.filter = None
        elif filter == "KF":
            X_init = np.zeros([6, 1]); X_init[0][0] = 3.5
            P_init = np.zeros([6, 6]); P_init[3][3] = 4; P_init[4][4] = 5; P_init[5][5] = 3
            R = np.array([[7.9882659, 3.17199785, 1.58456132],
                          [3.17199785, 15.04112204, 0.14100749],
                          [1.58456132, 0.14100749, 3.98863264]]
            )
            self.filter = KalmanFilter(X_init, P_init, np.zeros([6, 6]), R)

    def _getImage(self, view_mode: bool = False) -> torch.Tensor:
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
                [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)],
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

    def _cheat_move(
            self,
            position_vec: Optional[torch.Tensor] = None,
            velocity_vec: Optional[torch.Tensor] = None
        ) -> Future:
        """
        This function is designed to help me determine what information to use as ground truth, when
        training the neural network.
        You should specify exactly one of the two arguments!
        - position_vec: if specified, should contain the latest position of the leadingUAV
        - velocity_vec: if specified, should contain the velocities used by moveByVelocity() method on the last call for the leadingUAV
        """
        if ((position_vec is None and velocity_vec is None) or \
             (position_vec is not None and velocity_vec is not None)):
            raise Exception("EgoUAV::_cheat_move: Exactly one of two arguments should not be None")

        if velocity_vec is not None:
            self.lastAction = self.moveByVelocityAsync(*(velocity_vec.tolist()), duration=config.move_duration)
        elif position_vec is not None:
            self.lastAction = self.moveToPositionAsync(*(position_vec.tolist()))

        return self.lastAction

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
            expected_bbox_width_px = config.focal_length_x * config.pawn_size_x / x_distance

            if bbox.x2 < config.img_width/2: # The leadingUAV is at the left of the egoUAV
                bbox_fixed_center_y = bbox.x1 + expected_bbox_width_px/2
            elif bbox.x1 > config.img_width/2:
                bbox_fixed_center_y = bbox.x2 - expected_bbox_width_px/2
            else:
                bbox_fixed_center_y = bbox.x_center

            y_box_displacement = bbox_fixed_center_y - config.img_width/2
        else:
            mode2 = mode
            y_box_displacement = bbox.x_center - config.img_width/2

        if mode2 == "focal":
            return y_box_displacement * x_distance / config.focal_length_x

        if mode2 == "mpp":
            img_width_meters = 2 * x_distance / math.tan((math.pi - config.horiz_fov)/2)
            return y_box_displacement * (img_width_meters / config.img_width)

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
            expected_bbox_height_px = config.focal_length_y * config.pawn_size_z / x_distance

            if bbox.y2 < config.img_height/2: # The leadingUAV is higher than the egoUAV
                bbox_fixed_center_z = bbox.y1 + expected_bbox_height_px/2
            elif bbox.y1 > config.img_height/2:
                bbox_fixed_center_z = bbox.y2 - expected_bbox_height_px/2
            else:
                bbox_fixed_center_z = bbox.y_center

            z_box_displacement = bbox_fixed_center_z - config.img_height/2
        else:
            mode2 = mode
            z_box_displacement = bbox.y_center - config.img_height/2

        if mode2 == "focal":
            return z_box_displacement * x_distance / config.focal_length_y

        if mode2 == "mpp":
            img_height_meters = 2 * x_distance / math.tan((math.pi - config.vert_fov)/2)
            return z_box_displacement * (img_height_meters / config.img_height)

        raise ValueError("_get_z_distance: Invalid mode!")

    def _get_yaw_angle(self, dist_x: float, dist_y: float) -> float:
        """
        Returns the angle at which the point given by dist_x and dist_y is located.
        The 2D space is defined by the plane found in the 3D space of the UAVs, at z=z0.
        Where z0 can be thought of as the position on the z axis of the EgoUAV.
        (In reality it may be any plane, since we just project the 3D points to the 2D space, by just ignoring
        the value on the z axis).
        It is also important to note that the other two axis are identical with EgoUAV's coordinate frame.
        This coordinate frame is defined by the orientation of the EgoUAV, when spawned in the map.
        On the other hand, the point provided by dist_x and dist_y is on the coordinate frame defined by EgoUAV's
        camera orientation, during the simulation.

        Args:
        dist_x: The distance of the point on the x axis (the x axis is defined in EgoUAV's camera coordinate frame)
        dist_y: The distance of the point on the y axis (the y axis is defined in EgoUAV's camera coordinate frame)

        Returns:
        The yaw angle for the EgoUAV, in the EgoUAV's coordinate frame.

        Note: We introduce a bit of confusion by using two different coordinate frames in this function.
        We could just provide dist_x and dist_y in EgoUAV's coordinate frame aswell and ignore the fact that
        during simulation EgoUAV's camera defines a different coordinate system. This way the caller of the
        function would only have to project the point found in the image (and is currently in the camera's
        coordinate frame) onto EgoUAV's coordinate frame and thus we would have less things to worry about.
        In this scenario, the output of this function would only be math.degrees(math.atan(dist_y/dist_x)).
        The problem with this approach is that if the EgoUAV is already at x degrees and we require it to rotate,
        at some y, the UAV may decide to rotate left by 360 + y degrees, instead of rotating y degrees to the
        opposite direction.
        This is fixed in our approach.
        """
        return math.degrees(math.atan(dist_y/dist_x)) + self.getPitchRollYaw()[2]

    def get_yaw_angle_from_bbox(self, bbox: Optional[BoundingBox]) -> float:
        """
        Wraps self._get_yaw_angle() into a function that will also derive the distances
        required by this method in order to calculate the yaw angle.
        Thus it may be used by higher level functions.

        Args:
        bbox: The bounding box, whose center we are going to use as the point for
        which we will derive the distance.

        Returns:
        The yaw angle for the EgoUAV, in it's coordinate frame.
        Note that this might be different than the coordinate frame defined by the
        orientation of the camera.
        """
        if not bbox:
            return self.getPitchRollYaw()[2]
        dist_x = config.focal_length_x * config.pawn_size_y / bbox.width
        dist_y = self._get_y_distance(dist_x, bbox, "focal")
        return self._get_yaw_angle(dist_x, dist_y)

    def get_distance_from_bbox(self, bbox: Optional[BoundingBox]) -> np.ndarray:
        """
        Args:
        bbox: The bbox predicted using a NN, on EgoUAV's image, or None.

        Returns:
        A (3x1) column vector containing the estimated distance on the (x, y, z) axis,
        between the EgoUAV and the target.
        If the bbox is None, all values will be zero.
        """
        dist = np.zeros([3,1])
        if not bbox:
            return dist
        x_offset = config.focal_length_x * config.pawn_size_y / bbox.width
        dist[0] = x_offset
        dist[1] = self._get_y_distance(x_offset, bbox, "focal")
        dist[2] = self._get_z_distance(x_offset, bbox, "focal")

        # The distance on the x axis so far, is from EgoUAV's camera,
        # to the back side of the LeadingUAV. We require this distance to be
        # from one center to the other.
        dist[0] -= (config.camera_offset_x + config.pawn_size_x/2)
        return dist

    def moveToBoundingBoxAsync(self,
                               bbox: Optional[BoundingBox],
                               dt: float
                            ) -> Future:
        """
        Given a BoundingBox object, calculate its relative distance
        (offset) on the x axis, using the focal length.
        Then using some trigonomerty determine the offset on the
        y and z axis.
        Lastly, add to your current coordinates this calculated offset
        and move towards that object, using moveToPositionAsync().

        Args:
        - bbox: The BoundingBox object to move towards | or None
        - time_interval: 1/(nets inference frequency)

        Returns:
        - The Future returned by AirSim

        (If time_interval is not 0 -> the egoUAV will first calculate
        it's offset from the bbox, normalize it (thus preserving it's direction)
        and multiply it by the desired velocity (i.e. config.uav_velocity).
        Converting a distance vector to a velocity vector of a specific
        magnitude (config.uav_velocity) is an overconstrained problem.
        Consider a bbox that is further than the maximum distance EgoUAV
        can travel in time_interval seconds.
        Else the offset will be directly used as arguments
        in moveToPositionAsync)

        Note: moveToPositionAsync seems to work the best, the attempt was to
        make the EgoUAV movement smooth.
        """
        distance = self.get_distance_from_bbox(bbox)
        yaw_deg = self.get_yaw_angle_from_bbox(bbox)

        # Use a filter to predict LeadingUAV's movement, and/or smooth the
        # measurements that have both process and measurement noise.
        measurement = np.pad(distance, ((0, 3),(0, 0)))
        if self.filter:
            ego_estimated_pos = np.expand_dims(self.getMultirotorState().kinematics_estimated.position.to_numpy_array(), axis=1)
            measurement[0:3, :] += ego_estimated_pos
            measurement = self.filter.step(measurement, dt)
            measurement[0:3, :] -= ego_estimated_pos

        # Use the controller to extract the target velocity, that
        # will help us reach the estimated point, on which the LeadingUAV
        # is at.
        velocity = self.controller.step(measurement[0:3,:], dt=dt)
        self.lastAction = self.moveFrontFirstByVelocityAsync(*(velocity.squeeze()), duration=dt, yaw_deg=yaw_deg)
        return self.lastAction
