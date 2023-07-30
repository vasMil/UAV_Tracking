from typing import Tuple, Optional, List
import math

import airsim
from msgpackrpc.future import Future

class UAV():
    """
    The base class for all UAV instances in the simulation:
    - LeadingUAV
    - EgoUAV
    """

    def __init__(self,
                 name: str,
                 vel_magn: float = 0.,
                 port: int = 41451,
                 genmode: bool = False) -> None:
        self.name = name
        self.vel_magn = vel_magn
        self.client = airsim.MultirotorClient(port=port)
        print(f"UAV {self.name} listens at port {port}")

        # Preserve the origin for the current object's coordinate system in global coordinates
        # this may later be used in order to find the offset between the coordinate systems
        # of different UAVs.
        self.sim_global_coord_frame_origin = self.client.simGetObjectPose(object_name=name).position

        # Should not allow the UAV to go below a certain height, since it may collide with the ground.
        # In this case we won't allow it to go lower than the position at which it is placed after takeoff.
        self.min_z = self.client.simGetObjectPose(object_name=self.name).position.z_val
        self.last_collision_time_stamp = self.client.simGetCollisionInfo(vehicle_name=name).time_stamp
        self.enable()
        
        # Perform takeoff
        if not genmode:
            self.lastAction = self.client.takeoffAsync(vehicle_name=name)

    def disable(self) -> None:
        """
        Disarms the vehicle and disables the ApiControl
        """
        self.client.armDisarm(False, vehicle_name=self.name)
        self.client.enableApiControl(False, vehicle_name=self.name)
     
    def enable(self) -> None:
        """
        Enables the ApiControl and arms vehicle 
        """
        self.client.enableApiControl(True, vehicle_name=self.name)
        self.client.armDisarm(True, vehicle_name=self.name)
     
    def moveToPositionAsync(self, x, y, z,
                            velocity: Optional[float] = None,
                            drivetrain: int = airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode: airsim.YawMode = airsim.YawMode()
                        ) -> Future:
        """
        Uses the AirSim API to move this UAV at the specified position (x, y, z),
        using a constant velocity.

        Reminder: The airsim API considers as the position of the origin point
        (i.e. (0,0,0)) is the location where the drone spawned!
        (source: https://github.com/microsoft/AirSim/issues/4413)
        """
        vel = self.vel_magn if velocity is None else velocity
        self.lastAction = self.client.moveToPositionAsync(x, y, z,
                                                          velocity=vel,
                                                          drivetrain=drivetrain,
                                                          yaw_mode=yaw_mode,
                                                          vehicle_name=self.name)
        return self.lastAction

    def moveByVelocityAsync(self, vx, vy, vz,
                            duration,
                            drivetrain: int = airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode: airsim.YawMode = airsim.YawMode()
                        ) -> Future:
        """
        Uses the AirSim API to move this UAV at a constant velocity (vx, vy, vz),
        for duration.

        Reminder: The airsim API considers as the position of the origin point
        (i.e. (0,0,0)) is the location where the drone spawned!
        (source: https://github.com/microsoft/AirSim/issues/4413)
        """
        self.lastAction = self.client.moveByVelocityAsync(vx, vy, vz,
                                                          duration,
                                                          yaw_mode=yaw_mode,
                                                          drivetrain=drivetrain,
                                                          vehicle_name=self.name)
        return self.lastAction

    def moveOnPathAsync(self,
                        path,
                        velocity: Optional[float] = None,
                        drivetrain: int = airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode = airsim.YawMode()
                    ) -> Future:
        vel = self.vel_magn if velocity is None else velocity
        self.lastAction = self.client.moveOnPathAsync(path,
                                                      velocity=vel,
                                                      drivetrain=drivetrain,
                                                      yaw_mode=yaw_mode,
                                                      vehicle_name=self.name
                                                )
        return self.lastAction

    def getMultirotorState(self) -> airsim.MultirotorState:
        return self.client.getMultirotorState(vehicle_name=self.name)

    def getPitchRollYaw(self) -> Tuple[float, float, float]:
        """
        Using getMultirotorState(), it extracts the orientation of this UAV,
        converts it to eularian angles (which are in radians) and returns the
        pitch, roll and yaw in degrees, since most AirSim API calls use degrees.

        Retruns:
        `Tuple[pitch, roll, yaw]` in degrees
        """
        pitch, roll, yaw = airsim.to_eularian_angles(self.getMultirotorState().kinematics_estimated.orientation)
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)
        yaw = math.degrees(yaw)
        return (pitch, roll, yaw,)

    def rotateToYawAsync(self, yaw) -> Future:
        """
        Rotates this UAV to a specific yaw angle.
        
        Args:
        - yaw: The angle in degrees.
        """
        self.lastAction = self.client.rotateToYawAsync(yaw=yaw, vehicle_name=self.name)
        return self.lastAction

    def simGetObjectPose(self) -> airsim.Pose:
        """ Returns the Pose of the current vehicle, position is in world coordinates """
        return self.client.simGetObjectPose(object_name=self.name)
    
    def simSetVehiclePose(self, pose: airsim.Pose, ignore_collision=True) -> None:
        """The position is on the UAV's local coordinate frame"""
        self.client.simSetVehiclePose(pose=pose, ignore_collision=ignore_collision, vehicle_name=self.name)

    def simGetGroundTruthKinematics(self) -> airsim.KinematicsState:
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.name)
    
    def simGetGroundTruthEnvironment(self) -> airsim.EnvironmentState:
        return self.client.simGetGroundTruthEnvironment(vehicle_name=self.name)

    def simGetCollisionInfo(self) -> airsim.CollisionInfo:
        return self.client.simGetCollisionInfo(vehicle_name=self.name)

    def simSetDetectionFilterRadius(self,
                                    camera_name: str = "0",
                                    image_type: airsim.ImageType = airsim.ImageType.Scene, # type: ignore
                                    radius_cm: float = 10. * 100.
                                    ) -> None:
        self.client.simSetDetectionFilterRadius(camera_name=camera_name,
                                                image_type=image_type,
                                                radius_cm=radius_cm,
                                                vehicle_name=self.name)

    def simAddDetectionFilterMeshName(self,
                                      mesh_name: str,
                                      camera_name: str = "0",
                                      image_type: airsim.ImageType = airsim.ImageType.Scene, # type: ignore
                                      ) -> None:
        self.client.simAddDetectionFilterMeshName(camera_name=camera_name,
                                                  image_type=image_type,
                                                  mesh_name=mesh_name,
                                                  vehicle_name=self.name) 

    def simClearDetectionMeshNames(self,
                                   camera_name: str,
                                   image_type: airsim.ImageType = airsim.ImageType.Scene, # type: ignore
        ) -> None:
        self.client.simClearDetectionMeshNames(camera_name=camera_name,
                                               image_type=image_type,
                                               vehicle_name=self.name)

    def simGetDetections(self,
                         camera_name: str = "0",
                         image_type: airsim.ImageType = airsim.ImageType.Scene # type: ignore
        ) -> List[airsim.DetectionInfo]:
        return self.client.simGetDetections(camera_name=camera_name,
                                            image_type=image_type,
                                            vehicle_name=self.name)

    def simTestLineOfSightToPoint(self, geo_point: airsim.GeoPoint):
            return self.client.simTestLineOfSightToPoint(geo_point, vehicle_name=self.name)

    def hasCollided(self) -> bool:
        if self.simGetCollisionInfo().time_stamp > self.last_collision_time_stamp:
            self.last_collision_time_stamp = self.simGetCollisionInfo().time_stamp
            return True
        return False
