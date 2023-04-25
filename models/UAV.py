import airsim
from msgpackrpc.future import Future

from GlobalConfig import GlobalConfig as config

class UAV():
    """
    The base class for all UAV instances in the simulation:
    - LeadingUAV
    - EgoUAV
    """

    def __init__(self, name: str, port: int) -> None:
        self.name = name
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
        self.lastAction = self.client.takeoffAsync(vehicle_name=name)

    def disable(self) -> None:
        self.client.armDisarm(False, vehicle_name=self.name)
        self.client.enableApiControl(False, vehicle_name=self.name)
     
    def enable(self) -> None:
        self.client.enableApiControl(True, vehicle_name=self.name)
        self.client.armDisarm(True, vehicle_name=self.name)
     
    def moveToPositionAsync(self, x, y, z,
                            velocity=config.uav_velocity,
                            drivetrain: int = airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode: airsim.YawMode = airsim.YawMode()
                        ) -> Future:
        """
        Reminder: The airsim API uses the world frame
        ((0,0,0) is the location where the drone spawned)!
        (source: https://github.com/microsoft/AirSim/issues/4413)
        """
        # self.lastAction = self.client.moveToPositionAsync(x, y, z, velocity=velocity, vehicle_name=self.name, yaw_mode=yaw_mode)
        self.lastAction = self.client.moveToPositionAsync(x, y, z,
                                                          velocity=velocity,
                                                          drivetrain=drivetrain,
                                                          yaw_mode=yaw_mode,
                                                          vehicle_name=self.name)
        return self.lastAction
    
    def moveByVelocityAsync(self, vx, vy, vz,
                            duration,
                            drivetrain: int = airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode: airsim.YawMode = airsim.YawMode()
                        ) -> Future:
        self.lastAction = self.client.moveByVelocityAsync(vx, vy, vz,
                                                          duration,
                                                          yaw_mode=yaw_mode,
                                                          drivetrain=drivetrain,
                                                          vehicle_name=self.name)
        return self.lastAction

    def getMultirotorState(self) -> airsim.MultirotorState:
        return self.client.getMultirotorState(vehicle_name=self.name)

    def simGetObjectPose(self) -> airsim.Pose:
        """ Returns the Pose of the current vehicle, position is in world coordinates """
        return self.client.simGetObjectPose(object_name=self.name)
    
    def simGetGroundTruthKinematics(self) -> airsim.KinematicsState:
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.name)
    
    def simGetGroundTruthEnvironment(self) -> airsim.EnvironmentState:
        return self.client.simGetGroundTruthEnvironment(vehicle_name=self.name)

    def simGetCollisionInfo(self) -> airsim.CollisionInfo:
        return self.client.simGetCollisionInfo(vehicle_name=self.name)
    
    def hasCollided(self) -> bool:
        if self.simGetCollisionInfo().time_stamp > self.last_collision_time_stamp:
            self.last_collision_time_stamp = self.simGetCollisionInfo().time_stamp
            return True
        return False
