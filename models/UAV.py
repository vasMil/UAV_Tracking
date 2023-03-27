import airsim
from msgpackrpc.future import Future

from GlobalConfig import GlobalConfig as config

class UAV():
    """
    The base class for all UAV instances in the simulation:
    - LeadingUAV
    - EgoUAV
    """
    def __init__(self, client: airsim.MultirotorClient, name: str) -> None:
        self.name = name
        self.client = client
        # Should not allow the UAV to go below a certain height, since it may collide with the ground.
        # In this case we won't allow it to go lower than the position at which it is placed after takeoff.
        self.min_z = client.simGetObjectPose(object_name=self.name).position.z_val
        self.last_collision_time_stamp = client.simGetCollisionInfo(vehicle_name=name).time_stamp
        self.enable()
        # Perform takeoff
        self.lastAction = client.takeoffAsync(vehicle_name=name)

    def disable(self) -> None:
        self.client.armDisarm(False, vehicle_name=self.name)
        self.client.enableApiControl(False, vehicle_name=self.name)
     
    def enable(self) -> None:
        self.client.enableApiControl(True, vehicle_name=self.name)
        self.client.armDisarm(True, vehicle_name=self.name)
     
    def moveToPositionAsync(self, x, y, z, velocity=config.leading_velocity) -> Future:
        """
        Reminder: The airsim API uses the world frame
        ((0,0,0) is the location where the drone spawned)!
        (source: https://github.com/microsoft/AirSim/issues/4413)
        """
        self.lastAction = self.client.moveToPositionAsync(x, y, z, velocity=velocity, vehicle_name=self.name)
        return self.lastAction
    
    def moveByVelocityAsync(self, vx, vy, vz, duration) -> Future:
        self.lastAction = self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name=self.name)
        return self.lastAction

    def simGetObjectPose(self) -> airsim.Pose:
        """ Returns the Pose of the current vehicle, position is in world coordinates """
        return self.client.simGetObjectPose(object_name=self.name)
    
    def simGetGroundTruthKinematics(self) -> airsim.KinematicsState:
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.name)
    
    def simSetKinematics(self, state: airsim.KinematicsState, ignore_collision: bool) -> bool:
        return self.client.simSetKinematics(state, ignore_collision, vehicle_name=self.name)
    
    def simGetGroundTruthEnvironment(self) -> airsim.EnvironmentState:
        return self.client.simGetGroundTruthEnvironment(vehicle_name=self.name)

    def simGetCollisionInfo(self) -> airsim.CollisionInfo:
        return self.client.simGetCollisionInfo(vehicle_name=self.name)
    
    def hasCollided(self) -> bool:
        if self.simGetCollisionInfo().time_stamp > self.last_collision_time_stamp:
            self.last_collision_time_stamp = self.simGetCollisionInfo().time_stamp
            return 1
        return 0
