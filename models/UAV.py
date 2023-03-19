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
        self.enable()

    def disable(self):
        self.client.armDisarm(False, vehicle_name=self.name)
        self.client.enableApiControl(False, vehicle_name=self.name)
     
    def enable(self):
        self.client.enableApiControl(True, vehicle_name=self.name)
        self.client.armDisarm(True, vehicle_name=self.name)
     
    def moveToPositionAsync(self, x, y, z, velocity=config.leading_velocity) -> Future:
        """
        Reminder: The airsim API uses the world frame!
        (source: https://github.com/microsoft/AirSim/issues/4413)
        """
        return self.client.moveToPositionAsync(x, y, z, velocity=velocity, vehicle_name=self.name)
    
    def moveByVelocityAsync(self, vx, vy, vz, duration):
        return self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name=self.name)

    def simGetObjectPose(self) -> airsim.Pose:
        """ Returns the Pose of the current vehicle, position is in world coordinates """
        return self.client.simGetObjectPose(object_name=self.name)
    
    def simGetGroundTruthKinematics(self) -> airsim.KinematicsState:
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.name)
    
    def simSetKinematics(self, state: airsim.KinematicsState, ignore_collision: bool) -> bool:
        return self.client.simSetKinematics(state, ignore_collision, vehicle_name=self.name)
    
    def simGetGroundTruthEnvironment(self) -> airsim.EnvironmentState:
        return self.client.simGetGroundTruthEnvironment(vehicle_name=self.name)
