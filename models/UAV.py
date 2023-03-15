import airsim

class UAV():
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
     