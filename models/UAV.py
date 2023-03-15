import airsim

class UAV():
    def __init__(self, name) -> None:
         self.name = name
         self.client = self.get_client(name)

    # def __del__(self):
    #     self.client.armDisarm(False, vehicle_name=self.name)
    #     self.client.reset()
    #     self.client.enableApiControl(False, vehicle_name=self.name)

    def get_client(self, name):
         client = airsim.MultirotorClient()
         client.confirmConnection()
         client.enableApiControl(True, vehicle_name=name)
         client.armDisarm(True, vehicle_name=name)
         return client
    