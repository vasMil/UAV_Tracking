import airsim
from .UAV import UAV

class LeadingUAV(UAV):
    def __init__(self, client: airsim.MultirotorClient, name: str) -> None:
        super().__init__(client, name)
        # airsim.wait_key('Press any key to takeoff')
        client.takeoffAsync(vehicle_name=name)
