import airsim
from .UAV import UAV

class LeadingUAV(UAV):
    def __init__(self, name) -> None:
        super().__init__(name)
        # airsim.wait_key('Press any key to takeoff')
        self.client.takeoffAsync(vehicle_name=name).join()
