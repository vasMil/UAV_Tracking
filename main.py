import time
import airsim
from models.LeadingUAV import LeadingUAV

client = airsim.MultirotorClient()
client.confirmConnection()
print(f"Vehicle List: {client.listVehicles()}\n")

leadingUAV = LeadingUAV(client, "LeadingUAV")
egoUAV = LeadingUAV(client, "EgoUAV")

print("SLEEP")
time.sleep(10)

#  CLEANUP
# Reset the location of all Multirotors
client.reset()
# Do not forget to disable all Multirotors
leadingUAV.disable()
egoUAV.disable()
