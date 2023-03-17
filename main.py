import time

import airsim

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV

# Make sure move_duration exceeds sleep_duration
# otherwise in each iteration of the game loop the
# leading vehicle will "run out of moves" before the next iteration
assert(config.move_duration > config.sleep_const)

# Create a client to communicate with the UE
client = airsim.MultirotorClient()
client.confirmConnection()
# client.simRunConsoleCommand("stat fps")
print(f"Vehicle List: {client.listVehicles()}\n")

# Create the vehicles and perform the takeoff
leadingUAV = LeadingUAV(client, "LeadingUAV", config.leadingUAV_seed)
leadingUAV.lastAction.join()

# The game loop
for i in range(0, config.game_loop_steps):
    leadingUAV.random_move()
    time.sleep(config.sleep_const)

# Reset the location of all Multirotors
client.reset()
# Do not forget to disable all Multirotors
leadingUAV.disable()
