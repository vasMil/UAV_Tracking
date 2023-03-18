import time
import math

import airsim
import matplotlib.pyplot as plt

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV

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
egoUAV = EgoUAV(client, "EgoUAV")
egoUAV.lastAction.join()
leadingUAV.lastAction.join() # Just to make sure

# The game loop
plt.ion()
for _ in range(0, config.game_loop_steps):
    leadingUAV.random_move()
    t_stop = time.time() + config.sleep_const
    while time.time() < t_stop:
        egoUAV.view_video_feed()
        plt.pause(0.2)
plt.ioff()

# Reset the location of all Multirotors
client.reset()
# Do not forget to disable all Multirotors
leadingUAV.disable()
egoUAV.disable()
