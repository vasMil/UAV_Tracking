import time

import airsim

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV

# Make sure move_duration exceeds sleep_duration
# otherwise in each iteration of the game loop the
# leading vehicle will "run out of moves" before the next iteration
assert(config.move_duration > config.sleep_const)

def view_video_feed(egoUAV: EgoUAV, leadingUAV: LeadingUAV):
    import matplotlib.pyplot as plt
    plt.ion()
    for _ in range(0, config.game_loop_steps):
        leadingUAV.random_move()
        t_stop = time.time() + config.sleep_const
        while time.time() < t_stop:
            plt.imshow(egoUAV._getImage())
            plt.show()
            plt.pause(0.2)
    plt.ioff()

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
for _ in range(0, config.game_loop_steps):
    leadingUAV.random_move()
    time.sleep(config.sleep_const)

# Reset the location of all Multirotors
client.reset()
# Do not forget to disable all Multirotors
leadingUAV.disable()
egoUAV.disable()
