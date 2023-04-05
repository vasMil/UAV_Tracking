import time

import airsim
import torch
from torchvision.utils import save_image

from GlobalConfig import GlobalConfig as config
from models.LeadingUAV import LeadingUAV
from models.EgoUAV import EgoUAV
from nets.FasterRCNN import FasterRCNN

# Make sure move_duration exceeds sleep_duration
# otherwise in each iteration of the game loop the
# leading vehicle will "run out of moves" before the next iteration
assert(config.move_duration > config.sleep_const)

def view_video_feed(egoUAV: EgoUAV, leadingUAV: LeadingUAV):
    import matplotlib.pyplot as plt
    plt.ion()
    for _ in range(0, config.game_loop_steps):
        leadingUAV.random_move()
        # egoUAV._cheat_move(velocity_vec=velocity_vec)
        egoUAV._cheat_move(position_vec=torch.tensor([*(leadingUAV.simGetObjectPose().position)]))
        t_stop = time.time() + config.sleep_const
        while time.time() < t_stop:
            plt.imshow(egoUAV._getImage(view_mode=True))
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

# Move the leadingUAV to a random position - within egoUAV's FOV
leadingUAV.sim_move_within_FOV(egoUAV, True)

# Pause the simulation
client.simPause(True)

# Take an image and use the trained model to predict the bounding box
img = egoUAV._getImage()
rcnn = FasterRCNN("data/empty_map/train/", "data/empty_map/train/empty_map.json",
                  "data/empty_map/test/", "data/empty_map/test/empty_map.json")
rcnn.load("nets/trained/faster_rcnn_state_dict_epoch50")
bbox = rcnn.eval(img)

# Calculate the distances on each axis, using the focal length
x_offset = 46 * 3.5 / bbox.width
print(x_offset)

# Continue the simulation
client.simPause(False)

# Move the egoUAV to that position and hope the two UAVs collide


# TODO: Wait for the lastActions to finish?
# Reset the location of all Multirotors
client.reset()
# Do not forget to disable all Multirotors
leadingUAV.disable()
egoUAV.disable()
