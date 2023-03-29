# import time

# import airsim
# import torch

# from GlobalConfig import GlobalConfig as config
# from models.LeadingUAV import LeadingUAV
# from models.EgoUAV import EgoUAV

# # Make sure move_duration exceeds sleep_duration
# # otherwise in each iteration of the game loop the
# # leading vehicle will "run out of moves" before the next iteration
# assert(config.move_duration > config.sleep_const)

# def view_video_feed(egoUAV: EgoUAV, leadingUAV: LeadingUAV):
#     import matplotlib.pyplot as plt
#     plt.ion()
#     for _ in range(0, config.game_loop_steps):
#         leadingUAV.random_move()
#         # egoUAV._cheat_move(velocity_vec=velocity_vec)
#         egoUAV._cheat_move(position_vec=torch.tensor([*(leadingUAV.simGetObjectPose().position)]))
#         t_stop = time.time() + config.sleep_const
#         while time.time() < t_stop:
#             plt.imshow(egoUAV._getImage(view_mode=True))
#             plt.show()
#             plt.pause(0.2)
#     plt.ioff()

# # Create a client to communicate with the UE
# client = airsim.MultirotorClient()
# client.confirmConnection()
# # client.simRunConsoleCommand("stat fps")
# print(f"Vehicle List: {client.listVehicles()}\n")

# # Create the vehicles and perform the takeoff
# leadingUAV = LeadingUAV(client, "LeadingUAV", config.leadingUAV_seed)
# egoUAV = EgoUAV(client, "EgoUAV")
# egoUAV.lastAction.join()
# leadingUAV.lastAction.join() # Just to make sure

# view_video_feed(egoUAV, leadingUAV)

# # TODO: Wait for the lastActions to finish?
# # Reset the location of all Multirotors
# client.reset()
# # Do not forget to disable all Multirotors
# leadingUAV.disable()
# egoUAV.disable()

import torch
from torch.utils.data import DataLoader

from nets.FasterRCNN import FasterRCNN
from models.BoundingBox import BoundingBoxDataset



# Fast test
print(f"Allocated CUDA memory before network initialization: {torch.cuda.memory_allocated(0)}")
frcnn = FasterRCNN(root_train_dir="data/empty_map/train/", json_train_labels="data/empty_map/train/empty_map.json",
                   root_test_dir="data/empty_map/test/", json_test_labels="data/empty_map/test/empty_map.json")
print(f"Allocated CUDA memory right after network initialization: {torch.cuda.memory_allocated(0)}")

frcnn.load("./nets/trained/mdl")

dataset = BoundingBoxDataset(
            root_dir="data/empty_map/train/", 
            json_file="data/empty_map/train/empty_map.json"
          )
dataloader = DataLoader(
                dataset, batch_size=4, shuffle=True, 
                collate_fn=frcnn._collate_fn
            )

images, targets = next(iter(dataloader))
dev_images = [image.to(torch.device("cuda")) for image in images]
frcnn.model.eval()
dev_preds = frcnn.model(dev_images)
pred = []
for dev_pred in dev_preds:
  temp = {}
  temp["boxes"] = []
  temp["boxes"].append(dev_pred["boxes"].to(torch.device("cpu")))
  pred.append(temp)
frcnn._show_bounding_boxes_batch(images, pred)
print(f"Allocated CUDA memory before deleting evaluation data: {torch.cuda.memory_allocated(0)}")
del dev_preds
del dev_images
print(f"Allocated CUDA memory after deleting evaluation data: {torch.cuda.memory_allocated(0)}")

del frcnn.model
print(f"Allocated CUDA memory after deleting frcnn.model: {torch.cuda.memory_allocated(0)}")
del frcnn
print(f"Allocated CUDA memory after deleting frcnn: {torch.cuda.memory_allocated(0)}")
