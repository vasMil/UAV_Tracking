# import time
# import multiprocessing as mp

# import airsim

# from GlobalConfig import GlobalConfig as config
# from models.LeadingUAV import LeadingUAV
# from models.EgoUAV import EgoUAV

# # Make sure move_duration exceeds sleep_duration
# # otherwise in each iteration of the game loop the
# # leading vehicle will "run out of moves" before the next iteration
# assert(config.move_duration > config.sleep_const)

# def leadingUAV_loop(exit_signal, port: int, time_interval: int):
#     leadingUAV = LeadingUAV("LeadingUAV", port, config.leadingUAV_seed)
#     leadingUAV.lastAction.join()
#     with exit_signal.get_lock():
#         exit_status = exit_signal.value # type: ignore
#     # time.sleep(60)
#     while not exit_status:
#         leadingUAV.random_move()
#         time.sleep(time_interval)
#         with exit_signal.get_lock():
#             exit_status = exit_signal.value # type: ignore
#     leadingUAV.disable()

# def egoUAV_loop(exit_signal, port: int):
#     """
#     Follows the leadingUAV, using it's NN, by predicting the bounding box
#     and then moving towards it.
#     """
#     egoUAV = EgoUAV("EgoUAV", port)
#     egoUAV.lastAction.join()
#     with exit_signal.get_lock():
#         exit_status = exit_signal.value # type: ignore
        
#     while not exit_status:
#         img = egoUAV._getImage()
#         bbox = egoUAV.rcnn.eval(img)
#         future = egoUAV.moveToBoundingBoxAsync(bbox)
#         print("UAV detected, moving towards it..." if future else "Lost tracking!!!")
#         with exit_signal.get_lock():
#             exit_status = exit_signal.value # type: ignore
#     egoUAV.disable()

# if __name__ == '__main__':
#     # Create a client to communicate with the UE
#     client = airsim.MultirotorClient()
#     client.confirmConnection()
#     print(f"Vehicle List: {client.listVehicles()}\n")
#     # Start recording
#     print("\n*****************")
#     print("Recording Started")
#     print("*****************\n")
#     client.startRecording()

#     # Communication variables
#     exit_signal = mp.Value('i', False)

#     # Create two processes
#     leadingUAV_process = mp.Process(target=leadingUAV_loop, args=(exit_signal, config.port, 2))
#     egoUAV_process = mp.Process(target=egoUAV_loop, args=(exit_signal, config.port))
#     leadingUAV_process.start()
#     egoUAV_process.start()

#     time.sleep(120)
#     with exit_signal.get_lock():
#         exit_signal.value = True # type: ignore
    
#     leadingUAV_process.join()
#     egoUAV_process.join()

#     # Stop recording
#     client.stopRecording()
#     print("\n*****************")
#     print("Recording Stopped")
#     print("*****************\n")

#     client.reset()

from nets.SSD import SSD256x144_VGG16

ssd = SSD256x144_VGG16(root_train_dir="data/empty_map/train/",
                       json_train_labels="data/empty_map/train/empty_map.json",
                       root_test_dir="data/empty_map/test/",
                       json_test_labels="data/empty_map/test/empty_map.json"
                    )

ssd.train(1)
ssd.save("temp.checkpoint")