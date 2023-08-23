import os
import re

from nets.DetectionNets import Detection_SSD
from pruning import get_pruning_report

def filter_folders(folder_path, pattern):
    filtered_folders = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path) and re.match(pattern, subfolder):
            filtered_folders.append(os.path.join(folder_path, subfolder))

    return filtered_folders

if __name__ == '__main__':
    # Parent folder path
    parent_folder = "./nets/checkpoints/pruning/ssd_pretrained/finetuning"
    # Pattern to match: 4layers_x_x_x_x where x can be digits
    pattern = r'4layers_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}'

    # Get filtered subfolders
    filtered_subfolders = filter_folders(parent_folder, pattern)

    get_pruning_report([os.path.join(path, "pretrained60_sparse80.model") for path in filtered_subfolders],
                       [os.path.join(path, "pretrained60_sparse80_finetuned80.checkpoint") for path in filtered_subfolders],
                       original_checkpoint="nets/checkpoints/ssd/rand_init/ssd60.checkpoint",
                       train_folder="./data/empty_map/train",
                       train_json="./data/empty_map/train/bboxes.json",
                       test_folder="./data/empty_map/test",
                       test_json="./data/empty_map/test/bboxes.json")
