import os
import re

from pruning import get_pruning_report, plot_report, prune_ssd
from nets.DetectionNets import Detection_SSD

def filter_folders(folder_path, pattern):
    filtered_folders = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path) and re.match(pattern, subfolder):
            filtered_folders.append(os.path.join(folder_path, subfolder))

    return filtered_folders

if __name__ == '__main__':
    # # Parent folder path
    # parent_folder = "./nets/checkpoints/pruning/ssd_pretrained/finetuning"
    # # Pattern to match: 4layers_x_x_x_x where x can be digits
    # pattern = r'4layers_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}'

    # # Get filtered subfolders
    # filtered_subfolders = filter_folders(parent_folder, pattern)

    # # stats = get_pruning_report([os.path.join(path, "pretrained60_sparse80.model") for path in filtered_subfolders],
    # #                            [os.path.join(path, "pretrained60_sparse80_finetuned80.checkpoint") for path in filtered_subfolders],
    # #                            original_checkpoint="nets/checkpoints/ssd/rand_init/ssd60.checkpoint",
    # #                            train_folder="./data/empty_map/train",
    # #                            train_json="./data/empty_map/train/bboxes.json",
    # #                            test_folder="./data/empty_map/test",
    # #                            test_json="./data/empty_map/test/bboxes.json",
    # #                            out_file="nets/checkpoints/pruning/ssd_pretrained/report.json")
    # plot_report(stats=None,
    #             stats_path="nets/checkpoints/pruning/ssd_pretrained/report.json",
    #             plotx_key=lambda stat: stat["layer_sparsity"][0],            # type: ignore
    #             ploty_key=lambda stat: stat["map_dict"]["map_75"], # type: ignore
    #             x_label="Sparsity",
    #             y_label="mAP@75",
    #             plot_filename="report.png")
    for sp1 in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for sp4 in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                                json_train_labels="./data/empty_map/train/bboxes.json",
                                root_test_dir="./data/empty_map/test",
                                json_test_labels="./data/empty_map/test/bboxes.json",
                                checkpoint_path="./nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse80.checkpoint"
            )
            os.mkdir(f"./nets/checkpoints/pruning/ssd_pretrained/finetuning/4layers_{sp1}_09_09_{sp4}/")
            prune_ssd(ssd=ssd,
                    ssd_checkpoint_filename="pretrained60_sparse80.checkpoint",
                    sparsities=(sp1, 0.9, 0.9, sp4,),
                    checkpoint_folder=f"./nets/checkpoints/pruning/ssd_pretrained/finetuning//4layers_{sp1}_09_09_{sp4}/",
                    save_checkpoint_step=80
            )
