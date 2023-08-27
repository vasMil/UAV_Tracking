from typing import Tuple
import os
from itertools import product

from pruning import prune_ssd
from nets.DetectionNets import Detection_SSD

def sparsities_to_folder_name(sparsities: Tuple[float, float, float, float]) -> str:
    folder_name = "4layers"
    for sparsity in sparsities:
        # folder_name += f"_{str(sparsity).replace('.', '')}"
        folder_name += f"_{int(sparsity*100)}"
    return folder_name

if __name__ == '__main__':
    sparsities = [s/100 for s in range(80, 96, 5)]
    for cur_sparsities in product(sparsities, repeat=4):
        folder_name = sparsities_to_folder_name(cur_sparsities)
        ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                            json_train_labels="./data/empty_map/train/bboxes.json",
                            root_test_dir="./data/empty_map/test",
                            json_test_labels="./data/empty_map/test/bboxes.json",
                            checkpoint_path="./nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse80.checkpoint"
        )
        os.mkdir(f"./nets/checkpoints/pruning/ssd_pretrained/finetuning/{folder_name}")
        prune_ssd(ssd=ssd,
                  ssd_checkpoint_filename="pretrained60_sparse80.checkpoint",
                  sparsities=cur_sparsities,
                  checkpoint_folder=f"./nets/checkpoints/pruning/ssd_pretrained/finetuning/{folder_name}/",
                  calc_metrics_step=10,
                  save_checkpoint_step=80
        )
