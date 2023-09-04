from typing import Tuple
import os
from itertools import product

from pruning import prune_ssd, layer_group_sparsity_heatmap, load_pruning_report, save_pruning_report, get_stats_4_models_in_folder, impact_of_sparsity_for_grouplayer
from nets.DetectionNets import Detection_SSD

def sparsities_to_folder_name(sparsities: Tuple[float, float, float, float]) -> str:
    folder_name = "4layers"
    for sparsity in sparsities:
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

    stats = get_stats_4_models_in_folder("nets/checkpoints/pruning/ssd_pretrained/finetuning",
                                         "pretrained60_sparse80.model",
                                         "pretrained60_sparse80_finetuned80.checkpoint",
                                         "nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse80.checkpoint",
                                         train_folder="data/empty_map/train",
                                         train_json="data/empty_map/train/bboxes.json",
                                         test_folder="data/empty_map/test",
                                         test_json="data/empty_map/test/bboxes.json")

    save_pruning_report(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/report.json")
    # stats = load_pruning_report("nets/checkpoints/pruning/ssd_pretrained/finetuning/report.json")
    layer_group_sparsity_heatmap(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/stats_heatmap_map.png", map_key="map")
    layer_group_sparsity_heatmap(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/stats_heatmap_map50.png", map_key="map_50")
    layer_group_sparsity_heatmap(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/stats_heatmap_map75.png", map_key="map_75")
    layer_group_sparsity_heatmap(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/stats_heatmap_mapsmall.png", map_key="map_small")
    layer_group_sparsity_heatmap(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/stats_heatmap_maplarge.png", map_key="map_large")
    layer_group_sparsity_heatmap(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/stats_heatmap_mapmedium.png", map_key="map_medium")
    impact_of_sparsity_for_grouplayer(stats, 1, [0.9, 0.9, 0.9], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_x909090.png")
    impact_of_sparsity_for_grouplayer(stats, 1, [0.8, 0.8, 0.8], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_x808080.png")
    impact_of_sparsity_for_grouplayer(stats, 1, [0.95, 0.95, 0.95], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_x959595.png")
    impact_of_sparsity_for_grouplayer(stats, 4, [0.9, 0.9, 0.9], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_909090x.png")
    impact_of_sparsity_for_grouplayer(stats, 4, [0.8, 0.8, 0.8], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_808080x.png")
    impact_of_sparsity_for_grouplayer(stats, 2, [0.8, 0.8, 0.8], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_80x8080.png")
    impact_of_sparsity_for_grouplayer(stats, 3, [0.8, 0.8, 0.8], "nets/checkpoints/pruning/ssd_pretrained/finetuning/impact_map75_8080x80.png")
