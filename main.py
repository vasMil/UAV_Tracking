import torch
import torch.backends.cudnn
from torch_pruning import GroupNormPruner
from torch_pruning.pruner.importance import GroupNormImportance

from nets.DetectionNets import Detection_SSD

if __name__ == '__main__':
    ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                        json_train_labels="./data/empty_map/train/bboxes.json",
                        root_test_dir="./data/empty_map/test",
                        json_test_labels="./data/empty_map/test/bboxes.json",
                        checkpoint_path="./nets/checkpoints/ssd/rand_init/ssd60.checkpoint"
    )

    # Save the initialized model
    ssd.save_checkpoint(f"./nets/checkpoints/pruning/ssd_pretrained/initialized_rand_init_pretrained60.checkpoint")

    ignored_layers = [ssd.model.head]

    pruner = GroupNormPruner(
        ssd.model,
        example_inputs=torch.rand([1,3,144,256]).to(device=ssd.device),
        importance=GroupNormImportance(p=2),
        ignored_layers=ignored_layers
    )

    torch.backends.cudnn.benchmark = True
    step = 10
    for i in range(0, 80, step):
        ssd.train(step, pruner.regularize)
        ssd.calculate_metrics(True)
        ssd.save_checkpoint(f"./nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse{i+step}.checkpoint")
        ssd.plot_losses(f"./nets/checkpoints/pruning/ssd_pretrained/sparse_training/losses.png")
        ssd.plot_mAPs(f"./nets/checkpoints/pruning/ssd_pretrained/sparse_training/map50.png", "map_50")
        ssd.plot_mAPs(f"./nets/checkpoints/pruning/ssd_pretrained/sparse_training/map75.png", "map_75")
