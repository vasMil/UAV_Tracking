import torch
import torch.backends.cudnn
from torch_pruning import GroupNormPruner
from torch_pruning.pruner.importance import GroupNormImportance

from nets.DetectionNets import Detection_SSD

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    # Train SSD using sparse the training technique
    # introduced in DepGraph: Towards Any Structural Pruning
    ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                        json_train_labels="./data/empty_map/train/bboxes.json",
                        root_test_dir="./data/empty_map/test",
                        json_test_labels="./data/empty_map/test/bboxes.json"
    )

    ignored_layers = [ssd.model.head]

    pruner = GroupNormPruner(
        ssd.model,
        example_inputs=torch.rand([1,3,144,256]).to(device=ssd.device),
        importance=GroupNormImportance(p=2),
        ignored_layers=ignored_layers
    )

    ssd.train(100, pruner.regularize)
    ssd.calculate_metrics(True)
    ssd.save("nets/checkpoints/prunable_ssd/prunable_ssd100.checkpoint")
    ssd.train(50, pruner.regularize)
    ssd.calculate_metrics(True)
    ssd.save("nets/checkpoints/prunable_ssd/prunable_ssd150.checkpoint")
    ssd.train(50, pruner.regularize)
    ssd.calculate_metrics(True)
    ssd.save("nets/checkpoints/prunable_ssd/prunable_ssd200.checkpoint")
    for i in range(0,100,10):
        ssd.train(10, pruner.regularize)
        ssd.calculate_metrics(True)
        ssd.save(f"nets/checkpoints/prunable_ssd/prunable_ssd{200+(i+10)}.checkpoint")
        ssd.plot_losses("nets/checkpoints/prunable_ssd/")

    ssd.plot_mAPs("nets/checkpoints/prunable_ssd/map_50.png", "map_50")
    ssd.plot_mAPs("nets/checkpoints/prunable_ssd/map_75.png", "map_75")
    ssd.plot_mAPs("nets/checkpoints/prunable_ssd/map_95.png", "map_95")
