from typing import List, TypedDict
from pprint import pprint

import torch
import torch.nn as nn
import torch_pruning as tp

from nets.DetectionNets import Detection_SSD
from config import DefaultTrainingConfig

class Pruning_layer_t(TypedDict):
    sparsity: float
    module_names: List[str]

#############################
# Prepare model for pruning #
#############################
ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                    json_train_labels="./data/empty_map/train/bboxes.json",
                    root_test_dir="./data/empty_map/test",
                    json_test_labels="./data/empty_map/test/bboxes.json",
                    checkpoint_path="./nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse80.checkpoint"
)
model = ssd.model
model.eval()
# Make all layers trainable, so tracing succeeds
for p in model.parameters():
    p.requires_grad_(True)

# Statistics
orig_params = tp.utils.count_params(model)
orig_map = ssd.mAP_dicts[-1][1]
orig_fps = ssd.get_inference_frequency(1000, 100, True)

# Pruning utilities
example_inputs = torch.rand(1,3,144,256).to(device=ssd.device)
importance = tp.importance.MagnitudeImportance(p=1)

# Define the layers (list of modules) to prune
pruning_layers: List[Pruning_layer_t] = [
    {
        "sparsity": 0.,
        "module_names": [
            "backbone.features.0",
            "backbone.features.2",
            "backbone.features.5",
            "backbone.features.7",
            "backbone.features.10",
            "backbone.features.12",
            "backbone.features.14",
            "backbone.features.17",
            "backbone.features.19",
            "backbone.features.21"
        ]
    },
    {
        "sparsity": 0.5,
        "module_names": [
            "backbone.features.21",
            "backbone.extra.0.1",
            "backbone.extra.0.3",
            "backbone.extra.0.5",
            "backbone.extra.0.7.1",
            "backbone.extra.0.7.3"
        ]
    },
    {
        "sparsity": 0.5,
        "module_names": [
            "backbone.extra.0.7.3",
            "backbone.extra.1.0",
            "backbone.extra.1.2"
        ]
    },
    {
        "sparsity": 0.5,
        "module_names": [
            "backbone.extra.1.2",
            "backbone.extra.2.0",
            "backbone.extra.2.2",
            "backbone.extra.3.0",
            "backbone.extra.3.2",
            "backbone.extra.4.0",
            "backbone.extra.4.2"
        ]
    }
]

#####################
# Prune - layerwise #
#####################
for i, pruning_layer in enumerate(pruning_layers):
    print(f"\nPruning layer {i}...")

    # Extract the modules to be pruned at this layer, into a list
    layer_modules = []
    for module_name in pruning_layer["module_names"]:
        layer_modules.append(model.get_submodule(module_name))

    # Put all other modules (that won't be pruned) into an "ignored" list
    ignored_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module not in layer_modules:
            ignored_layers.append(module)

    tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        ch_sparsity=pruning_layer["sparsity"],
        global_pruning=False,
        ignored_layers=ignored_layers,
    ).step()

    ###############
    # Testing ... #
    ###############
    with torch.no_grad():
        out = model(example_inputs)

        if isinstance(out, (dict,list,tuple)):
            print("Inference output shapes:")
            for o in tp.utils.flatten_as_list(out):
                print(o.shape)
        else:
            print("Inference output shape:", out.shape)

    pruned_params = tp.utils.count_params(model)
    print("Params: %s => %s" % (orig_params, pruned_params))

##############
# Finetuning #
##############
# Change the training parameters of the network, since the model
# is now smaller (ex. use a larger lr)
config = DefaultTrainingConfig(
    default_batch_size=64,
    sgd_learning_rate=0.001,
    scheduler_milestones=[],
    scheduler_gamma=1,
    losses_plot_ylabel=ssd.losses_plot_ylabel
)
ssd.reset(config)
ssd.save_model("nets/checkpoints/pruning/ssd_pretrained/finetuning/pretrained60_sparse80.model")
step = 10
for i in range(0,80,step):
    ssd.train(10)
    ssd.calculate_metrics(True)
    ssd.save_checkpoint(f"./nets/checkpoints/pruning/ssd_pretrained/finetuning/pretrained60_sparse80_finetuned{i+step}_4layers_00_05_05_05.checkpoint")
    ssd.plot_losses("./nets/checkpoints/pruning/ssd_pretrained/finetuning/losses.png")
    ssd.plot_mAPs("./nets/checkpoints/pruning/ssd_pretrained/finetuning/map50.png", "map_50")
    ssd.plot_mAPs("./nets/checkpoints/pruning/ssd_pretrained/finetuning/map75.png", "map_75")

pruned_map = ssd.mAP_dicts[-1]
pruned_fps = ssd.get_inference_frequency(1000, 100, True)
pruned_params = tp.utils.count_params(model)
print("Before Pruning: --------------------------------------------------------")
print(f"Number of parameters: {orig_params}")
pprint(orig_map)
pprint(orig_fps)
print("After Pruning: ---------------------------------------------------------")
print(f"Number of parameters: {pruned_params}")
pprint(pruned_map)
pprint(pruned_fps)
