from typing import List, TypedDict

import torch
import torch.nn as nn
import torch_pruning as tp

from nets.DetectionNets import Detection_SSD

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
                    checkpoint_path="./nets/checkpoints/ssd/ssd250.checkpoint"
)
model = ssd.model
model.eval()
# Make all layers trainable, so tracing succeeds
for p in model.parameters():
    p.requires_grad_(True)

# Statistics
ori_size = tp.utils.count_params(model)
ori_map = ssd.mAP_dicts[-1][1]
ssd.get_inference_frequency(100, 100, True)

# Pruning utilities
example_inputs = torch.rand(1,3,144,256).to(device=ssd.device)
importance = tp.importance.MagnitudeImportance(p=1)

# Define the layers (list of modules) to prune
pruning_layers: List[Pruning_layer_t] = [
    {
        "sparsity": 0.1,
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
        "sparsity": 0.8,
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
        "sparsity": 0.8,
        "module_names": [
            "backbone.extra.0.7.3",
            "backbone.extra.1.0",
            "backbone.extra.1.2"
        ]
    },
    {
        "sparsity": 0.9,
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

    params_after_prune = tp.utils.count_params(model)
    print("Params: %s => %s" % (ori_size, params_after_prune))

##############
# Finetuning #
##############
ssd.train(20)
pruned_map = ssd.calculate_metrics(False)
print("Before Pruning: --------------------------------------------------------")
print(ori_map)
print("After Pruning: ---------------------------------------------------------")
print(pruned_map)
ssd.get_inference_frequency(100, 100, True)
