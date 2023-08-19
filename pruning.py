from typing import Tuple, List, TypedDict

import torch
import torch.nn as nn
import torch_pruning as tp
from torchvision.models.detection.ssd import ssd300_vgg16

class Pruning_layer_t(TypedDict):
    sparsity: float
    module_names: List[str]

#############################
# Prepare model for pruning #
#############################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ssd300_vgg16(pretrained=True, trainable_backbone_layers=5)
model.to(device=device).eval()
# Make all layers trainable, so tracing succeeds
for p in model.parameters():
    p.requires_grad_(True)

# Statistics
ori_size = tp.utils.count_params(model)

# Pruning utilities
example_inputs = torch.rand(1,3,144,256).to(device=device)
importance = tp.importance.MagnitudeImportance(p=1)

# Define the layers (list of modules) to prune
pruning_layers: List[Pruning_layer_t] = [
    {
        "sparsity": 0.5,
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

# Prune - layerwise
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

    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        ch_sparsity=pruning_layer["sparsity"],
        global_pruning=False,
        ignored_layers=ignored_layers,
    )
    pruner.step()

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
