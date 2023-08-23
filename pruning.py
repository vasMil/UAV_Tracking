from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch_pruning as tp

from nets.DetectionNets import Detection_SSD
from config import DefaultTrainingConfig
from project_types import Pruning_layer_t

def prune_ssd(ssd: Detection_SSD,
              ssd_checkpoint_filename: str,
              sparsities: Tuple[float, float, float, float],
              checkpoint_folder: str,
              save_checkpoint_step: int = 10,
              finetuning_epochs: int = 80,
              learning_rate: float = 0.001,
              batch_size: int = 64
):
    """
    Prunes and finetunes SSD.
    A structured layer-wise pruning method will be used.
    The layer-wise idea can be found here: https://arxiv.org/abs/1811.08342.
    We simplify the method above only on how the sparsity is introduced to the network.
    Instead of training the model using an L1-norm loss function, the applying a threashold
    for each layer and pruning (in a structured manner) the filters with the highest sparsity,
    we follow the methodology described in this paper: https://arxiv.org/abs/2301.12900.
    Thus we utilize the regularization function provided by the library they introduce and apply
    it after each training epoch, to force a more "organized" sparsity.

    Args:
    - ssd: The Detection_SSD object you want to prune. It should have the weights, based on which
    the model wil be pruned, already loaded.
    - ssd_checkpoint_filename: The filename of the checkpoint whose weights will be pruned. This name
    will be used as part of the name of each checkpoint that is going to be saved.
    (ex. pretrained60_sparse80, that means it has been pretrained for 60 epochs and then sparsly trained for another 80)
    - sparsities: The sparsities (percentage of filters to be pruned) for each of the 4 layers.
    - checkpoint_folder: The parent folder where all finetuning checkpoints will be saved. The .model file is the whole
    model as torch.save() stores it and can be used to instanciate a DetectionNetBench object.
    - save_checkpoint_step: Number of epochs between two consecutive saved checkpoints.
    - finetunin_epochs: Number of finetuning training epochs to perform after pruning.
    - learning_rate: The learning rate to use for the finetuning.
    - barch_size: The batch size to use for the finetuning.
    """
    ssd.model.eval()
    # Make all layers trainable, so tracing succeeds
    for p in ssd.model.parameters():
        p.requires_grad_(True)

    # Pruning utilities
    example_inputs = torch.rand(1,3,144,256).to(device=ssd.device)
    importance = tp.importance.MagnitudeImportance(p=1)

    # Define the layers (list of modules) to prune
    pruning_layers: List[Pruning_layer_t] = [
        {
            "sparsity": sparsities[0],
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
            "sparsity": sparsities[1],
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
            "sparsity": sparsities[2],
            "module_names": [
                "backbone.extra.0.7.3",
                "backbone.extra.1.0",
                "backbone.extra.1.2"
            ]
        },
        {
            "sparsity": sparsities[3],
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

    for i, pruning_layer in enumerate(pruning_layers):
        # Extract the modules to be pruned at this layer, into a list
        layer_modules = []
        for module_name in pruning_layer["module_names"]:
            layer_modules.append(ssd.model.get_submodule(module_name))

        # Put all other modules (that won't be pruned) into an "ignored" list
        ignored_layers = []
        for module in ssd.model.modules():
            if isinstance(module, nn.Conv2d) and module not in layer_modules:
                ignored_layers.append(module)

        tp.pruner.MagnitudePruner(
            ssd.model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=1,
            ch_sparsity=pruning_layer["sparsity"],
            global_pruning=False,
            ignored_layers=ignored_layers,
        ).step()

        # Make sure inference still works
        with torch.no_grad():
            ssd.model(example_inputs)

    ##############
    # Finetuning #
    ##############
    # Change the training parameters of the network
    config = DefaultTrainingConfig(
        default_batch_size=batch_size,
        sgd_learning_rate=learning_rate,
        scheduler_milestones=[],
        scheduler_gamma=1,
        losses_plot_ylabel=ssd.losses_plot_ylabel
    )
    ssd.reset(config)
    ssd_checkpoint_filename = ssd_checkpoint_filename.split('.')[0]
    partial_model_path = os.path.join(checkpoint_folder, ssd_checkpoint_filename)
    ssd.save_model(partial_model_path + ".model")
    step = save_checkpoint_step
    for i in range(0, finetuning_epochs, step):
        ssd.train(step)
        ssd.calculate_metrics(True)
        ssd.save_checkpoint(f"{partial_model_path}_finetuned{i+step}.checkpoint")
        ssd.plot_losses(f"{checkpoint_folder}/losses.png")
        ssd.plot_mAPs(f"{checkpoint_folder}/map50.png", "map_50")
        ssd.plot_mAPs(f"{checkpoint_folder}/map75.png", "map_75")
