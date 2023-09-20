from typing import List, Tuple, Optional, Callable, Union
import os
import shutil
import glob
import math
from itertools import product

import json
import numpy as np
import torch.nn as nn
import torch.backends.cudnn
import torch_pruning as tp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as sns

from nets.DetectionNets import Detection_SSD
from nets.DetectionNetBench import DetectionNetBench
from models.pareto import PruningSpeedupAccuracyProblem, SparsitySampling, SparsityCrossover, SparsityMutation, SparsityDuplicateElimination
from config import DefaultTrainingConfig
from project_types import Pruned_model_stats_t

SSD_LAYERS = [
    [
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
    ],
    [
        "backbone.features.21",
        "backbone.extra.0.1",
        "backbone.extra.0.3",
        "backbone.extra.0.5",
        "backbone.extra.0.7.1",
        "backbone.extra.0.7.3"
    ],
    [
        "backbone.extra.0.7.3",
        "backbone.extra.1.0",
        "backbone.extra.1.2"
    ],
    [
        "backbone.extra.1.2",
        "backbone.extra.2.0",
        "backbone.extra.2.2",
        "backbone.extra.3.0",
        "backbone.extra.3.2",
        "backbone.extra.4.0",
        "backbone.extra.4.2"
    ]
]

def prune_ssd(ssd: Detection_SSD,
              sparsities: Tuple[float, float, float, float],
              config: DefaultTrainingConfig,
              model_path: str
    ):
    """
    Prunes an SSD model, resets its checkpoint and training configuration and saves the pruned model.

    A structured group-layer-wise pruning method will be used.
    The group-layer-wise idea can be found here: https://arxiv.org/abs/1811.08342.
    We simplify the method above only on how the sparsity is introduced to the network.
    Instead of training the model using an L1-norm loss function, then applying a threshold
    for each layer and lastly pruning (in a structured manner) the filters with the highest sparsity,
    we follow the methodology described in this paper: https://arxiv.org/abs/2301.12900.
    Thus we utilize the regularization function provided by the library they introduce and apply
    it after each training epoch, to force a more "organized" sparsity.

    Args:
    - ssd: The Detection_SSD object you want to prune. The weights that will be used to decide which filters to prune,
    should have been already loaded to the model.
    - sparsities: The sparsities (percentage of filters to be pruned) for each of the 4 layers.
    - config: The training config that will be used for the finetuning process. It contains info like the learning rate.
    It will be used to reset model's losses and mAPs and reconfigure the training config.
    - model_path: The path to save model at (including the name of the model filename).
    The .model file that will be output is the whole model as torch.save() stores it.
    It can be used to instanciate a DetectionNetBench object.

    Proposed config:
    `
    config = DefaultTrainingConfig(\
                default_batch_size = 64,\
                sgd_learning_rate=0.001,\
                scheduler_milestones=[],\
                scheduler_gamma=1,\
                losses_plot_ylabel="Sum of localization (Smooth L1 loss) and classification (Softmax loss)"\
            )
    `
    """
    ssd.model.eval()
    # Make all layers trainable, so tracing succeeds
    for p in ssd.model.parameters():
        p.requires_grad_(True)

    # Pruning utilities
    example_inputs = torch.rand(1,3,144,256).to(device=ssd.device)
    importance = tp.importance.MagnitudeImportance(p=1)

    for sparsity, module_names in zip(sparsities, SSD_LAYERS):
        # Extract the modules to be pruned at this layer, into a list
        layer_modules = []
        for module_name in module_names:
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
            ch_sparsity=sparsity,
            global_pruning=False,
            ignored_layers=ignored_layers,
        ).step()

        # Make sure inference still works
        with torch.no_grad():
            ssd.model(example_inputs)
    
    ssd.reset(config)
    ssd.save_model(model_path + ".model")
    ssd.config.save(os.path.join(os.path.dirname(model_path), "training_config.json"))

def sparse_training_ssd(ssd: Detection_SSD,
                        num_sparse_epochs: int,
                        parent_folder: str,
                        checkpoint_folder: str,
                        checkpoint_filename: str
    ):
    ssd.reset()
    # Make all layers trainable
    for p in ssd.model.parameters():
        p.requires_grad_(True)

    # Define a Group Pruner in order to utilize its regularizer
    pruner = tp.GroupNormPruner(
        ssd.model,
        example_inputs=torch.rand([1,3,144,256]).to(device=ssd.device),
        importance=tp.importance.GroupNormImportance(p=2)
    )

    ssd.save_model(os.path.join(parent_folder, "initialized.model"))
    ssd.save_checkpoint(os.path.join(parent_folder, "initialized.checkpoint"))
    
    # Training Loop
    step = 10
    for epoch in range(0, num_sparse_epochs, 10):
        ssd.train(step, pruner.regularize)
        ssd.calculate_metrics(True)
        ssd.save_checkpoint(os.path.join(checkpoint_folder, f"{checkpoint_filename}_sparse{epoch+step}.checkpoint"))
        ssd.plot_losses(os.path.join(checkpoint_folder, "losses.png"))
        ssd.plot_mAPs(os.path.join(checkpoint_folder, "map50.png"), "map_50")
        ssd.plot_mAPs(os.path.join(checkpoint_folder, "map75.png"), "map_75")

def finetuning(ssd: DetectionNetBench,
               epochs: int,
               finetuning_folder_to_save_checkpoints: str,
               checkpoint_prefix_name: str,
               save_checkpoint_step: int = 10,
               calc_metrics_step: int = 10
    ) -> None:
    """
    Finetunes a pruned SSD model.

    Args:
    - ssd: The DetectionNetBench object with the loaded SSD model and the correct checkpoint, you want to finetune on.
    - epochs: The number of epochs to finetune for.
    - finetuning_folder_to_save_checkpoints: The folder where all finetuning checkpoints will be saved.
    - checkpoint_prefix_name: The prefix name to be used for each checkpoint. This name may specify important information
    about the whole process the model went through before reaching this point.
    An example would be: `pretrained250_sparse270`, which indicates that the model has been trained with all its weights
    for 250 epochs and then it was sparsly trained for another 270 epochs.
    This prefix will result to a checkpoint filename like: `pretrained250_sparse270_finetuned.checkpoint`
    - save_checkpoint_step: Number of epochs between two consecutive saved checkpoints.
    - calc_metrics_step: Number of epochs between two consecutive calculations of the metrics.
    to be used.
    """
    curr_epoch = ssd.epoch
    checkpoint_prefix = (checkpoint_prefix_name + "_") if checkpoint_prefix_name else ""
    step = math.gcd(save_checkpoint_step, calc_metrics_step, epochs)
    for i in range(0, epochs, step):
        ssd.train(step)
        if (i % calc_metrics_step) == 0:
            ssd.calculate_metrics(True)
            ssd.plot_losses(f"{finetuning_folder_to_save_checkpoints}/losses.png")
            ssd.plot_mAPs(f"{finetuning_folder_to_save_checkpoints}/map50.png", "map_50")
            ssd.plot_mAPs(f"{finetuning_folder_to_save_checkpoints}/map75.png", "map_75")
        if i and ((i+step) % save_checkpoint_step) == 0:
            ssd.save_checkpoint(os.path.join(finetuning_folder_to_save_checkpoints,
                                             f"{checkpoint_prefix}finetuned{curr_epoch+i+step}.checkpoint"))

def count_layers_params(model: nn.Module) -> List[int]:
    seen_submodules = []
    layer_params = [0 for _ in SSD_LAYERS]
    for i, layer in enumerate(SSD_LAYERS):
        for submodule_name in layer:
            submodule = model.get_submodule(submodule_name)
            if submodule in seen_submodules:
                continue
            seen_submodules.append(submodule)
            layer_params[i] += tp.utils.count_params(submodule)
    return layer_params

def calc_sparsity(orig_params_cnt: int, pruned_params_cnt: int) -> float:
    return (orig_params_cnt - pruned_params_cnt)/orig_params_cnt

def extract_sparsity_from_model_id(model_id: str) -> List[float]:
    str_spars = model_id.split('_')
    if "layers" in str_spars[0]:
        str_spars = str_spars[1:]
    str_spars = [int(s)/10 if int(s)<=10 else int(s)/100 for s in str_spars]
    return str_spars

def get_model_stats(pruned_net: DetectionNetBench,
                    orig_model_stats: Optional[Pruned_model_stats_t] = None
    ) -> Pruned_model_stats_t:
    model_stats: Pruned_model_stats_t = {
        "num_params": 0,
        "model_id": pruned_net.model_id,
        "flops": 0,
        "layer_params": count_layers_params(pruned_net.model),
        "map_dict": pruned_net.mAP_dicts[-1][1],
        "losses": pruned_net.losses[-1],
        "epoch": pruned_net.epoch,
        "infer_freqs": pruned_net.get_inference_frequency(1000, 1000, True),
        "sparsity": None,
        "layer_sparsity": extract_sparsity_from_model_id(pruned_net.model_id) if orig_model_stats else [1]*4,
        "theoretical_speedup": None,
        "actual_speedup": None
    }
    example_input = torch.rand(1,3,144,256).to(device=pruned_net.device)
    model_stats["flops"], model_stats["num_params"]= tp.utils.op_counter.count_ops_and_params(pruned_net.model, example_input)
    if orig_model_stats:
        model_stats["sparsity"] = calc_sparsity(orig_model_stats["num_params"], model_stats["num_params"])
        model_stats["theoretical_speedup"] = orig_model_stats["flops"] / model_stats["flops"]
        model_stats["actual_speedup"] = orig_model_stats["infer_freqs"]["eval_avg_freq_Hz"] / model_stats["infer_freqs"]["eval_avg_freq_Hz"]
    return model_stats

def get_stats_4_models_in_folder(parent_folder: str,
                                 model_filename: str,
                                 checkpoint_filename: str,
                                 original_checkpoint: str,
                                 train_folder: str,
                                 train_json: str,
                                 test_folder: str,
                                 test_json: str
    ) -> List[Pruned_model_stats_t]:
    ssd = Detection_SSD(checkpoint_path=original_checkpoint)
    orig_stats = get_model_stats(ssd)

    stats = []
    for subfolder in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(full_path):
            checkpoint = os.path.join(full_path, checkpoint_filename)
            model = os.path.join(full_path, model_filename)
            if not (os.path.exists(checkpoint) and os.path.exists(model)):
                raise Exception(f"Checkpoint: {checkpoint} or Model: {model} does not exist!")
            pruned_ssd = DetectionNetBench(model_id=subfolder,
                                           model_path=model,
                                           checkpoint_path=checkpoint)
            stats.append(get_model_stats(pruned_ssd, orig_stats))
    return stats

def save_pruning_report(stats: List[Pruned_model_stats_t], out_file: str) -> None:
    with open(out_file, 'w') as f:
        json.dump(stats, f)

def load_pruning_report(filename: str) -> List[Pruned_model_stats_t]:
    with open(filename, 'r') as f:
        stats = json.load(f)
    return stats

def plot_report(stats: List[Pruned_model_stats_t],
                plotx_key: Callable[[Pruned_model_stats_t], Union[float, int]],
                ploty_key: Callable[[Pruned_model_stats_t], Union[float, int]],
                x_label: str,
                y_label: str,
                plot_filename: str
):
    stats.sort(key=lambda stat: plotx_key(stat)) # type: ignore
    fig, ax = plt.subplots(1, 1)
    x_vals = [plotx_key(stat) for stat in stats] # type: ignore
    y_vals = [ploty_key(stat) for stat in stats] # type: ignore
    id_vals = [stat["model_id"] for stat in stats] # type: ignore
    ax.plot(x_vals, y_vals)
    ax.scatter(x_vals, y_vals)
    # for i, label in enumerate(id_vals):
    #     plt.annotate(label, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(plot_filename)

def layer_group_sparsity_heatmap(stats: List[Pruned_model_stats_t],
                                 filename: str,
                                 map_key: str = "map_75"
    ):
    # Extract the different sparsities at which each layer is evaluated
    num_layers = len(stats[0]["layer_sparsity"])
    per_glayer_sparsities = [[] for _ in range(num_layers)]
    for stat in stats:
        sparsities = stat["layer_sparsity"]
        for layer in range(num_layers):
            if sparsities[layer] not in per_glayer_sparsities[layer]:
                per_glayer_sparsities[layer].append(sparsities[layer])
    
    per_glayer_sparsities[0].sort()
    for layer in range(1, num_layers):
        per_glayer_sparsities[layer].sort()
        if per_glayer_sparsities[0] != per_glayer_sparsities[layer]:
            raise Exception("Not all layers have tests for the same sparsities")

    group_layer_sparsities = per_glayer_sparsities[0]
    
    # Shape the heatmap (i.e. how many group-layer sparsities on the rows and how many on cols)
    # For example if there are 4 diff sparsities for each group-layer, the sparsities for the
    # first two layers will be used as a tuple index for the rows of the heatmap.
    # The rest will be used as a tuple index for the cols of the heatmap.
    num_glayers_in_row = math.floor(num_layers/2)
    num_glayers_in_col = num_layers - num_glayers_in_row
    num_rows = num_glayers_in_row**len(group_layer_sparsities)
    num_cols = num_glayers_in_col**len(group_layer_sparsities)
    row_labels = list(product(group_layer_sparsities, repeat=num_glayers_in_row))
    col_labels = list(product(group_layer_sparsities, repeat=num_glayers_in_col))
    
    heatmap = np.zeros([num_rows, num_cols])
    for stat in stats:
        sparsities = stat["layer_sparsity"]
        row_idx = row_labels.index(tuple(sparsities[:num_glayers_in_row]))
        col_idx = col_labels.index(tuple(sparsities[num_glayers_in_row:]))
        heatmap[row_idx, col_idx] = stat["map_dict"][map_key]

    fig = plt.figure(figsize=(20, 14))
    ax = fig.subplots()
    palette = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(heatmap, ax=ax, xticklabels=row_labels, yticklabels=col_labels, annot=True, linewidth=.5, cmap=palette) # type: ignore
    ax.xaxis.tick_top()
    ax.set_xlabel(f"Layers: {list(range(num_glayers_in_row))}")
    ax.set_ylabel(f"Layers: {list(range(num_glayers_in_row, num_layers))}")
    ax.set_title(f"{map_key} for different group-layer sparsity configurations, for sparsities \u2208 {group_layer_sparsities}", pad=20)
    fig.savefig(filename)
    plt.close(fig)

def impact_of_sparsity_for_grouplayer(stats: List[Pruned_model_stats_t],
                                      grouplayer_num: int,
                                      other_grouplayer_sparsities: List[float],
                                      filename: str,
                                      all_sparsities: List[float] = [0.8, 0.85, 0.9, 0.95],
                                      map_key: str = "map_75"
    ) -> None:
    """
    For a fixed sparsity at all group-layers, except one,
    plot the mAP for all different sparsities for that one group-layer.

    Args:
    stats: The stats as returned by get_stats_4_models_in_folder().
    grouplayer_num: The number of the "special" group-layer (not index, start from 1).
    other_grouplayer_sparsities: The fixed sparsities of the other group-layers\
    must be sorted by the number of the corresponding group-layer.
    all_sparsities: All different sparsities for the "special" group-layer.
    """
    # Construct a list with all possible sparsity lists
    # i.e. For each item in the outer list the other_layer_sparsities will be constant
    # and the sparsity of the layer (layer_num) will vary.
    diff_sparsity_setups = []
    for sparsity in all_sparsities:
        temp = other_grouplayer_sparsities.copy()
        temp.insert(grouplayer_num-1, sparsity)
        diff_sparsity_setups.append(temp)

    # Organize the useful (sparsity, mAP) tuples into a list
    useful_mAPs: List[Tuple[float, float]] = []
    for stat in stats:
        if stat["layer_sparsity"] in diff_sparsity_setups:
            useful_mAPs.append((stat["layer_sparsity"][grouplayer_num-1], stat["map_dict"][map_key]))

    # Sort the list by the sparsity of the "special" group-layer
    useful_mAPs.sort(key=lambda x: x[0])
    x_vals = [u[0] for u in useful_mAPs]
    y_vals = [u[1] for u in useful_mAPs]
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x_vals, y_vals)
    ax.scatter(x_vals, y_vals)
    ax.set_xlabel(f"All different sparsities of group-layer {grouplayer_num}", labelpad=10)
    str_all_sparsities = [str(x) for x in all_sparsities]
    ax.set_xticks(all_sparsities)
    ax.set_xticklabels(str_all_sparsities)
    str_other_grouplayer_sparsities = [str(x) for x in other_grouplayer_sparsities]
    str_other_grouplayer_sparsities.insert(grouplayer_num-1, 'x')
    formated_sparsities = ", ".join(str_other_grouplayer_sparsities)
    ax.set_ylabel(f"mAPs for the sparsity setup: {formated_sparsities}", labelpad=10)
    ax.set_yticks([y/10 for y in range(0, 11)])
    ax.set_title(f"Impact on {map_key} of pruning group-layer {grouplayer_num} at different sparsities,"
                 f"\nwhile holding the other group-layer sparsities constant",
                 pad=20)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.85)
    fig.savefig(filename)
    plt.close(fig)

def extract_pareto_stats_using_grouplayer_sparsity(stats: List[Pruned_model_stats_t],
                                    pareto_sparsities: List[Tuple]
    ) -> List[Pruned_model_stats_t]:
    filtered_stats = []
    encountered_sparsities = []

    for stat in stats:
        layer_sparsity = tuple(stat["layer_sparsity"])

        if layer_sparsity in pareto_sparsities:
            if layer_sparsity in encountered_sparsities:
                raise ValueError(f"Duplicate pareto sparsity: {layer_sparsity}")
            
            filtered_stats.append(stat)
            encountered_sparsities.append(layer_sparsity)

    return filtered_stats

def get_pruning_pareto(stats: List[Pruned_model_stats_t],
                       map_key: str = "map_75",
                       min_map: float = 0.1
                    ) -> List[Pruned_model_stats_t]:
    """
    Calculates the pareto front for a discrete set of points.
    Currently the stats object should contain all combinations of [0.8, 0.85, 0.9, 0.95].

    Args:
    - stats: The stats of all pruned models as returned by get_stats_4_models_in_folder.
    - map_key: A key of stats' dictionary map_dict (ex. 'map_75', 'map_50', 'map_large').

    Returns:
    A List of Pruned_model_stats_t with the stats of the pareto points.
    """
    # Find the pareto
    problem = PruningSpeedupAccuracyProblem(stats, map_key=map_key)
    algorithm = NSGA2(pop_size=20,
                      sampling=SparsitySampling(),                          # type: ignore
                      crossover=SparsityCrossover(),                        # type: ignore
                      mutation=SparsityMutation(),                          # type: ignore
                      eliminate_duplicates=SparsityDuplicateElimination())

    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=('n_gen', 100))
    pareto_sparsities: np.ndarray = res.X.squeeze()     # type: ignore
    pareto_stats = extract_pareto_stats_using_grouplayer_sparsity(stats, pareto_sparsities.tolist())
    
    filtered_pareto_stats = []
    for stat in pareto_stats:
        if stat["map_dict"][map_key] >= min_map:
            filtered_pareto_stats.append(stat)

    return filtered_pareto_stats

def plot_pruning_pareto(stats: List[Pruned_model_stats_t], filename: str, map_key: str = "map_75") -> None:
    # Extract information to plot for all stats (included or not in the pareto)
    accuracies = []
    theoretical_speedups = []
    for stat in stats:
        accuracies.append(stat["map_dict"][map_key])
        theoretical_speedups.append(stat["theoretical_speedup"])

    # Extract information to plot for the pareto points
    pareto_stats = get_pruning_pareto(stats, map_key=map_key)
    pareto_sparsities = []
    pareto_accuracies = []
    pareto_theoretical_speedups = []
    for stat in pareto_stats:
        pareto_sparsities.append(stat["layer_sparsity"])
        pareto_accuracies.append(stat["map_dict"][map_key])
        pareto_theoretical_speedups.append(stat["theoretical_speedup"])

    # Sort the parrallel arrays above, so they plot nicely on the x axis
    sorted_indices = np.argsort(pareto_theoretical_speedups)
    pareto_theoretical_speedups = pareto_theoretical_speedups[sorted_indices]
    pareto_accuracies = pareto_accuracies[sorted_indices]
    pareto_sparsities = pareto_sparsities[sorted_indices]

    # Plot all data points and highlight the pareto
    fig = plt.figure()
    ax = fig.subplots()
    ax.scatter(x=theoretical_speedups, y=accuracies)
    ax.plot(pareto_theoretical_speedups, pareto_accuracies, marker='x', linestyle='-', color="red")
    # # Add labels to the data points
    # for i, label in enumerate(pareto_sparsities):
    #     plt.annotate(label, (pareto_theoretical_speedup[i], pareto_accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_xlabel("Theoretical Speedup (%)", labelpad=10)
    ax.set_ylabel(f"{map_key} Accuracy", labelpad=10)
    ax.set_title("Pareto front for different sparsities at different layer-groups", pad=20)
    fig.savefig(filename, format="svg")
    plt.close(fig)

def copy_pareto_folders(parent_folder: str,
                        pareto_stats: List[Pruned_model_stats_t],
                        dest_parent_folder: str
    ):
    # Create parent destination folder if it doesn't already exist
    os.makedirs(dest_parent_folder, exist_ok=True)

    # For each pareto point 
    for stat in pareto_stats:
        # Configure the destination folder
        curr_pareto_folder = os.path.join(parent_folder, stat["model_id"])
        dest_pareto_folder = os.path.join(dest_parent_folder, stat["model_id"])
        os.mkdir(dest_pareto_folder)

        # Move the model and checkpoint files to the new folder and rename them so
        # the contents of all files are uniform.
        model_files = glob.iglob(os.path.join(curr_pareto_folder, "*.model"))
        checkpoint_files = glob.iglob(os.path.join(curr_pareto_folder, "*.checkpoint"))
        for model_file, checkpoint_file in zip(model_files, checkpoint_files):
            if os.path.isfile(model_file) and os.path.isfile(checkpoint_file):
                shutil.copy2(model_file, os.path.join(dest_pareto_folder, "model.pth"))
                shutil.copy2(checkpoint_file, os.path.join(dest_pareto_folder, "checkpoint.pth"))
            else:
                print('F')
