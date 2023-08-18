import torch
import torch.nn as nn
import torch_pruning as tp
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models import vgg16

model = ssd300_vgg16(pretrained=True, trainable_backbone_layers=5)
example_inputs = torch.rand(1,3,144,256)
model_name = "ssd300_vgg16"
output_transform = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ori_size = tp.utils.count_params(model)
model.cpu().eval()

# Make all layers trainable, so tracing succeeds
for p in model.parameters():
    p.requires_grad_(True)

#########################################
# Ignore unprunable modules
#########################################
ignored_layers = []
# ignored_layers.append(model.get_submodule("head.classification_head.module_list.5"))
# ignored_layers.append(model.get_submodule("head.regression_head.module_list.5"))
prunable_layers = [
    model.get_submodule("backbone.features.0"),
    model.get_submodule("backbone.features.2"),
    model.get_submodule("backbone.features.5"),
    model.get_submodule("backbone.features.7"),
    model.get_submodule("backbone.features.10"),
    model.get_submodule("backbone.features.12"),
    model.get_submodule("backbone.features.14"),
    model.get_submodule("backbone.features.17"),
    model.get_submodule("backbone.features.19"),
    model.get_submodule("backbone.features.21"),
    model.get_submodule("backbone.features.17"),
    model.get_submodule("head.classification_head.module_list.0"),
    model.get_submodule("head.regression_head.module_list.0")
]
for m in model.modules():
    if m not in prunable_layers:
        ignored_layers.append(m)
print(ignored_layers)

#########################################
# (Optional) Register unwrapped nn.Parameters 
# TP will automatically detect unwrapped parameters and prune the last dim for you by default.
# If you want to prune other dims, you can register them here.
#########################################
unwrapped_parameters = None
# if model_name=='ssd300_vgg16':
#     unwrapped_parameters=[ (model.backbone.scale_weight, 0) ] # pruning the 0-th dim of scale_weight

importance = tp.importance.MagnitudeImportance(p=1)
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=example_inputs,
    importance=importance,
    iterative_steps=1,
    ch_sparsity=0.5,
    global_pruning=False,
    unwrapped_parameters=unwrapped_parameters, # type: ignore
    ignored_layers=ignored_layers,
)

#########################################
# Pruning 
#########################################
print("==============Before pruning=================")
print("Model Name: {}".format(model_name))
print(model)

layer_channel_cfg = {}
for module in model.modules():
    if module not in pruner.ignored_layers:
        if isinstance(module, nn.Conv2d):
            layer_channel_cfg[module] = module.out_channels
        elif isinstance(module, nn.Linear):
            layer_channel_cfg[module] = module.out_features

pruner.step()
print("==============After pruning=================")
print(model)

#########################################
# Testing 
#########################################
with torch.no_grad():
    if isinstance(example_inputs, dict):
        out = model(**example_inputs)
    else:
        out = model(example_inputs)
    if output_transform:
        out = output_transform(out)
    print("{} Pruning: ".format(model_name))
    params_after_prune = tp.utils.count_params(model)
    print("  Params: %s => %s" % (ori_size, params_after_prune))
    
    if 'rcnn' not in model_name and model_name!='ssdlite320_mobilenet_v3_large': # RCNN may return 0 proposals, making some layers unreachable during tracing.
        for module, ch in layer_channel_cfg.items():
            if isinstance(module, nn.Conv2d):
                #print(module.out_channels, layer_channel_cfg[module])
                assert int(0.5*layer_channel_cfg[module]) == module.out_channels
            elif isinstance(module, nn.Linear):
                #print(module.out_features, layer_channel_cfg[module])
                assert int(0.5*layer_channel_cfg[module]) == module.out_features

    if isinstance(out, (dict,list,tuple)):
        print("  Output:")
        for o in tp.utils.flatten_as_list(out):
            print(o.shape)
    else:
        print("  Output:", out.shape)
    print("------------------------------------------------------\n")
