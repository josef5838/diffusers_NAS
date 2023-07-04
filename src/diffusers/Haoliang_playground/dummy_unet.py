import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)

torch.manual_seed(0)
unet = UNet2DConditionModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32,
)
# def prune_layer(layer, prune_ratio=0.0, strategy="structured", metric="L1"):
#     weight = layer.weight.data
    
#     if prune_ratio == 0.0: return
#     # assert target in ["weight", "bias"], "You could only prune weight or bias!"
#     assert strategy in ["structured", "unstructured"], "Strategy has to be structured or unstructured!"
#     assert metric in ["L1", "L2", "random"], "L1, L2, or random"
#     #coarse weight pruning
#     if strategy == "structured":
#         if metric == "L1":
#             prune.ln_structured(layer, amount=prune_ratio, n=1, dim=0)
#         if metric == "L2":
#             prune.ln_structured(layer, amount=prune_ratio, n=2, dim=0)
#         if metric == "random":
#             prune.random_structured(layer, amount=prune_ratio)

#         pruned_weights = weight[layer.weight_mask.sum(dim=(1, 2, 3)) != 0]
#         pruned_bias = layer.bias.data[layer.weight_mask.sum(dim=(1, 2, 3)) != 0]
        
#         return pruned_weights, pruned_bias

# def subsample_block(name: str, unet: torch.nn.Module, new_state_dict, new_ModuleList:nn.ModuleList):
#     for name, module in unet.named_modules():
#         if len(list(module.children())) == 0:
#             if (
#                 isinstance(module, nn.Conv2d)
#                 # isinstance(module, nn.MaxPool2d) or 
#                 # isinstance(module, nn.AvgPool2d) or 
#                 # isinstance(module, nn.GroupNorm) or 
#                 # isinstance(module, nn.SiLU)
#             ):
#                 weight, bias = prune_layer(layer=module, prune_ratio=0.2, metric='L1')

#                 new_state_dict[f"{name}.weight"] = weight
#                 new_state_dict[f"{name}.bias"] = bias
#                 if isinstance(module, nn.Conv2d):
#                     new_ModuleList.append()
#             else:
#                 new_state_dict[f"{name}.weight"] = module.state_dict()[[f"{name}.weight"]]
#                 new_state_dict[f"{name}.bias"] = module.state_dict()[[f"{name}.bias"]]
                # new_ModuleList.append(module)
# for name, module in unet.named_children():
#     print(name)
    # print(unet.down_blocks[1].named_modules())
# print(unet.named_children())
# print(type(unet.down_blocks))
# print()
# print(unet.down_blocks)
print(unet.down_blocks)