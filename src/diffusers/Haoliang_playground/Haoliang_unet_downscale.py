import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

from torch.nn.utils import prune
m = prune.ln_structured(
...     nn.Conv2d(5, 3, 2), 'weight', amount=0.3, dim=1, n=float('-inf')
... )

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
#dummy unet
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
def prune_weights(target_layer, strategy, scale_factor, ):
    #input conv layer output conv layer
    pass
def down_scale(full_size_unet:UNet2DConditionModel, scale:int)->UNet2DConditionModel:
    
    pretrained_weights = full_size_unet.state_dict()
    new_state_dict = OrderedDict()
    
    #this method constructs a new state dict for the module
    def fn_recursive_construct_state_dict(module_name: str, module: torch.nn.Module, new_state_dict: OrderedDict, scale_factors):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.SiLU) or isinstance(module, nn.ReLU):
            pruned = prune_weights(module,)
            new_state_dict[f"{module_name}"] = pruned

        for sub_name, child in module.named_children():
            fn_recursive_construct_state_dict(f"{name}.{sub_name}", child, new_state_dict)


    for name, module in full_size_unet.named_children():
        fn_recursive_construct_state_dict(name, module, )
