import json
import math
from itertools import groupby
import os
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            #raise ValueError(
            #    f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            #)
            print(f"LoRA rank {r} is too large. setting to: {min(in_features, out_features)}")
            r = min(in_features, out_features)

        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()
        self.lora_gate = nn.Linear(in_features,1, bias=False)
        nn.init.normal_(self.lora_gate.weight, std=1 / r)

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        x = input.to(self.lora_down.weight.dtype)

        scale = torch.sigmoid(self.lora_gate(x))
        scale = torch.mean(scale, dim=0)
        scale = torch.mean(scale, dim=0)
        
        # for cogvideo
        scale *= self.scale

        scale = scale.to(input.dtype)

        
        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(x)))).to(input.dtype)
            * scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data, self.lora_down.weight.data, self.lora_gate.weight.data
   
    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

TRANSFORMER_DEFAULT_TARGET_REPLACE = {"Attention", "FeedForward"}

UNET_EXTENDED_TARGET_REPLACE = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = TRANSFORMER_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"



def _find_modules(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = None,
    log_scale: bool = False,
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for name, module in model.named_modules()
            if module.__class__.__name__ in ancestor_class # and ('transformer_in' not in name)
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for idx, ancestor in enumerate(ancestors):
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                continue_flag = True
                if 'CogVideoXBlock' in ancestor_class and ('attn1' in fullname or 'ff' in fullname):
                    ancestor_name = f'CogVideoXBlock_{idx}'
                    continue_flag = False
                if continue_flag:
                    continue
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                if name in ['lora_up', 'dropout', 'lora_down', 'lora_gate']:
                    continue
                # Otherwise, yield it
                yield parent, name, module


def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_EXTENDED_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    dropout_p: float = 0.0,
    scale: float = 1.0,
    weight_dtype=None
):
    """
    inject lora into model, and returns lora parameter groups.
    """
    require_grad_params = []
    names = []
    if loras != None:
        loras = torch.load(loras, map_location=model.device)
    if True:
        for target_replace_module_i in target_replace_module:
            for _module, name, _child_module in _find_modules(
                model, [target_replace_module_i], search_class=[nn.Linear, nn.Conv2d, nn.Conv3d]
            ):
                if _child_module.__class__ == nn.Linear:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = LoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r=r,
                        dropout_p=dropout_p,
                        scale=scale,
                    )
                    _tmp.linear.weight = weight
                    if bias is not None:
                        _tmp.linear.bias = bias
                # switch the module
                if weight_dtype is not None:
                    _tmp.to(_child_module.weight.device).to(weight_dtype)
                else:
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                if bias is not None:
                    if weight_dtype is not None:
                        _tmp.to(_child_module.bias.device).to(weight_dtype)
                    else:
                        _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

                _module._modules[name] = _tmp
                require_grad_params.append(_module._modules[name].lora_up.parameters())
                require_grad_params.append(_module._modules[name].lora_down.parameters())
                require_grad_params.append(_module._modules[name].lora_gate.parameters())

                if loras != None:
                    _module._modules[name].lora_up.weight = loras.pop(0)
                    _module._modules[name].lora_down.weight = loras.pop(0)
                    _module._modules[name].lora_gate.weight = loras.pop(0)

                _module._modules[name].lora_up.weight.requires_grad = True
                _module._modules[name].lora_down.weight.requires_grad = True
                _module._modules[name].lora_gate.weight.requires_grad = True
                names.append(name)

    return require_grad_params, names


def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):
    loras = []

    for target_replace_module_i in target_replace_module:

        find_modules = _find_modules(
            model,
            [target_replace_module_i],
            search_class=[LoraInjectedLinear],
        )
        for _p, _n, _child_module in find_modules:
            loras.append((_child_module.lora_up, _child_module.lora_down, _child_module.lora_gate))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module=DEFAULT_TARGET_REPLACE,
    flag=None
):  
    weights = []

    loras = extract_lora_ups_down(model, target_replace_module=target_replace_module)
    for _up, _down, _gate in loras:
        weights.append(_up.weight.to("cpu").to(torch.float32))
        weights.append(_down.weight.to("cpu").to(torch.float32))
        weights.append(_gate.weight.to("cpu").to(torch.float32))

    if not flag:
        torch.save(weights, path)
    else:
        weights_new=[]
        for i in range(0, len(weights), 4):
            subset = weights[i+(flag-1)*2:i+(flag-1)*2+2]
            weights_new.extend(subset)
        torch.save(weights_new, path)