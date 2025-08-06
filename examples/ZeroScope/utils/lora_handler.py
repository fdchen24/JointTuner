import os
from logging import warnings
import torch
from typing import Union
from types import SimpleNamespace
from models.unet_3d_condition import UNet3DConditionModel
from transformers import CLIPTextModel

from .lora import (
    extract_lora_ups_down,
    inject_trainable_lora_extended,
    save_lora_weight,
)


FILE_BASENAMES = ['unet', 'text_encoder']
LORA_FILE_TYPES = ['unet.pt', 'unet.safetensors']
CLONE_OF_SIMO_KEYS = ['model', 'loras', 'target_replace_module', 'r']
STABLE_LORA_KEYS = ['model', 'target_module', 'search_class', 'r', 'dropout', 'lora_bias']

lora_versions = dict(
    stable_lora = "stable_lora",
    cloneofsimo = "cloneofsimo"
)

lora_func_types = dict(
    loader = "loader",
    injector = "injector"
)

lora_args = dict(
    model = None,
    loras = None,
    target_replace_module = [],
    target_module = [],
    r = 4,
    search_class = [torch.nn.Linear],
    dropout = 0,
    lora_bias = 'none'
)

LoraVersions = SimpleNamespace(**lora_versions)
LoraFuncTypes = SimpleNamespace(**lora_func_types)

LORA_VERSIONS = [LoraVersions.stable_lora, LoraVersions.cloneofsimo]
LORA_FUNC_TYPES = [LoraFuncTypes.loader, LoraFuncTypes.injector]

# Filters the dictionary by keys, returning a new dictionary with only the specified keys
def filter_dict(_dict, keys=[]):
    if len(keys) == 0:
        assert "Keys cannot empty for filtering return dict."
    
    for k in keys:
        if k not in lora_args.keys():
            assert f"{k} does not exist in available LoRA arguments"
            
    return {k: v for k, v in _dict.items() if k in keys}

# Class for handling LoRA (Low-Rank Adaptation) integration into models like UNet and CLIP
class LoraHandler(object):
    def __init__(self, 
                 version: LORA_VERSIONS = LoraVersions.cloneofsimo,  # Version of the LoRA implementation
                 use_unet_lora: bool = False,                       # Flag to indicate if UNet should use LoRA
                 use_text_lora: bool = False,                       # Flag to indicate if text encoder should use LoRA
                 save_for_webui: bool = False,                      # Flag to save LoRA weights for WebUI compatibility
                 only_for_webui: bool = False,                      # Flag to save weights only for WebUI
                 lora_bias: str = 'none',                           # Bias type for LoRA
                 unet_replace_modules: list = None,                 # List of modules to replace in UNet
                 text_encoder_replace_modules: list = None          # List of modules to replace in text encoder
                ):
        # Initializes the LoraHandler class with configuration options
        self.version = version
        self.lora_injector = inject_trainable_lora_extended  # Injector function for LoRA weights
        self.lora_bias = lora_bias
        self.use_unet_lora = use_unet_lora
        self.use_text_lora = use_text_lora
        self.save_for_webui = save_for_webui
        self.only_for_webui = only_for_webui
        self.unet_replace_modules = unet_replace_modules
        self.text_encoder_replace_modules = text_encoder_replace_modules
        self.use_lora = any([use_text_lora, use_unet_lora])  # Checks if LoRA should be used on either module

    def is_cloneofsimo_lora(self):
        # Returns whether the current LoRA version is 'cloneofsimo'
        return self.version == LoraVersions.cloneofsimo

    def check_lora_ext(self, lora_file: str):
        # Checks if a file has a valid LoRA extension
        return lora_file.endswith(tuple(LORA_FILE_TYPES))

    def get_lora_file_path(self, 
                           lora_path: str, 
                           model: Union[UNet3DConditionModel, CLIPTextModel]
                          ):
        # Retrieves the path to a valid LoRA file based on the model type
        if os.path.exists(lora_path):
            lora_filenames = [fns for fns in os.listdir(lora_path)]
            is_lora = self.check_lora_ext(lora_path)

            is_unet = isinstance(model, UNet3DConditionModel)
            is_text =  isinstance(model, CLIPTextModel)

            idx = 0 if is_unet else 1
            base_name = FILE_BASENAMES[idx]

            for lora_filename in lora_filenames:
                is_lora = self.check_lora_ext(lora_filename)
                if not is_lora:
                    continue

                if base_name in lora_filename:
                    return os.path.join(lora_path, lora_filename)

        return None

    def get_lora_func_args(self, 
                           lora_path, 
                           use_lora, 
                           model, 
                           replace_modules, 
                           r, 
                           dropout, 
                           lora_bias, 
                           scale
                          ):
        # Prepares arguments for LoRA function injection based on configuration
        return_dict = lora_args.copy()
    
        if self.is_cloneofsimo_lora():
            return_dict = filter_dict(return_dict, keys=CLONE_OF_SIMO_KEYS)  # Filters LoRA arguments for cloneofsimo
            return_dict.update({
                "model": model,
                "loras": self.get_lora_file_path(lora_path, model),
                "target_replace_module": replace_modules,
                "r": r,
                "scale": scale,
                "dropout_p": dropout
            })

        return return_dict

    def do_lora_injection(self, 
                          model, 
                          replace_modules, 
                          bias='none',
                          dropout=0,
                          r=4,
                          lora_loader_args=None,
                         ):
        # Performs LoRA injection into the model based on the current configuration
        REPLACE_MODULES = replace_modules

        params = None
        negation = None
        is_injection_hybrid = False

        if self.is_cloneofsimo_lora():
            is_injection_hybrid = True
            injector_args = lora_loader_args

            params, negation = self.lora_injector(**injector_args)  # Calls the LoRA injector function
            loras = extract_lora_ups_down(model, target_replace_module=REPLACE_MODULES)
            for _up, _down, _gate in loras:
                if all(x is not None for x in [_up, _down, _gate]):
                    print(f"Lora successfully injected into {model.__class__.__name__}.")
                break

            return params, negation, is_injection_hybrid

        return params, negation, is_injection_hybrid

    def add_lora_to_model(self, 
                          use_lora, 
                          model, 
                          replace_modules, 
                          dropout=0.0, 
                          lora_path='', 
                          r=16, 
                          scale=1.0
                         ):
        # Adds LoRA parameters to the model after successful injection
        params = None
        negation = None

        lora_loader_args = self.get_lora_func_args(
            lora_path,
            use_lora,
            model,
            replace_modules,
            r,
            dropout,
            self.lora_bias,
            scale
        )

        params, negation, is_injection_hybrid = self.do_lora_injection(
            model, 
            replace_modules, 
            bias=self.lora_bias,
            lora_loader_args=lora_loader_args,
            dropout=dropout,
            r=r,
        )
        
        params = model if params is None else params
        return params, negation

    def save_cloneofsimo_lora(self, 
                              model, 
                              save_path, 
                              step, 
                              flag
                             ):
        # Saves the LoRA weights for 'cloneofsimo' version to the specified path
        def save_lora(model, name, condition, replace_modules, step, save_path, flag=None):
            if condition and replace_modules is not None:
                save_path = f"{save_path}/{step}_{name}.pt"
                save_lora_weight(model, save_path, replace_modules, flag)

        save_lora(
            model.unet, 
            FILE_BASENAMES[0], 
            self.use_unet_lora, 
            self.unet_replace_modules, 
            step,
            save_path,
            flag
        )
        save_lora(
            model.text_encoder, 
            FILE_BASENAMES[1], 
            self.use_text_lora, 
            self.text_encoder_replace_modules, 
            step, 
            save_path,
            flag
        )

    def save_lora_weights(self, 
                          model: None, 
                          save_path: str ='', 
                          step: str = '', 
                          flag=None
                         ):
        # Saves the LoRA weights to the specified path for the model
        save_path = f"{save_path}/lora"
        os.makedirs(save_path, exist_ok=True)

        if self.is_cloneofsimo_lora():
            if any([self.save_for_webui, self.only_for_webui]):
                warnings.warn(
                    """
                    You have 'save_for_webui' enabled, but are using cloneofsimo's LoRA implemention.
                    Only 'stable_lora' is supported for saving to a compatible webui file.
                    """
                )
            self.save_cloneofsimo_lora(model, save_path, step, flag)
