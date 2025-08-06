import os
import torch
from types import SimpleNamespace

from .lora import (
    extract_lora_ups_down,
    inject_trainable_lora_extended,
    save_lora_weight,
)

FILE_BASENAMES = ['unet', 'text_encoder', 'transformer']
LORA_FILE_TYPES = ['unet.pt', 'unet.safetensors', 'transformer.pt']
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

def filter_dict(_dict, keys=[]):
    if len(keys) == 0:
        assert "Keys cannot empty for filtering return dict."
    
    for k in keys:
        if k not in lora_args.keys():
            assert f"{k} does not exist in available LoRA arguments"
            
    return {k: v for k, v in _dict.items() if k in keys}

class LoraHandler(object):
    def __init__(
        self, 
        version: LORA_VERSIONS = LoraVersions.cloneofsimo, 
        lora_bias: str = 'none',
        replace_modules: list = None,
    ):
        self.version = version
        self.lora_injector = inject_trainable_lora_extended
        self.lora_bias = lora_bias
        self.replace_modules = replace_modules

    def is_cloneofsimo_lora(self):
        return self.version == LoraVersions.cloneofsimo
                
    def check_lora_ext(self, lora_file: str):
        return lora_file.endswith(tuple(LORA_FILE_TYPES))

    def get_lora_file_path(
        self, 
        lora_path: str, 
        model
    ):
        if os.path.exists(lora_path):
            lora_filenames = [fns for fns in os.listdir(lora_path)]

            for lora_filename in lora_filenames:
                is_lora = self.check_lora_ext(lora_filename)
                if not is_lora:
                    continue

                return os.path.join(lora_path, lora_filename)

        return None

    def handle_lora_load(self, file_name:str, lora_loader_args: dict = None):
        self.lora_loader(**lora_loader_args)
        print(f"Successfully loaded LoRA from: {file_name}")
    
    def load_lora(self, model, lora_path: str = '', lora_loader_args: dict = None,):
        try:
            lora_file = self.get_lora_file_path(lora_path, model)

            if lora_file is not None:
                lora_loader_args.update({"lora_path": lora_file})
                self.handle_lora_load(lora_file, lora_loader_args)

            else:
                print(f"Could not load LoRAs for {model.__class__.__name__}. Injecting new ones instead...")

        except Exception as e:
            print(f"An error occurred while loading a LoRA file: {e}")
                 
    def get_lora_func_args(self, lora_path, model, replace_modules, r, dropout, scale, weight_dtype):
        return_dict = lora_args.copy()
    
        if self.is_cloneofsimo_lora():
            return_dict = filter_dict(return_dict, keys=CLONE_OF_SIMO_KEYS)
            return_dict.update({
                "model": model,
                "loras": self.get_lora_file_path(lora_path, model),
                "target_replace_module": replace_modules,
                "r": r,
                "scale": scale,
                "dropout_p": dropout,
                "weight_dtype": weight_dtype
            })

        return return_dict

    def do_lora_injection(
        self, 
        model, 
        replace_modules, 
        bias='none',
        dropout=0,
        r=4,
        lora_loader_args=None,
    ):
        REPLACE_MODULES = replace_modules

        params = None
        negation = None
        is_injection_hybrid = False

        if self.is_cloneofsimo_lora():
            is_injection_hybrid = True
            injector_args = lora_loader_args

            params, negation = self.lora_injector(**injector_args)  # inject_trainable_lora_extended
            loras =  extract_lora_ups_down(model, target_replace_module=REPLACE_MODULES)
            for _up, _down, _gate in loras:
                if all(x is not None for x in [_up, _down, _gate]):
                    print(f"Lora successfully injected into {model.__class__.__name__}.")
                break
           

            return params, negation, is_injection_hybrid

        return params, negation, is_injection_hybrid

    def add_lora_to_model(self, model, replace_modules, dropout=0.0, lora_path='', r=16, scale=1.0, weight_dtype=torch.float32):
        params = None
        negation = None

        lora_loader_args = self.get_lora_func_args(
            lora_path,
            model,
            replace_modules,
            r,
            dropout,
            scale,
            weight_dtype,
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

    def save_cloneofsimo_lora(self, model, save_path, step, flag):
        def save_lora(model, name, replace_modules, step, save_path, flag=None):
            if replace_modules is not None:
                save_path = f"{save_path}/{step}_{name}.pt"
                save_lora_weight(model, save_path, replace_modules, flag)

        save_lora(
            model, 
            FILE_BASENAMES[2], 
            self.replace_modules, 
            step,
            save_path,
            flag
        )


    def save_lora_weights(self, model: None, save_path: str ='',step: str = '', flag=None):
        save_path = f"{save_path}/lora"
        os.makedirs(save_path, exist_ok=True)

        if self.is_cloneofsimo_lora():
            self.save_cloneofsimo_lora(model, save_path, step, flag)
