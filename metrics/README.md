# Appearance-motion combined customization evaluation

# Environment

```shell
cd metrics

conda create -n jointtuner-eval python=3.10
conda activate jointtuner-eval

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/Vchitect/VBench.git

pip install -r requirements.txt
```

## Modify vbench output format
```python
# These changes are in VBench/vbench/__init__.py
class VBench(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        if len(output_path) > 0:
            os.makedirs(self.output_path, exist_ok=True)

    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='vbench_standard', **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'vbench.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        if len(self.output_path) > 0:
            output_name = os.path.join(self.output_path, name+'_eval_results.json')
            if get_rank() == 0:
                save_json(results_dict, output_name)
                print0(f'Evaluation results saved to {output_name}')
        else:
            return results_dict
```

## Install VBench
```shell
cd VBench

pip install .
```

# Downloading pretrained models

```shell
# For CogVideoX-2B 
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --resume-download

# For CogVideoX-5B
huggingface-cli download yuvalkirstain/PickScore_v1 --resume-download
```

# Evaluating

```shell
bash eval_folder.sh
```