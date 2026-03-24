
<p align="center">

  <h2 align="center" style="vertical-align: middle; margin-right: 2px;">
  <img src="asserts/favicon.ico" alt="JointTuner Icon" width="24" height="24" style="vertical-align: middle; margin-right: 2px;">
  JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation
  </h2>
  <p align="center">
    <a href="https://fdchen24.github.io/"><strong>Fangda Chen</strong></a>
    ·
    <a href="https://sshan-zhao.github.io/"><strong>Shanshan Zhao</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=7OZMTAMAAAAJ"><strong>Chuanfu Xu</strong></a>
    ·
    <a href="https://lan-long.github.io/"><strong>Long Lan</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/pdf/2503.23951"><img src='https://img.shields.io/badge/arXiv-2503.23951-b31b1b.svg'></a>
        <a href='https://fdchen24.github.io/JointTuner-Website/'><img src='https://img.shields.io/badge/Project_Page-JointTuner-blue'></a>
        <a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>
    <br>
    <b>National University of Defense Technology &nbsp;·&nbsp; Alibaba International Digital Commerce Group</b>
  </p>

https://github.com/user-attachments/assets/c7703569-189c-444e-a954-c3e88ba058dd

<p align="center">
<em>JointTuner enables joint optimization of appearance and motion for high-fidelity customized video generation, mitigating concept interference and appearance contamination.</em>
</p>

## 🎯 Task Definition
**Appearance-Motion Combined Customization** of Text-to-Video Diffusion Models: 
Given reference images (for appearance) and reference videos (for motion), the task is to adapt pre-trained text-to-video diffusion models to generate videos that faithfully preserve the desired appearance while accurately replicating the target motion pattern.

## 🔧 Compatibility
JointTuner is architecture-agnostic and supports both:
- **UNet-based models** (e.g., ZeroScope, ModelScope)
- **Diffusion Transformer-based models** (e.g., CogVideo-2B/5B, Wan2.1-1.3B)

## 📰 News
- [2025.08.06] The code for JointTuner, implemented based on UNet (e.g., ZeroScope and ModelScope) and DiT (e.g., CogVideoX), along with the constructed benchmark, has been released.
- [2025.03.31] Paper and project page released.

## ✅ ToDo List
- [ ] Release the code implemented on WanVideo
- [x] Implementations based on WanVideo
- [x] Implementations based on CogVideoX
- [x] Implementations based on ZeroScope and ModelScope


## 📁 Project Overview

- `datasets/`

Contains various datasets of our proposed benchmark. This directory includes three subfolders.

`subject/`: Contains data files related to subjects.
`motion_16f/`: Contains data videos related to motions with 16 frames.
`motion_49f/`: Contains data videos related to motions with 49 frames.

- `examples/`

This directory includes implementations of JointTuner on different text-to-video models.

`ZeroScope/` and `ModelScope/`: ZeroScope and ModelScope share a common codebase. For specific implementation steps, refer to the `README.md` in the `ZeroScope/` folder.

`CogVideo/`: In CogVideo, both CogVideoX-2B and CogVideoX-5B share a common codebase. The implementation in the CogVideo folder can be found with the respective `README.md` for detailed instructions.


- `metrics/`

Contains the evaluation code of our proposed benchmark, used to assess the performance of the models. For detailed implementation steps, refer to the `README.md` in the respective folder.

## ⚡ Quik Start

### Setup

```
git clone https://github.com/fdchen24/JointTuner.git
cd JointTuner
```

To get started with JointTuner, simply navigate to the examples directory and choose your preferred base text-to-video model.

### UNet-based models (ZeroScope or ModelScope)

1. **Enter the specified project**

```bash
cd examples/ZeroScope
```

2. **GPU Requirements**

The training and inference are conducted on 1 RTX 4090 GPU (24GB VRAM)

3. **Environment**

```shell
cd examples/ZeroScope

conda create -n jointtuner-zs python=3.8
conda activate jointtuner-zs

pip install -r requirements.txt
```

4. **Downloading pretrained models**

```shell
# For ZeroScope 
huggingface-cli download cerspense/zeroscope_v2_576w --resume-download

# For ModelScope
huggingface-cli download damo-vilab/text-to-video-ms-1.7b --resume-download
```

5. **Training**

```shell
python train.py --config configs/JointTuner/bear_plushie_person_playing_flute.yaml
```


6. **Inference**

```shell
bash run_infer.sh
```


### CogVideo-based models (CogVideoX-2B/5B)

1. **Enter the specified project**

```bash
cd examples/CogVideo
```

2. **GPU Requirements**

- For CogVideoX-2B, the training and inference are conducted on 1 RTX 4090 GPU (24GB VRAM)

- For CogVideoX-5B, the training and inference are conducted on 1 A100 GPU (40GB VRAM)

3. **Environment**

```shell
cd examples/CogVideo

conda create -n jointtuner-cog python=3.10
conda activate jointtuner-cog

pip install -r requirements.txt
```

4. **Downloading pretrained models**

```shell
# For CogVideoX-2B 
huggingface-cli download zai-org/CogVideoX-2b --resume-download

# For CogVideoX-5B
huggingface-cli download zai-org/CogVideoX-5b --resume-download
```

5. **Training**

```shell
cd finetune
bash train_jointtuner.sh
```


6. **Inference**

```shell
bash run_infer.sh
```

### Evaluate

1. **Environment**

```shell
cd metrics

conda create -n jointtuner-eval python=3.10
conda activate jointtuner-eval

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/Vchitect/VBench.git

pip install -r requirements.txt
```

2. **Modify vbench output format**

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

3. **Install VBench**
```shell
cd VBench

pip install .
```

4. **Downloading pretrained models**

```shell
# For CogVideoX-2B 
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --resume-download

# For CogVideoX-5B
huggingface-cli download yuvalkirstain/PickScore_v1 --resume-download
```

5. **Evaluating**

```shell
bash eval_folder.sh
```


## 💻 GPU Requirements

- **JointTuner based on ZeroScope or ModelScope**: Training and inference are performed on a single **RTX 4090 GPU** with **24GB VRAM**. 

- **JointTuner based on CogVideoX-2B**: Training and inference are performed on a single **RTX 4090 GPU** with **24GB VRAM**. 

- **JointTuner based on CogVideoX-5B**: Training and inference are performed on a single **NVIDIA A100 GPU** with **40GB VRAM**. 

## 📚 Citation


```bibtex
@article{chen2025jointtuner,
  title={JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation},
  author={Fangda Chen and Shanshan Zhao and Chuanfu Xu and Long Lan},
  journal={arXiv preprint arXiv:2503.23951},
  year={2025}
}
```

## ❤️ Thanks
- This code builds on [diffusers](https://github.com/huggingface/diffusers), [MotionDirector](https://github.com/showlab/MotionDirector), [CogVideo](https://github.com/zai-org/CogVideo) and [VBench]([VBench/prompts at master · Vchitect/VBench](https://github.com/Vchitect/VBench)). Thanks for open-sourcing!
