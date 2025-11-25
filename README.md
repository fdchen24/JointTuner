
<p align="center">

  <!-- <h2 align="center" style="vertical-align: middle; margin-right: 2px;">
  <img src="asserts/favicon.ico" alt="JointTuner Icon" width="48" height="48" style="vertical-align: middle; margin-right: 2px;">
  JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation
  </h2> -->
  <h2 align="center">
  <div style="display: flex; align-items: flex-start; justify-content: center;">
    <img src="asserts/favicon.ico" alt="JointTuner Icon" width="48" height="48" style="margin-right: 12px; margin-top: 0;">
    <span style="font-size: 1.5em;">JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation</span>
  </div>
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

https://github.com/user-attachments/assets/a1eef2b4-55ba-43a9-998a-864970442244

<p align="center">
<em>JointTuner enables joint optimization of appearance and motion for high-fidelity customized video generation, mitigating concept interference and appearance contamination.</em>
</p>

## 🎯 Task Definition
**Appearance-Motion Combined Customization** of Text-to-Video Diffusion Models: 
Given reference images (for appearance) and reference videos (for motion), the task is to adapt pre-trained text-to-video diffusion models to generate videos that faithfully preserve the desired appearance while accurately replicating the target motion pattern.

## 🔧 Compatibility
JointTuner is architecture-agnostic and supports both:
- **UNet-based models** (e.g., ZeroScope)
- **Diffusion Transformer (DiT)-based models** (e.g., CogVideoX)

## 📰 News
- [2025.08.06] The code for JointTuner, implemented based on UNet (e.g., ZeroScope and ModelScope) and DiT (e.g., CogVideoX), along with the constructed benchmark, has been released.
- [2025.03.31] Paper and project page released.

## ✅ ToDo List
- [ ] Implementations based on WanVideo
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

- **Setup**

```
git clone https://github.com/fdchen24/JointTuner.git
cd JointTuner
```

To get started with JointTuner, simply navigate to the examples directory and choose your preferred base text-to-video model:

- For **UNet-based models** (e.g., ZeroScope or ModelScope), go to:

```bash
cd examples/ZeroScope
```

and follow the instructions in the `README.md`.

- For **DiT-based models** (e.g., CogVideoX-2B or CogVideoX-5B), go to:

```bash
cd examples/CogVideo
```

and refer to the `README.md` for training, and inference details.

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
- This code builds on [diffusers](https://github.com/huggingface/diffusers), [MotionDirector](https://github.com/showlab/MotionDirector) and [CogVideo](https://github.com/zai-org/CogVideo). Thanks for open-sourcing!