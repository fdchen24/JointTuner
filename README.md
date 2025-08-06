# JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation

# Requirements

- **JointTuner based on ZeroScope or ModelScope**:
  - Training and inference are performed on a single **RTX 4090 GPU** with **24GB VRAM**. 

- **JointTuner based on CogVideoX-2B**:
  - Training and inference are performed on a single **RTX 4090 GPU** with **24GB VRAM**. 

- **JointTuner based on CogVideoX-5B**:
  - Training and inference are performed on a single **NVIDIA A100 GPU** with **40GB VRAM**. 


# Project Directory Overview

## datasets
Contains various datasets of our proposed benchmark. This directory includes three subfolders: `subject`, `motion_16f` and `motion_49f`.

- `subject`: Contains data files related to subjects.
- `motion_16f`: Contains data videos related to motions with 16 frames.
- `motion_49f`: Contains data videos related to motions with 49 frames.

## examples
This directory includes implementations of JointTuner on different text-to-video models:
- `ZeroScope` and `ModelScope` share a common codebase. For specific implementation steps, refer to the `README.md` in the `ZeroScope` folder.
- In `CogVideo`, both `CogVideoX-2B` and `CogVideoX-5B` share a common codebase. The implementation in the `CogVideo` folder can be found with the respective `README.md` for detailed instructions.

## metrics
Contains the evaluation code of our proposed benchmark, used to assess the performance of the models. For detailed implementation steps, refer to the `README.md` in the respective folder.
