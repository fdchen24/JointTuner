# JointTuner based on CogVideoX (CogVideoX-2B or CogVideoX-5B)

# Requirements

- For CogVideoX-2B, the training and inference are conducted on 1 RTX 4090 GPU (24GB VRAM)

- For CogVideoX-5B, the training and inference are conducted on 1 A100 GPU (40GB VRAM)

# Environment

```shell
cd examples/CogVideo

conda create -n jointtuner-cog python=3.10
conda activate jointtuner-cog

pip install -r requirements.txt
```

# Downloading pretrained models

```shell
# For CogVideoX-2B 
huggingface-cli download zai-org/CogVideoX-2b --resume-download

# For CogVideoX-5B
huggingface-cli download zai-org/CogVideoX-5b --resume-download
```

# Training

```shell
cd finetune
bash train_jointtuner.sh
```


# Inference

```shell
bash run_infer.sh
```