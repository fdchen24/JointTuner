# JointTuner based on ZeroScope (or ModelScope)

# Requirements

The training and inference are conducted on 1 RTX 4090 GPU (24GB VRAM)

# Environment

```shell
cd examples/ZeroScope

conda create -n jointtuner-zs python=3.8
conda activate jointtuner-zs

pip install -r requirements.txt
```

# Downloading pretrained models

```shell
# For ZeroScope 
huggingface-cli download cerspense/zeroscope_v2_576w --resume-download

# For ModelScope
huggingface-cli download damo-vilab/text-to-video-ms-1.7b --resume-download
```

# Training

```shell
python train.py --config configs/JointTuner/bear_plushie_person_playing_flute.yaml
```


# Inference

```shell
bash run_infer.sh
```