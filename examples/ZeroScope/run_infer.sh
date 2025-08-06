#!/bin/bash

lora_path="outputs/train/JointTuner/bear_plushie_person_playing_flute/train_xxx/lora"

python infer.py \
    --model "cerspense/zeroscope_v2_576w" \
    --prompt "A bear plushie playing the flute on the grass." \
    --output_dir "outputs/inference/JointTuner/bear_plushie_person_playing_flute" \
    --lora_path $lora_path \
    --seed 696886