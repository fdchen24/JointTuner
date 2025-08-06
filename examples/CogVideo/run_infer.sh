#!/bin/bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false


sample="bear_plushie_person_playing_flute"
seed=4399

python cli_demo.py \
  --prompt "A whimsical bear plushie sitting on a lush green meadow, serenely playing a wooden flute under a gentle, golden sunset." \
  --model_path THUDM/CogVideoX-2b \
  --lora_path finetune/outputs/JointTuner/$sample/checkpoint-1000 \
  --output_path outputs/JointTuner/$sample/0-seed-$seed.mp4 \
  --seed $seed \
  --fps 8 \
  --dtype "float16" \
  --lora_alpha 1.0
