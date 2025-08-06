#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

sample="bear_plushie_person_playing_flute"
echo "Sample: $sample"

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-2b"  # "THUDM/CogVideoX-5B"
    --model_name "cogvideox-t2v"
    --model_type "t2v"
    --training_type "jointtuner"
    --rank 128
    --lora_alpha 1
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "outputs/JointTuner/$sample"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "data/JointTuner/$sample"
    --caption_column "prompts.txt"
    --video_column "videos.txt"
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1， 480x720
)

# Training Configuration
TRAIN_ARGS=(
    --train_steps 1000
    --learning_rate 2e-4  # 2e-5 for CogVideoX-2B, 1e-4 for CogVideoX-5B
    --lr_scheduler "constant"
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "fp16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 500 # save checkpoint every x steps
    --checkpointing_limit 20 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "outputs/JointTuner/$sample/checkpoint-500"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "data/JointTuner/$sample"
    --validation_steps 500  # should be multiple of checkpointing_steps
    --validation_prompts "validations.txt"
    --gen_fps 8
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
