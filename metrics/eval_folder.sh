#!/bin/bash

python evaluate.py \
    --file examples/bear_plushie_person_playing_flute \
    --model JointTuner \
    --exp_name JointTuner \
    --refer_image_path ../datasets/JointTuner/subject/bear_plushie/00.jpg \
    --refer_video_path ../datasets/JointTuner/motion_49f/person_playing_flute/00.mp4 \
    --dataset JointTuner \
    --per_samples 1 \
    --file_type folder \
    --video_type infer \
    --target_fps 8 \
    --video_caption examples/bear_plushie_person_playing_flute/caption.txt \