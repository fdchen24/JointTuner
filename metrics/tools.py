import os
import re

import cv2
import csv

join = os.path.join

def ensure_dir(dir_path):
    if os.path.isfile(dir_path):
        d = os.path.dirname(dir_path)
    else:
        d = dir_path
    if not os.path.exists(d):
        os.makedirs(d)

def get_subdirs(root_dir):
    return [name for name in os.listdir(root_dir) if os.path.isdir(join(root_dir, name))]

def read_csv_line_num(path):
    with open(path, "a+") as writefile:
        pass
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return len(list(reader))

def extract_caption(video_path, video_type='infer'):
    caption = ''
    step = -1
    if video_type == 'infer':
        extracted_text = video_path.split('/')[-1].split('.')[0].split('_')[:-1]
        caption = ' '.join(extracted_text) + '.'
    else:
        print('Invalid video type')

    if caption == '':
        print(f'{video_path} No caption found for {video_type}.')
    return caption, step

def calculate_average_of_dicts(dict_list):
    if not dict_list:
        return {}

    sum_dict = {}
    count_dict = {}

    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] = sum_dict.get(key, 0) + value
            count_dict[key] = count_dict.get(key, 0) + 1

    average_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return average_dict

def split_video_to_rgb_images(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Can not open video file.")
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, int(original_fps / target_fps))

    rgb_images = []

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_frame)


        frame_index += 1

    cap.release()

    return rgb_images


def get_all_video_files(directory, suffix='mp4'):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(f".{suffix}"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files