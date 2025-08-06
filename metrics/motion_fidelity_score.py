import time

import torch
from einops import rearrange
from moviepy.editor import VideoFileClip
import numpy as np
import cv2

from tools import get_all_video_files, calculate_average_of_dicts


def adjust_resolution_to_numpy(source_video, target_video):
    with VideoFileClip(target_video) as target_clip:
        target_width, target_height = target_clip.size
        target_array = np.array([frame for frame in target_clip.iter_frames()])

    with VideoFileClip(source_video) as source_clip:
        resized_clip = source_clip.resize((target_width, target_height))

        resized_source_array = np.array([frame for frame in resized_clip.iter_frames()])

    return resized_source_array, target_array


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)


def eval_per_video(cotracker_model_path, gen_video_path, refer_video_path, metrics, device=None):
    per_video_res = {}

    if device is not None:
        device = torch.device(f'cuda:{device}' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device('cpu')

    model = torch.hub.load(cotracker_model_path, "cotracker3_offline").to(device)
    model = model.to(device)

    if 'motion-fidelity' in metrics:
        print('*' * 10, 'Calculating Motion Fidelity Score', '*' * 10)
        print(gen_video_path)
        print(refer_video_path)

        edit_tracklets = get_tracklets(model, gen_video_path, device=device)
        original_tracklets = get_tracklets(model, refer_video_path, device=device)
        similarity_matrix = get_similarity_matrix(edit_tracklets, original_tracklets)
        score = calculate_score(similarity_matrix)
        per_video_res = {'motion-fidelity': score*100}

    return per_video_res

def eval_video_folder(cotracker_model_path, video_folder, refer_video_path, metrics, per_samples, device=None):
    video_path_list = get_all_video_files(video_folder, suffix='mp4')
    assert len(video_path_list) == per_samples, f"Need {per_samples} sample, but get {len(video_path_list)} sample"
    folder_res = []
    for video_path in video_path_list:
        res = eval_per_video(cotracker_model_path, video_path, refer_video_path, metrics, device)
        print(res)
        folder_res.append(res)

    folder_video_res = calculate_average_of_dicts(folder_res)

    return folder_video_res

def get_similarity_matrix(tracklets1, tracklets2):
    displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

    displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
    displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)

    if displacements1.shape[1] < displacements2.shape[1]:
        pad_num = displacements2.shape[1] - displacements1.shape[1]
        pad_tensor = torch.zeros(displacements1.shape[0], pad_num, displacements1.shape[2], device=displacements1.device)
        displacements1 = torch.cat((displacements1, pad_tensor), dim=1)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
    return similarity_matrix

def calculate_score(similarity_matrix):
    similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
    
    # for each row find the most similar element
    max_similarity, _ = similarity_matrix_eye.max(dim=1)
    
    # Ensure no NaN values are present
    max_similarity = torch.nan_to_num(max_similarity, nan=0.0)  # Replace NaN with 0
    
    average_score = max_similarity.mean()
    return average_score.item()

def get_trackletsv1(model, video_path, mask, device):
    video = read_video_from_path(video_path)  # (f,h,w,c)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)
    pred_tracks_small, pred_visibility_small = model(video, grid_size=55, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small

def get_tracklets(cotracker, video_path, device):
    video = read_video_from_path(video_path)  # (f,h,w,c)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)
    # Run Offline CoTracker:
    grid_size = 55
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)  # B T N 2,  B T N 1
    pred_tracks = rearrange(pred_tracks, "b t l c -> (b l) t c ")
    return pred_tracks

