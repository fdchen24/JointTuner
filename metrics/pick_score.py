import numpy as np
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

from tools import extract_caption, split_video_to_rgb_images, get_all_video_files, calculate_average_of_dicts


def eval_per_video(processor_name_or_path, model_name_or_path, metrics, video_path, video_type, target_fps=8, device=None, video_caption=None):
    per_video_res = {}

    if video_caption is None:
        video_caption, step = extract_caption(video_path, video_type)
    else:
        step = -1

    frame_list = split_video_to_rgb_images(video_path, target_fps)
    num_frames = len(frame_list)

    if device is not None:
        device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).eval().to(device)

    if 'pick-score' in metrics:
        print('*' * 10, 'Calculating Pick Score', '*' * 10)
        print(step, video_caption)
        print(video_path)
        score = calculate_pick_score(processor, model, video_caption, frame_list, device)

        per_video_res = {'pick-score': score}

    return per_video_res

def eval_video_folder(processor_name_or_path, model_name_or_path, metrics, video_folder, video_type, per_samples, target_fps=8, device=None, video_caption_file=None):
    video_path_list = get_all_video_files(video_folder, suffix='mp4')
    video_path_list.sort()
    assert len(video_path_list) == per_samples, f"Need {per_samples} sample, but got {len(video_path_list)} sample"

    folder_res = []
    
    if video_caption_file:
        with open(video_caption_file, 'r') as f:
            video_captions = f.readlines()
        video_captions = [caption.strip() for caption in video_captions]
        
        if len(video_captions) != len(video_path_list):
            raise ValueError("The number of video paths and captions do not match")
    else:
        video_captions = [None] * len(video_path_list)

    for video_path, video_caption in zip(video_path_list, video_captions):
        res = eval_per_video(processor_name_or_path, model_name_or_path, metrics, video_path, video_type, target_fps, device, video_caption)
        print(res)
        folder_res.append(res)
    
    folder_video_res = calculate_average_of_dicts(folder_res)
    print('folder_video_res:', folder_video_res)

    return folder_video_res


def calculate_pick_score(processor, model, prompt, images, device=None):
    images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]

    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        scores = scores.cpu().numpy()

        mean_score = np.mean(scores)

    return mean_score