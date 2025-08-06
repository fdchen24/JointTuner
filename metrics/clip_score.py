import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import numpy as np

from tools import *


class DummyDataset(Dataset):
    FLAGS = ['img', 'txt']

    def __init__(self, real_data, fake_data,
                 real_flag: str = 'img',
                 fake_flag: str = 'img',
                 transform=None,
                 tokenizer=None) -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.FLAGS, real_flag, fake_flag
            )
        self.real_data = real_data
        self.real_flag = real_flag
        self.fake_data = fake_data
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.real_data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real = self.real_data[index]
        fake = self.fake_data[index]

        if self.transform is not None:
            if self.real_flag == 'img':
                if isinstance(real, np.ndarray):
                    real = Image.fromarray(real)
                real = self.transform(real)
            if self.fake_flag == 'img':
                if isinstance(fake, np.ndarray):
                    fake = Image.fromarray(fake)
                fake = self.transform(fake)

        if self.tokenizer is not None:
            if self.real_flag == 'txt':
                real = self.tokenizer(real, truncate=True).squeeze()
            if self.fake_flag == 'txt':
                fake = self.tokenizer(fake, truncate=True).squeeze()

        sample = dict(real=real, fake=fake)
        return sample


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    for batch_data in dataloader:
        real = batch_data['real']
        real_features = forward_modality(model, real, real_flag)
        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)

        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)

        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]

    return score_acc / sample_num


def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


def clip_score_pipeline(clip_model, preprocess, real_data, fake_data, real_flag, fake_flag, batch_size=8, num_workers=4):
    dataset = DummyDataset(real_data, fake_data,
                           real_flag, fake_flag,
                           transform=preprocess, tokenizer=clip.tokenize)
    dataloader = DataLoader(dataset, batch_size,
                            num_workers=num_workers, pin_memory=True)

    clip_score = calculate_clip_score(dataloader, clip_model,
                                      real_flag, fake_flag)
    clip_score = clip_score.cpu().item()
    return clip_score

def eval_per_video(clip_model_path, metrics, video_path, video_type, target_fps=8, refer_image_path=None, device=None, batch_size=8, num_workers=4, video_caption=None):
    per_video_res = {}

    if video_caption is None:
        video_caption, step = extract_caption(video_path, video_type)
    else:
        step = -1

    frame_list = split_video_to_rgb_images(video_path, target_fps)
    num_frames = len(frame_list)

    if device is not None:
        device = torch.device(f'cuda:{device}' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device('cpu')
    
    clip_model, preprocess = clip.load(clip_model_path, device=device)
    clip_model = clip_model.eval()

    for metric in metrics:
        res = {}
        if metric == 'clip-text':
            """frame - caption"""
            if video_caption is None:
                print("No video caption available")
                continue
            video_caption_list = [video_caption] * num_frames
            real_flag = 'img'
            fake_flag = 'txt'
            print('*' * 10, 'Calculating CLIP Text Score', '*' * 10)
            print(f"Step: {step}, Caption: {video_caption}")
            print(f"Video Path: {video_path}")
            score = clip_score_pipeline(clip_model, preprocess, frame_list, video_caption_list, real_flag, fake_flag, batch_size, num_workers)
            res = {metric: score}
        elif metric == 'clip-image':
            """frame - reference image"""
            if refer_image_path is None:
                print("No reference image provided!")
                continue
            refer_image = Image.open(refer_image_path).convert('RGB')
            refer_image_list = [refer_image] * num_frames
            real_flag = 'img'
            fake_flag = 'img'
            print('*' * 10, 'Calculating CLIP Image Score', '*' * 10)
            print(f"Step: {step}, Caption: {video_caption}")
            print(f"Video Path: {video_path}")
            score = clip_score_pipeline(clip_model, preprocess, frame_list, refer_image_list, real_flag, fake_flag, batch_size, num_workers)
            res = {metric: score}
        elif metric == 'temp-con':
            "temporal consistency: frame - next frame"
            prev_frame_list = frame_list[:num_frames - 1]
            next_frame_list = frame_list[1:]
            real_flag = 'img'
            fake_flag = 'img'
            print('*' * 10, 'Calculating Temp Consistency Score', '*' * 10)
            print(f"Step: {step}, Caption: {video_caption}")
            print(f"Video Path: {video_path}")
            score = clip_score_pipeline(clip_model, preprocess, prev_frame_list, next_frame_list, real_flag, fake_flag, batch_size, num_workers)
            res = {metric: score}

        per_video_res.update(res)

    return per_video_res

def eval_video_folder(clip_model_path, metrics, video_folder, video_type, per_samples, target_fps=8,  refer_image_path=None, device=None, batch_size=8, num_workers=4, video_caption=None):
    video_path_list = get_all_video_files(video_folder, suffix='mp4')
    assert len(video_path_list) == per_samples, f"Need {per_samples} samples, but got {len(video_path_list)} samples"
    
    if video_caption is not None and isinstance(video_caption, str) and os.path.isfile(video_caption):
        with open(video_caption, 'r', encoding='utf-8') as f:
            captions = [line.strip() for line in f.readlines()]
        
        if len(captions) != len(video_path_list):
            raise ValueError(f"Number of captions ({len(captions)}) does not match number of videos ({len(video_path_list)})")
        
        video_path_list_sorted = sorted(video_path_list)
    else:
        captions = [video_caption] * len(video_path_list)
        video_path_list_sorted = video_path_list
    
    folder_res = []
    for video_path, caption in zip(video_path_list_sorted, captions):
        res = eval_per_video(clip_model_path, metrics, video_path, video_type, target_fps, refer_image_path, device, batch_size, num_workers, video_caption=caption)
        print(res)
        folder_res.append(res)

    folder_video_res = calculate_average_of_dicts(folder_res)

    return folder_video_res