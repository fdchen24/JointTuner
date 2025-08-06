from vbench import VBench
import torch

from tools import extract_caption, get_all_video_files, calculate_average_of_dicts

def eval_per_video(metrics, video_path, video_type, device=None, video_caption=None):
    per_video_res = {}
    
    if video_caption is None:
        video_caption, step = extract_caption(video_path, video_type)
    else:
        step = -1

    if device is not None:
        device = torch.device(f'cuda:{device}' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device('cpu')

    if 'vbench' in metrics:
        print('*' * 10, 'Calculating Vbench Score', '*' * 10)
        print(f"Step: {step}, Caption: {video_caption}")
        print(f"Video Path: {video_path}")
        per_video_res = calculate_vbench_score([video_caption], video_path, device)

    return per_video_res

def eval_video_folder(metrics, video_folder, video_type, per_samples, device=None, video_caption=None):
    video_path_list = get_all_video_files(video_folder, suffix='mp4')
    assert len(video_path_list) == per_samples, f"Need {per_samples} samples, but got {len(video_path_list)}"
    
    if video_caption is not None:
        with open(video_caption, 'r', encoding='utf-8') as f:
            captions = [line.strip() for line in f.readlines()]
        assert len(captions) == len(video_path_list), f"Caption count ({len(captions)}) does not match video count ({len(video_path_list)})"
        video_path_list.sort()
    else:
        captions = [None] * len(video_path_list)

    folder_res = []
    for video_path, caption in zip(video_path_list, captions):
        res = eval_per_video(metrics, video_path, video_type, device, caption)
        print(res)
        folder_res.append(res)

    folder_video_res = calculate_average_of_dicts(folder_res)
    print('folder_video_res:', folder_video_res)

    return folder_video_res

def calculate_vbench_score(prompt_list, videos_path, device=None):
    vbench = VBench(device, 'metrics/VBench/vbench', '')
    scores = vbench.evaluate(
        videos_path = videos_path,
        prompt_list = prompt_list,
        name = 'calculate_vbench_metrics',
        dimension_list = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
        mode='custom_input'
    )

    scores = {key:item[0]*100 for key, item in scores.items()}

    return scores