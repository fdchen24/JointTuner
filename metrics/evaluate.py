import argparse
from datetime import datetime
import os
import pandas as pd

import clip_score
import pick_score
import vbench_score
import motion_fidelity_score
from tools import ensure_dir, read_csv_line_num

os.environ["TOKENIZERS_PARALLELISM"] = "True"

type2metric = {'fid-img': 'FID-Img', 'fid': 'FID', 'mae': 'MAE', "dtssd": "dtSSD",
               'fid-vid': "FVD-3DRN50", 'fvd': "FVD-3DInception", 'is': 'IS',
               'l1': 'L1', 'ssim': 'SSIM', 'lpips': 'LPIPS', 'psnr': 'PSNR',
               'clip-text': 'CLIP-Text', 'clip-image': 'CLIP-Image',
               'temp-con': 'Temporal-Consistency', 'pick-score': 'Pick-Score',
               'subject_consistency': 'Subject-Consistency','background_consistency': 'Background-Consistency',
               'motion_smoothness': 'Motion-Smoothness', 'dynamic_degree': 'Dynamic-Degree',
               'aesthetic_quality': 'Aesthetic-Quality', 'imaging_quality': 'Imaging-Quality',
               'motion-fidelity': 'Motion-Fidelity'}

vbench_metric = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Params')
    parser.add_argument('-f','--file', type=str, required=True, help='The folder or file to evaluate.')
    parser.add_argument('-m', '--model', type=str, default='JointTuner', help='The evaluated model.')
    parser.add_argument('-data', '--dataset', type=str, default='JointTuner', help='The benchmark to evaluate.')
    parser.add_argument('-ri', '--refer_image_path', type=str, help='The path of the reference image to evaluate.')
    parser.add_argument('-rv', '--refer_video_path', type=str, help='The path of the reference video to evaluate.')
    parser.add_argument('-e','--exp_name', type=str, default='', help='The experiment name.')
    parser.add_argument('-clip', '--clip_model_path', type=str, default='ViT-B/32', help='The name of the CLIP model.')
    parser.add_argument('-pickp', '--pick_processor_path', type=str, default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', help='The path of the processor for Pick Score.')
    parser.add_argument('-pickm', '--pick_model_path', type=str, default='yuvalkirstain/PickScore_v1', help='The path of the model for Pick Score.')
    parser.add_argument('-cotrackerm', '--cotracker_model_path', type=str, default='"facebookresearch/co-tracker"', help='The path of the model for Motion Fidelity Score.')
    parser.add_argument('-ftype', '--file_type', type=str, default='folder', help='["folder", "file"]')
    parser.add_argument('-met', '--metrics', nargs='+', default=['clip-text', 'clip-image', 'motion-fidelity', 'pick-score', 'vbench'], help='Multiple metric types to evaluate.')  # all: ['clip-text', 'clip-image', 'temp-con', 'pick-score', 'vbench', 'motion-fidelity']
    parser.add_argument('-fps', '--target_fps', type=int, default=8, help='The number of extract frames per second in the video.')
    parser.add_argument('-vtype', '--video_type', type=str, default='infer', help='The type of the video, ["infer", "val"].')
    parser.add_argument('-d', '--device', type=int, default=0, help='The device id.')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='The batch size.')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='The number of workers.')
    parser.add_argument('-per', '--per_samples', type=int, default=4, help='The number of per sample.')
    parser.add_argument('-c', '--video_caption', type=str, default=None, help='The path of the video caption file.')

    return parser.parse_args()

def exp_record(record_path, args, res):
    params = vars(args)
    dic = dict()
    dic['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    dic['model'] = params['model']
    dic['dataset'] = params['dataset']
    dic['exp_name'] = params['exp_name']
    dic['conclusion'] = ''
    dic['file'] = params['file'].split('/')[-1]
    dic['refer_image_path'] = params['refer_image_path']
    dic['file_type'] = params['file_type']
    file_type = params['file_type']

    metrics = ['Average-Score']
    for m in params['metrics']:
        if 'vbench' == m:
            metrics.extend([type2metric[type] for type in vbench_metric])
        else:
            metrics.append(type2metric[m])

    if file_type != 'step':
        for metric in metrics:
            dic[metric] = res[metric]
        dic['step'] = -1
        record_keys = ['video_type', 'target_fps']
        for key in record_keys:
            dic[key] = params[key]


        df = pd.DataFrame(dic, index=[0])
        exist_title = read_csv_line_num(record_path)
        df.to_csv(record_path, index=True, mode='a+', header=False if exist_title else True)
    else:
        for step, step_res in res.items():
            dic['step'] = step
            for metric in metrics:
                dic[metric] = step_res[metric]
            record_keys = ['video_type', 'target_fps']
            for key in record_keys:
                dic[key] = params[key]

            df = pd.DataFrame(dic, index=[0])
            exist_title = read_csv_line_num(record_path)
            df.to_csv(record_path, index=True, mode='a+', header=False if exist_title else True)


def evaluate(args):
    file_type = args.file_type
    file = args.file
    clip_model_path = args.clip_model_path
    metrics = args.metrics
    target_fps = args.target_fps
    video_type = args.video_type
    refer_image_path = args.refer_image_path
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    per_samples = args.per_samples
    pick_processor_path = args.pick_processor_path
    pick_model_path = args.pick_model_path
    cotracker_model_path = args.cotracker_model_path
    refer_video_path = args.refer_video_path
    video_caption = args.video_caption
    print(args)

    metric_res = {}
    if file_type == 'folder':
        """ Output: res_dict """
        pick_res = pick_score.eval_video_folder(pick_processor_path, pick_model_path, metrics, file, video_type, per_samples, target_fps, device, video_caption)
        metric_res.update(pick_res)
        clip_res = clip_score.eval_video_folder(clip_model_path, metrics, file, video_type, per_samples, target_fps, refer_image_path, device, batch_size, num_workers, video_caption)
        metric_res.update(clip_res)
        vbench_res = vbench_score.eval_video_folder(metrics, file, video_type, per_samples, device, video_caption)
        metric_res.update(vbench_res)
        motion_fidelity_res = motion_fidelity_score.eval_video_folder(cotracker_model_path, file, refer_video_path, metrics, per_samples, device)
        metric_res.update(motion_fidelity_res)
        for key in metrics:
            if key == 'vbench':
                for m in vbench_metric:
                    metric_res[type2metric[m]] = metric_res.pop(m)
            else:
                metric_res[type2metric[key]] = metric_res.pop(key)
    elif file_type == 'file':
        """ Output: res_dict """
        pick_res = pick_score.eval_per_video(pick_processor_path, pick_model_path, metrics, file, video_type, target_fps, device, video_caption)
        metric_res.update(pick_res)
        clip_res = clip_score.eval_per_video(clip_model_path, metrics, file, video_type, target_fps, refer_image_path, device, batch_size, num_workers, video_caption)
        metric_res.update(clip_res)
        vbench_res = vbench_score.eval_per_video(metrics, file, video_type, device, video_caption)
        metric_res.update(vbench_res)
        motion_fidelity_res = motion_fidelity_score.eval_per_video(cotracker_model_path, file, refer_video_path, metrics, device)
        metric_res.update(motion_fidelity_res)
        for key in metrics:
            if key == 'vbench':
                for m in vbench_metric:
                    metric_res[type2metric[m]] = metric_res.pop(m)
            else:
                metric_res[type2metric[key]] = metric_res.pop(key)

    average_score = sum(metric_res.values()) / len(metric_res)

    metric_res["Average-Score"] = average_score

    print(metric_res)

    return metric_res



if __name__ == '__main__':
    args = parse_args()
    res = evaluate(args)
    prefix_dir = 'result/'
    ensure_dir(prefix_dir)
    record_path = os.path.join(prefix_dir, 'exp_record.csv')
    exp_record(record_path, args, res)