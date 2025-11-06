#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser
import math
import numpy as np
import torchvision


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


from PIL import Image
from torchvision import transforms
transform = transforms.ToPILImage()
def save_gif_video(videos, save_path):
    video_frames_render = []
    for id in range(len(videos)):
        img = videos[id].squeeze(0).float()
        # img = transform((img + 1.0) / 2)
        img = transform(img)
        video_frames_render.append(np.array(img))
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in video_frames_render]
    pil_frames[0].save(save_path.replace('.mp4', '.gif'),
                        format='GIF',
                        append_images=pil_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0)


import cv2
def save_mp4_video(videos, save_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    frame_tensor = videos[0].squeeze(0).cpu().float()  
    frame = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8) 
    height, width, _ = frame.shape
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    for video in videos:
        img = video.squeeze(0).cpu().float() 
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    out.release()


def evaluate_renders(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        # try:
        scene_dir_name = str(scene_dir.split("/")[-1])
        print("Scene:", scene_dir)
        full_dict[scene_dir_name] = {}
        per_view_dict[scene_dir_name] = {}
        full_dict_polytopeonly[scene_dir_name] = {}
        per_view_dict_polytopeonly[scene_dir_name] = {}

        # test_dir = Path(scene_dir) / "test"
        test_dir = Path(scene_dir)
        print(test_dir)
        method = test_dir
        # for method in os.listdir(test_dir):
        print("Method:", method)

        full_dict[scene_dir_name][method] = {}
        per_view_dict[scene_dir_name][method] = {}
        full_dict_polytopeonly[scene_dir_name][method] = {}
        per_view_dict_polytopeonly[scene_dir_name][method] = {}

        method_dir = method  # test_dir / method
        gt_dir = method_dir / "gt"
        # renders_dir = method_dir / "renders"
        renders_dir = method_dir / "render"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        if renders[0].shape != gts[0].shape:
            for i in range(len(renders)):
                renders[i] = torch.nn.functional.interpolate(
                        renders[i],
                        size=(gts[i].shape[2], gts[i].shape[3]),
                        mode="bilinear") 

        # save resize 3dgs render image
        scene_dir_resize_root = os.path.join(scene_dir, 'resized_render')
        os.makedirs(scene_dir_resize_root, exist_ok=True)
        for i in range(len(image_names)):
            save_fixed_image_path = os.path.join(scene_dir_resize_root, image_names[i])
            torchvision.utils.save_image(renders[i], save_fixed_image_path)

        num_frames = 49
        for i in range(math.ceil(len(renders) / num_frames)):
            fixed_frames = renders[i*(num_frames - 1): num_frames + i * (num_frames - 1)]
            save_gif_video(fixed_frames, os.path.join(scene_dir, f"3dgs_render_{i}.mp4"))
            save_mp4_video(fixed_frames, os.path.join(scene_dir, f"3dgs_render_{i}.mp4"))

        



def evaluate_fix(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        # try:
        scene_dir_name = str(scene_dir.split("/")[-1])
        print("Scene:", scene_dir)
        full_dict[scene_dir_name] = {}
        per_view_dict[scene_dir_name] = {}
        full_dict_polytopeonly[scene_dir_name] = {}
        per_view_dict_polytopeonly[scene_dir_name] = {}

        # test_dir = Path(scene_dir) / "test"
        test_dir = Path(scene_dir)
        print(test_dir)
        method = test_dir
        # for method in os.listdir(test_dir):
        print("Method:", method)

        full_dict[scene_dir_name][method] = {}
        per_view_dict[scene_dir_name][method] = {}
        full_dict_polytopeonly[scene_dir_name][method] = {}
        per_view_dict_polytopeonly[scene_dir_name][method] = {}

        method_dir = method  # test_dir / method
        gt_dir = method_dir / "gt"
        # renders_dir = method_dir / "renders"
        renders_dir = method_dir / "fixed"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        if renders[0].shape != gts[0].shape:
            for i in range(len(renders)):
                renders[i] = torch.nn.functional.interpolate(
                        renders[i],
                        size=(gts[i].shape[2], gts[i].shape[3]),
                        mode="bilinear") 
        
        # save resize fixed image
        scene_dir_resize_root = os.path.join(scene_dir, 'resized_fixed')
        os.makedirs(scene_dir_resize_root, exist_ok=True)
        for i in range(len(image_names)):
            save_fixed_image_path = os.path.join(scene_dir_resize_root, image_names[i])
            torchvision.utils.save_image(renders[i], save_fixed_image_path)
            

        # save gif and mp4 video
        num_frames = 49
        for i in range(math.ceil(len(renders) / num_frames)):
            fixed_frames = renders[i*(num_frames - 1): num_frames + i * (num_frames - 1)]
            # save_gif_video(fixed_frames, os.path.join(scene_dir, f"gsfixer_fixed_{i}.mp4"))
            save_mp4_video(fixed_frames, os.path.join(scene_dir, f"gsfixer_fixed_{i}.mp4"))

            gt_frames = gts[i*(num_frames - 1): num_frames + i * (num_frames - 1)]
            # save_gif_video(gt_frames, os.path.join(scene_dir, f"gt_{i}.mp4"))
            save_mp4_video(gt_frames, os.path.join(scene_dir, f"gt_{i}.mp4"))
    


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--scene_text",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    scenes = []
    with open(args.scene_text, 'r') as file:
        for line in file:
            directory_path = line.strip()
            scenes.append(directory_path)

    for scene_name in scenes:
        scene_path = os.path.join(args.output_root, scene_name)
        evaluate_fix([scene_path])
        evaluate_renders([scene_path])
