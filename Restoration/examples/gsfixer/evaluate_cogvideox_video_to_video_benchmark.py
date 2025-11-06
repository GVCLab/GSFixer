"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import sys
sys.path.append('./vggt')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, unproject_depth_map_to_point_map_torch
from vggt.heads.dpt_head import custom_interpolate

import argparse
import os
import shutil
import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from packaging import version
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None


from transformers import T5EncoderModel, T5Tokenizer, AutoModel
from transformers.utils import ContextManagers
import datasets
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from crosstransformer3d import CogVideoXCrossTransformer3DModel

from pipeline_cogvideox_video2video_control import CogVideoXVideoToVideoControlPipeline
from diffusers.utils import export_to_video

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from utils.discrete_sampler import DiscreteSampling
from utils.fp8_optimization import convert_weight_dtype_wrapper
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from PIL import Image
import json
from torch.utils.data.dataset import Dataset
from scipy.spatial.transform import Rotation as R
from image_utils import psnr
from loss_utils import ssim


class ImageVideoControlDataset_val(Dataset):
    def __init__(
            self,
            scene_txt='',
            base_folders=None,
            ref_folders=None,
            num_views='3',
            save_root=None,
    ):
        self.base_folder = base_folders
        self.ref_folders = ref_folders
        self.base_scenes = set(os.listdir(self.base_folder))
        self.ref_scenes = set(os.listdir(self.ref_folders))
        os.makedirs(save_root, exist_ok=True)
        self.save_root = save_root
        self.scenes = []
        with open(scene_txt, 'r') as file:
            for line in file:
                directory_path = line.strip()
                self.scenes.append(directory_path)
        self.scenes = sorted(self.scenes)
        self.channels = 3
        self.ori_width = 960
        self.ori_height = 540
        self.width = 720
        self.height = 480
        sample_frames = 49
        self.sample_frames = sample_frames
        self.samples = []
        self.valid_scenes = []
        view_num = num_views
        self.view_num = view_num
        for scene in self.scenes:
            scene_path = os.path.join(self.base_folder, scene, 'lr', view_num, 'train_'+ view_num)
            if not os.path.exists(scene_path):
                # skip scene 280abf7bd93b81b077af1db638229dbb09869052fa0b7b57c81c94d2db893829 frame_00026.png missing
                # skip scene 5c8dafad7d782c76ffad8c14e9e1244ce2b83aa12324c54a3cc10176964acf04 frame_00077.png~frame_00080.png missing
                # skip scene 918c8dad730c3b804306c5da8486124be4aa0612e85fb825338fd350c912e1b0 frame_00072.png~frame_00076.png missing
                print("skip ", scene_path)
                continue
            self.valid_scenes.append(scene)
            frames = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path)])
            if len(frames) < sample_frames:
                print("skip ", scene_path)
                continue
            ref_indices = [index for index, value in enumerate(frames) if '_ref' in value]
         
            for start, end in zip(ref_indices, ref_indices[1:] + [None]):
                if end is None:
                    continue
                if end is not None:
                    if end - start != sample_frames:
                        if start + sample_frames > len(frames):
                            sample = frames[-sample_frames:]
                        else:
                            if end-start > sample_frames:
                                sample = frames[start:(end+1)]
                                sample = self.uniform_sample_with_fixed_count(sample, sample_frames)
                            else:
                                sample = frames[start: (start + sample_frames)]
                    else:
                        sample = frames[start:(end+1)]
                if len(sample) != sample_frames:
                    print("Error len: ", len(sample), sample_frames)
                self.samples.append(sample)

    def uniform_sample_with_fixed_count(self, lst, count):
        sampled_list = [lst[0]]
        remaining_count = count - 2
        sublist = lst[1:-1]
        interval = len(sublist) // remaining_count
        start = int((len(lst) - interval * (remaining_count-1)) / 2)
        
        for i in range(remaining_count):
            index = (start + i * interval) % len(sublist)
            sampled_list.append(sublist[index])

        sampled_list.append(lst[-1])
        return sampled_list
    
    def open_image(self, file_name):
        name, _ = os.path.splitext(file_name)
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        for ext in possible_extensions:
            full_path = name + ext
            full_path_upper = name + ext.upper()
            if os.path.isfile(full_path):
                return Image.open(full_path)
            elif os.path.isfile(full_path_upper):
                return Image.open(full_path_upper)
        raise FileNotFoundError("No image file found for {}".format(file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        selected_frames = self.samples[idx]
        scene_path = selected_frames[0]
        path_parts = scene_path.split(os.sep)
        chosen_scene = path_parts[-5]
  
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        condition_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        for i, frame_path in enumerate(selected_frames):
            frame_name = os.path.basename(frame_path)
            frame_path = os.path.join(self.base_folder, chosen_scene, 'lr', self.view_num, f"train_{self.view_num}", frame_name)
            
            img = Image.open(frame_path)
            img = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img)).float()

            if i == 0:
                self.height = img.height
                self.width = img.width
                pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
                condition_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

            img_normalized = img_tensor / 127.5 - 1
            img_normalized = img_normalized.permute(2, 0, 1)
            condition_pixel_values[i] = img_normalized

        select_frames_name = []
        ref_first_last_image_path = [] # for vggt
        for i, frame_path in enumerate(selected_frames):
            frame_name = os.path.basename(frame_path)
            ref_frame_name = frame_name.replace('_ref', '')
            gt_frame_path = os.path.join(self.ref_folders, chosen_scene, 'nerfstudio', 'images_4', ref_frame_name)

            save_gt_root = os.path.join(self.save_root, chosen_scene, 'gt')
            os.makedirs(save_gt_root, exist_ok=True)
            save_gt_name = os.path.basename(gt_frame_path)
            select_frames_name.append(save_gt_name)
            save_gt_frame_path = os.path.join(save_gt_root, save_gt_name)
            shutil.copy(gt_frame_path, save_gt_frame_path)

            img = Image.open(gt_frame_path)
            img = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img)).float()
            img_normalized = img_tensor / 127.5 - 1
            img_normalized = img_normalized.permute(2, 0, 1)
            pixel_values[i] = img_normalized
            
            if 'ref' in frame_name:
                ref_first_last_image_path.append(img)
        vggt_images = load_and_preprocess_images_(ref_first_last_image_path)

        pixel_values = pixel_values.permute(1, 0, 2, 3)
        condition_pixel_values = condition_pixel_values.permute(1, 0, 2, 3)

        # the first frame and last frame are reference view
        condition_pixel_values[:, 0:1, :, :] = pixel_values[:, 0:1, :, :]
        condition_pixel_values[:, -1, :, :] = pixel_values[:, -1, :, :]

        caption_path = os.path.join(self.base_folder, chosen_scene, 'lr', self.view_num, 'captions.json')
        with open(caption_path) as f:
            caption = json.load(f)  

        pixel_values = pixel_values.permute(1, 0, 2, 3)
        condition_pixel_values = condition_pixel_values.permute(1, 0, 2, 3)
        ref_first_last_pixel_values = torch.cat([pixel_values[0].unsqueeze(0), pixel_values[-1].unsqueeze(0)], dim=0)

        data = {'pixel_values': pixel_values, 'text': caption, 'control_pixel_values': condition_pixel_values, 'ref_first_last_pixel_values': ref_first_last_pixel_values, 
                    'ref_first_last_image_path': vggt_images, "image_paths": select_frames_name, "scene_name": chosen_scene}
        return data


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DL3DV-Res.")

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--dinov2_ckpt",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--vggt_ckpt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--base_folder",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--ref_folders",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num_views",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seed = 43
    weight_dtype = torch.bfloat16
    num_inference_steps = 50
    guidance_scale = 6.0
    sample_size = [480, 720]
    negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
    outpath = args.outpath
    model_name = args.model_name
    transformer_path = args.transformer_path
    dinov2_path = args.dinov2_ckpt
    vggt_path = args.vggt_ckpt
    device = args.device


    transformer = CogVideoXCrossTransformer3DModel.from_pretrained(transformer_path).to(
        weight_dtype
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name, 
        subfolder="vae"
    ).to(weight_dtype)

    tokenizer = T5Tokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )

    image_encoder = AutoModel.from_pretrained(dinov2_path)
    image_encoder.eval()
    image_encoder.to(device)

    vggt = VGGT()
    vggt.load_state_dict(torch.load(vggt_path))
    vggt.eval()
    vggt = vggt.to(device)

    scheduler = DDIMScheduler.from_pretrained(
        model_name, 
        subfolder="scheduler"
    )

    pipeline = CogVideoXVideoToVideoControlPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device=device).manual_seed(seed)
    train_dataset = ImageVideoControlDataset_val(
        scene_txt=args.scene_name,
        base_folders=args.base_folder,
        ref_folders=args.ref_folders,
        num_views=args.num_views,
        save_root=outpath,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        persistent_workers=True,
        num_workers=8,
    )

    for step, batch in enumerate(train_dataloader):
        pixel_values = batch["pixel_values"].to(weight_dtype)
        control_pixel_values = batch["control_pixel_values"].to(weight_dtype)
        ref_first_last_pixel_values = batch["ref_first_last_pixel_values"].to(weight_dtype)
        vggt_images = batch["ref_first_last_image_path"].to(device)
        prompt = batch['text']
        image_paths = batch["image_paths"]
        scene_name = batch["scene_name"][0]
        with torch.no_grad():
            # dinov2 features [batch, sequence_length, feature_dim]
            dino_latents = image_encoder(vggt_images.squeeze(0)).last_hidden_state[:, 5:, :].to(weight_dtype)
            # vggt features [batch, sequence_length, feature_dim]
            output_list, patch_start_idx = vggt.aggregator.forward(vggt_images)
            vggt_latents = output_list[-1][:, :, patch_start_idx:, :].squeeze(0).to(weight_dtype)
            ref_images = ref_first_last_pixel_values.permute(0, 2, 1, 3, 4)

        sample = pipeline(
            ref_image=ref_images,
            ref_dino_latents=dino_latents,
            ref_vggt_latents=vggt_latents,
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            control_video=control_pixel_values.permute(0, 2, 1, 3, 4),
        ).frames[0]

        save_render_root = os.path.join(outpath, scene_name, 'render')
        os.makedirs(save_render_root, exist_ok=True)
        save_fixed_root = os.path.join(outpath, scene_name, 'fixed')
        os.makedirs(save_fixed_root, exist_ok=True)
   
        for frame_id in range(len(image_paths)):
            frame_name = image_paths[frame_id][0]
            fixed_frame = sample[frame_id]
            fixed_frame.save(os.path.join(save_fixed_root, frame_name))

            render_frame = (control_pixel_values[0][frame_id] + 1.0) / 2
            torchvision.utils.save_image(render_frame, os.path.join(save_render_root, frame_name))

        print(f"finish scene {scene_name}")


if __name__ == "__main__":
    main()
