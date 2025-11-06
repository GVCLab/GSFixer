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
import glob
import shutil
import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.checkpoint
import transformers
from transformers import AutoProcessor, Blip2ForConditionalGeneration
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
        "--blip_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--outpath",
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
    width, height = 720, 480
    sample_size = [height, width]
    sample_frames = 49
    negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
    refine_prompt = ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    outpath = args.outpath
    model_name = args.model_name
    transformer_path = args.transformer_path
    dinov2_path = args.dinov2_ckpt
    vggt_path = args.vggt_ckpt
    blip_path = args.blip_path
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
    
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    ).to(device)

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
    
    images_root = args.images_root
    image_names = glob.glob(os.path.join(images_root, "*"))
    image_names = sorted(image_names)
    
    control_pixel_values = torch.empty((sample_frames, 3, height, width))
    ref_first_last_image_path = []
    for i, frame_path in enumerate(image_names):
        img = Image.open(frame_path)
        if i == sample_frames - 1:
            inputs = caption_processor(images=img, return_tensors="pt").to(
                device, torch.float16
            )
            generated_ids = captioner.generate(**inputs)
            generated_text = caption_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            prompt = generated_text + refine_prompt
            prompt = 'statue of liberty in front of the new york state supreme court. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.'
            
        img = img.resize((width, height))
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_normalized = img_tensor / 127.5 - 1
        img_normalized = img_normalized.permute(2, 0, 1)
        control_pixel_values[i] = img_normalized
        
        if i == 0 or i == sample_frames - 1:
            ref_first_last_image_path.append(frame_path)
            
    ref_first_last_pixel_values = torch.cat([control_pixel_values[0].unsqueeze(0), control_pixel_values[-1].unsqueeze(0)], dim=0)
    vggt_images = load_and_preprocess_images(ref_first_last_image_path)
    vggt_images = vggt_images.to(device)
    with torch.no_grad():
        # dinov2 features [batch, sequence_length, feature_dim]
        dino_latents = image_encoder(vggt_images).last_hidden_state[:, 5:, :].to(weight_dtype)
        # vggt features [batch, sequence_length, feature_dim]
        output_list, patch_start_idx = vggt.aggregator.forward(vggt_images.unsqueeze(0))
        vggt_latents = output_list[-1][:, :, patch_start_idx:, :].squeeze(0).to(weight_dtype)
        ref_images = ref_first_last_pixel_values.permute(1, 0, 2, 3).unsqueeze(0)

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
            control_video=control_pixel_values.permute(1, 0, 2, 3).unsqueeze(0),
        ).frames[0]

    scene_name = images_root.split('/')[-1]
    save_render_root = os.path.join(outpath, scene_name, 'render')
    os.makedirs(save_render_root, exist_ok=True)
    save_fixed_root = os.path.join(outpath, scene_name, 'fixed')
    os.makedirs(save_fixed_root, exist_ok=True)

    for frame_id in range(len(image_names)):
        frame_name = image_names[frame_id].split('/')[-1]
        fixed_frame = sample[frame_id]
        fixed_frame.save(os.path.join(save_fixed_root, frame_name))

        render_frame = (control_pixel_values[frame_id] + 1.0) / 2
        torchvision.utils.save_image(render_frame, os.path.join(save_render_root, frame_name))

    print(f"finish scene {scene_name}")


if __name__ == "__main__":
    main()
