import sys
sys.path.append('./gsfixer/vggt')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, unproject_depth_map_to_point_map_torch
from vggt.heads.dpt_head import custom_interpolate

import os
import argparse
import torch
import torchvision
import gc
import glob
import math
import shutil
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image
from gsfixer.cogvideo.pipeline_cogvideox_video2video_control import CogVideoXVideoToVideoControlPipeline
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from gsfixer.cogvideo.crosstransformer3d import CogVideoXCrossTransformer3DModel
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoModel

from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)


def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'}
    )


from PIL import Image
from torchvision import transforms
transform = transforms.ToPILImage()
def save_gif_video(videos, save_path):
    video_frames_render = []
    for id in range(len(videos)):
        img = videos[id].float()
        img = transform((img + 1.0) / 2)
        # img = transform(img)
        video_frames_render.append(np.array(img))
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in video_frames_render]
    pil_frames[0].save(save_path.replace('.mp4', '.gif'),
                        format='GIF',
                        append_images=pil_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0)


class GSFixer:
    def __init__(self, opts, gradio=False):
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(
            opts.blip_path, torch_dtype=torch.float16
        ).to(opts.device)
        # dinov2 model
        self.image_encoder = AutoModel.from_pretrained(opts.dinov2_ckpt)
        self.image_encoder.eval()
        self.image_encoder.to(opts.device)
        # vggt model
        self.vggt = VGGT()
        self.vggt.load_state_dict(torch.load(opts.vggt_ckpt))
        self.vggt.eval()
        self.vggt = self.vggt.to(opts.device)
        self.setup_diffusion(opts)
        self.opts = opts
        if gradio:
            self.opts = opts

    def repair(self, artifact_rgb):
        os.makedirs(self.opts.save_dir, exist_ok=True)
        # artifact_rgb = (artifact_rgb - 0.5) * 2
        repaired_rgb = []
        for i in range(math.ceil(len(artifact_rgb) / self.opts.num_frames)):
            frames = artifact_rgb[i*(self.opts.num_frames - 1): self.opts.num_frames + i * (self.opts.num_frames - 1)]

            prompt = self.get_caption(self.opts, frames[0])

            # control_pixel_values = torch.from_numpy(frames) / 127.5 - 1
            control_pixel_values = (frames - 0.5) * 2
            control_pixel_values = control_pixel_values.permute(1, 0, 2, 3).unsqueeze(0)
            ref_first_last_pixel_values = torch.cat([control_pixel_values[:, :, 0, :, :].unsqueeze(2), control_pixel_values[:, :, -1, :, :].unsqueeze(2)], dim=2)

            ref_first_last_image_path = [] # for vggt
            first_image = control_pixel_values[:, :, 0, :, :].squeeze(0).cpu().clone().permute(1, 2, 0).numpy()
            first_image = (first_image * 255).astype(np.uint8)
            first_image = Image.fromarray(first_image)
            ref_first_last_image_path.append(first_image)
            last_image = control_pixel_values[:, :, -1, :, :].squeeze(0).cpu().clone().permute(1, 2, 0).numpy()
            last_image = (last_image * 255).astype(np.uint8)
            last_image = Image.fromarray(last_image)
            ref_first_last_image_path.append(last_image)
            vggt_images = load_and_preprocess_images_(ref_first_last_image_path).to(self.opts.device)

            dino_latents = self.image_encoder(vggt_images).last_hidden_state[:, 5:, :].to(self.opts.weight_dtype)
            output_list, patch_start_idx = self.vggt.aggregator.forward(vggt_images.unsqueeze(0))
            vggt_latents = output_list[-1][:, :, patch_start_idx:, :].squeeze(0).to(self.opts.weight_dtype)
            
            generator = torch.Generator(device=self.opts.device).manual_seed(self.opts.seed)


            # # save render images
            # index = len([path for path in os.listdir(self.opts.save_dir)]) + 1
            # prefix = str(index).zfill(8)
            # # save_gif_video(control_pixel_values.permute(0, 2, 1, 3, 4)[0], os.path.join(self.opts.save_dir, prefix + "_render.mp4"))
            # for frame_id in range(control_pixel_values.squeeze(0).shape[1]): 
            #     frame = control_pixel_values.squeeze(0)[:, frame_id, :, :].cpu().clone().permute(1, 2, 0).numpy()
            #     frame = ((frame + 1.0) / 2.0 * 255).astype(np.uint8) 
            #     img = Image.fromarray(frame)
            #     index = len([path for path in os.listdir(self.opts.save_dir)]) + 1
            #     prefix = str(index).zfill(8)
            #     img.save(os.path.join(self.opts.save_dir, prefix + "_control_frame_{}.png".format(frame_id)))


            with torch.no_grad():
                sample = self.pipeline(
                    ref_image=ref_first_last_pixel_values,
                    ref_dino_latents=dino_latents,
                    ref_vggt_latents=vggt_latents,
                    prompt=prompt, 
                    negative_prompt=self.opts.negative_prompt,
                    height=self.opts.sample_size[0],
                    width=self.opts.sample_size[1],
                    generator=generator,
                    guidance_scale=self.opts.diffusion_guidance_scale,
                    num_inference_steps=self.opts.diffusion_inference_steps,
                    control_video=control_pixel_values,
                ).frames[0]

            
            # # save fixed video
            # video_path = os.path.join(self.opts.save_dir, prefix + "_enhance.mp4")
            # sample[0].save(
            #     video_path.replace('.mp4', '.gif'),
            #     format='GIF',
            #     save_all=True,
            #     append_images=sample[1:],
            #     duration=500,
            #     loop=0
            # )

            # # save fixed images
            # for frame_id, frame in enumerate(sample):
            #     frame_np = np.array(frame)
            #     if frame_np.dtype != np.uint8:
            #         frame_np = ((frame_np + 1.0) / 2.0 * 255).astype(np.uint8)
            #     img = Image.fromarray(frame_np)
            #     index = len([path for path in os.listdir(self.opts.save_dir)]) + 1
            #     prefix = str(index).zfill(8)
            #     img.save(os.path.join(self.opts.save_dir, prefix + "_fixed_frame_{}.png".format(frame_id)))


            sample = [torch.from_numpy(np.array(sample_view)).unsqueeze(0) / 255.0 for sample_view in sample]
            sample = torch.cat(sample, dim=0)
            repaired_rgb.append(sample)
            
        repaired_rgb_ = repaired_rgb[0]
        for i in range(1, math.ceil(len(artifact_rgb) / self.opts.num_frames)):
            repaired_rgb_ = torch.cat([repaired_rgb_, repaired_rgb[i]], dim=0)

        return repaired_rgb_


    def setup_diffusion(self, opts):
        transformer = CogVideoXCrossTransformer3DModel.from_pretrained_2d(opts.transformer_path).to(
            opts.weight_dtype
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            opts.model_name, subfolder="vae"
        ).to(opts.weight_dtype)
        tokenizer = T5Tokenizer.from_pretrained(
            opts.model_name, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            opts.model_name, subfolder="text_encoder", torch_dtype=opts.weight_dtype
        )
        # Get Scheduler
        scheduler = DDIMScheduler.from_pretrained(
            opts.model_name, subfolder="scheduler"
        )

        self.pipeline = CogVideoXVideoToVideoControlPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        ).to(opts.weight_dtype)

        if opts.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()

    def get_caption(self, opts, image):
        image_array = (image * 255).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(np.transpose(image_array, (1, 2, 0)))
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text + opts.refine_prompt

  
def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')
    parser.add_argument(
        '--out_dir', type=str, default='./experiments/', help='Output dir'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name, use video file name by default',
    )
    parser.add_argument(
        '--seed', type=int, default=43, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--video_length', type=int, default=49, help='Length of the video frames'
    )
    parser.add_argument('--fps', type=int, default=10, help='Fps for saved video')
    parser.add_argument(
        '--stride', type=int, default=1, help='Sampling stride for input video'
    )
    parser.add_argument('--server_name', type=str, help='Server IP address')

    # ## render
    parser.add_argument(
        '--mode', type=str, default='gradual', help='gradual, bullet or direct'
    )
 
    ## diffusion
    parser.add_argument(
        '--low_gpu_memory_mode',
        type=bool,
        default=False,
        help='Enable low GPU memory mode',
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='models/Diffusion_Transformer/CogVideoX-5b-I2V',
        help='Path to the model',
    )
    
    parser.add_argument(
        '--sampler_name',
        type=str,
        choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default='DDIM_Origin',
        help='Choose the sampler',
    )

    parser.add_argument(
        '--transformer_path',
        type=str,
        default="models/Diffusion_Transformer/CogVideoX-5b-I2V/transformer",
        help='Path to the pretrained transformer model',
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[480, 720],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--diffusion_guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--diffusion_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps',
    )
    parser.add_argument(
        '--prompt', type=str, default=None, help='Prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    parser.add_argument('--blip_path', type=str, default="./checkpoints/blip2-opt-2.7b")
    parser.add_argument(
        '--cpu_offload', type=str, default='model', help='CPU offload strategy'
    )
    parser.add_argument(
        '--depth_inference_steps', type=int, default=5, help='Number of inference steps'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--window_size', type=int, default=110, help='Window size for processing'
    )
    parser.add_argument(
        '--overlap', type=int, default=25, help='Overlap size for processing'
    )
    parser.add_argument(
        '--max_res', type=int, default=1024, help='Maximum resolution for processing'
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()  # infer config.py
    opts = parser.parse_args()
    opts.weight_dtype = torch.bfloat16

    opts.exp_name = opts.video_path.split('/')[-4] + '_gsfixer_cross_atten_ori'
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    os.makedirs(opts.save_dir, exist_ok=True)
    pvd = GSFixer(opts)
    if opts.mode == 'gradual':
        pvd.infer_gradual(opts)
    elif opts.mode == 'direct':
        pvd.infer_direct(opts)
    elif opts.mode == 'bullet':
        pvd.infer_bullet(opts)
