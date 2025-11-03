#! /bin/bash

CUDA_VISIBLE_DEVICES=8 python train.py \
    --data_dir [data_dir] \
    -m output_ours/counter_completion \
    --test_iterations 7_000 \
    --diffusion_ckpt ./diffusion_ckpt/epoch=59-step=34000.ckpt \
    --diffusion_config ./generation_infer.yaml \
    --num_frames 16 \
    --outpaint_type rotation \
    --add_indices 7 15 \
    --depth_loss \
    --iterations 26000 \
    --diffusion_resize_width 960 \
    --diffusion_resize_height 512 \
    --diffusion_crop_width 960 \
    --diffusion_crop_height 512 \
    --repair \
    --port 6678 \
    --densify_from_iter 500 \
    --densify_until_iter 12000 \
    --diffusion_until 30000 \
    --start_diffusion_iter 5000 \
    --diffusion_every 4000 \
    --opacity_reset_interval 15000 \
    --unconditional_guidance_scale 2.2 \
    --start_dist_iter 3000 \
    --camera_path_file [camera_path_file]
