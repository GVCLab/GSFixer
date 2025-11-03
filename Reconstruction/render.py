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

import sys
import torch
from datasets import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import DataParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    dp = DataParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument(
        "--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF"
    )
    parser.add_argument(
        "--depth_trunc", default=-1.0, type=float, help="Mesh: Max depth range for TSDF"
    )
    parser.add_argument(
        "--sdf_trunc", default=-1.0, type=float, help="Mesh: truncation value for TSDF"
    )
    parser.add_argument(
        "--num_cluster",
        default=50,
        type=int,
        help="Mesh: number of connected clusters to export",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="Mesh: using unbounded mode for meshing",
    )
    parser.add_argument(
        "--mesh_res",
        default=1024,
        type=int,
        help="Mesh: resolution for unbounded mesh extraction",
    )
    parser.add_argument("--mono_depth", action="store_true")
    parser.add_argument(
        "--downsample_factor",
        default=1,
        type=int,
        help="Downsample factor for the input images",
    )
    parser.add_argument("--video_only", action="store_true", help="only render video")
    parser.add_argument("--camera_path_file", type=str, default=None)

    # args = parser.parse_args(sys.argv[1:])

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.mono_depth = False
    args.repair = False

    dataset_args = dp.extract(args)
    args.dataset = dataset_args
    model_args = model.extract(args)
    iteration = args.iteration
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(model_args.sh_degree)
    scene = Scene(
        model_args.model_path, args, gaussians, load_iteration=iteration, shuffle=False
    )
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, "train")
    test_dir = os.path.join(args.model_path, "test")
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    if not args.video_only:
        if not args.skip_train:
            print("export training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(
                [scene.getTrainInstant() for _ in range(len(scene.trainset))]
            )
            gaussExtractor.export_image(train_dir)
            
        if (not args.skip_test) and (len(scene.valset) > 0):
            print("export rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(
                [scene.getTestInstant() for _ in range(len(scene.valset))]
            )
            gaussExtractor.export_image(test_dir)
            traj_dir = os.path.join(args.model_path, "videos")
            gaussExtractor.export_video(traj_dir + f"/{scene.loaded_iter}")

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, "videos")
        os.makedirs(traj_dir, exist_ok=True)
        n_frames = 120
        print(f"len(scene.valset): {len(scene.valset)}")
        cam_traj = generate_path(
            [scene.getTrainInstant() for _ in range(len(scene.trainset))],
            args=args,
            n_frames=n_frames,
        )
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_video(traj_dir + f"/{scene.loaded_iter}")
