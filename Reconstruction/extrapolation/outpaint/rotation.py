import torch
import imageio
import copy
from .base import OutpaintBase
from utils.camera_utils import CameraInfo
import os
import cv2
import numpy as np
from extrapolation.point_addition import add_gaussians
import random
from extrapolation.outpaint.render_utils import (
    transform_poses_pca,
    pad_poses,
    generate_ellipse_path,
)
from utils.camera_utils import SampleCamera
import time


class OutpaintRotation(OutpaintBase):
    def __init__(self, extrapolator, scene, args):
        super().__init__(extrapolator, scene, args)

    def sample_novel_views(self, start_index, end_index):
        num_sample_views = len(self.random_pose)
        if num_sample_views < self.args.num_frames:
            raise ValueError(
                "Not enough training views to sample 16 consecutive views."
            )
        K = copy.deepcopy(self.scene.trainset[0]["cam_info"].K)
        height, width = self.scene.trainset[0]["image"].shape[:2]
        new_x = np.random.randint(0, max(width - self.args.diffusion_crop_width, 1))
        new_y = np.random.randint(0, max(height - self.args.diffusion_crop_height, 1))
        K[0, 2] -= new_x
        K[1, 2] -= new_y
        novel_views = []
        print(f"start_index: {start_index}")
        for i in range(start_index, end_index):
            c2w = self.random_pose[i]
            w2c = np.linalg.inv(c2w)
            cam_info = CameraInfo(
                uid=None,
                colmapid=None,
                K=K,
                w2c=w2c,
                image_name=None,
                image_path=None,
                width=torch.tensor(self.args.diffusion_crop_width),
                height=torch.tensor(self.args.diffusion_crop_height),
            )
            novel_views.append(cam_info)
        return novel_views

    def get_ref_views(self, start_index, end_index):
        # Set random seed to current time
        random.seed(int(time.time()))
        ref_frame_index = torch.randint(start_index, end_index, (1,)).item()
        ref_image = self.scene.trainset[ref_frame_index]["image"]
        ref_image = ref_image.unsqueeze(0).permute(3, 0, 1, 2)
        return ref_image  # [3,1,H,W]

    def save_vis_results(
        self,
        iteration,
        artifact_rgb,
        artifact_depth,
        repaired_rgb,
        repaired_depth,
        start_index,
    ):
        artifact_rgb_video = (artifact_rgb + 1) / 2
        artifact_depth_video = (artifact_depth + 1) / 2
        compare_dir = os.path.join(self.args.model.model_path, "repair-compare")
        # Create directories for images
        img_dirs = {
            "rgb/generated": os.path.join(compare_dir, "rgb", "generated"),
            "rgb/artifact": os.path.join(compare_dir, "rgb", "artifact"),
            "depth/generated": os.path.join(compare_dir, "depth", "generated"),
            "depth/artifact": os.path.join(compare_dir, "depth", "artifact"),
        }
        for dir_path in img_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        def save_video_and_frames(frames, video_filename, img_dir, is_depth=False):
            writer = imageio.get_writer(video_filename, fps=30)
            for i, frame in enumerate(frames):
                frame = frame.detach().cpu().numpy()
                if is_depth:
                    frame = np.repeat(frame, 3, axis=-1)  # Convert to RGB
                else:
                    frame = np.clip(frame, 0, 1)  # Ensure values are in [0, 1] range
                frame = (frame * 255).astype(np.uint8)
                # Save video frame
                writer.append_data(frame)
                # Save individual frame as image
                img_path = os.path.join(
                    img_dir, f"{iteration}_{start_index:04d}_{i:04d}.png"
                )
                imageio.imwrite(img_path, frame)
            writer.close()

        # Process RGB frames
        repaired_rgb = repaired_rgb.permute(1, 2, 3, 0)
        artifact_rgb_video = artifact_rgb_video.permute(1, 2, 3, 0)
        save_video_and_frames(
            repaired_rgb,
            os.path.join(compare_dir, f"{iteration}_{start_index:04d}_rgb.mp4"),
            img_dirs["rgb/generated"],
        )
        save_video_and_frames(
            artifact_rgb_video,
            os.path.join(
                compare_dir, f"{iteration}_{start_index:04d}_rgb_artifact.mp4"
            ),
            img_dirs["rgb/artifact"],
        )

        # Process depth frames
        repaired_depth = repaired_depth.permute(1, 2, 3, 0)
        valid_mask = (repaired_depth != 0).float()
        repaired_depth = torch.where(
            valid_mask > 0,
            1 / (repaired_depth + 1e-6),
            torch.zeros_like(repaired_depth),
        )
        min_val = torch.min(repaired_depth * valid_mask + 1e6 * (1 - valid_mask))
        max_val = torch.max(repaired_depth * valid_mask)
        repaired_depth = torch.where(
            valid_mask > 0,
            (repaired_depth - min_val) / (max_val - min_val),
            torch.zeros_like(repaired_depth),
        )
        artifact_depth_video = artifact_depth_video.permute(1, 2, 3, 0)
        save_video_and_frames(
            repaired_depth,
            os.path.join(compare_dir, f"{iteration}_{start_index:04d}_depth.mp4"),
            img_dirs["depth/generated"],
            is_depth=True,
        )
        save_video_and_frames(
            artifact_depth_video,
            os.path.join(
                compare_dir, f"{iteration}_{start_index:04d}_depth_artifact.mp4"
            ),
            img_dirs["depth/artifact"],
            is_depth=True,
        )

    def generate_path(self, meta_data, n_frames=480):
        viewpoint_cameras = [SampleCamera(cam["cam_info"]) for cam in meta_data]
        c2ws = np.array(
            [
                np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy()))
                for cam in viewpoint_cameras
            ]
        )
        pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

        new_poses = generate_ellipse_path(
            args=self.args, poses=pose_recenter, n_frames=n_frames, z_variation=0
        )
        new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

        poses = []
        for c2w in new_poses:
            pose = c2w @ np.diag([1, -1, -1, 1])
            poses.append(pose)
        poses = np.stack(poses, 0)
        return poses

    def read_path_from_json(self, camera_path_file=None):
        import json

        camera_path_file = self.args.camera_path_file
        with open(camera_path_file, "r") as f:
            camera_path_data = json.load(f)
        sample_c2ws = []
        for i in range(len(camera_path_data["camera_path"])):
            c2w = np.array(camera_path_data["camera_path"][i]["camera_to_world"])
            c2w = c2w.reshape(4, 4)
            sample_c2ws.append(c2w)
        sample_c2ws = np.array(sample_c2ws)
        c2ws = sample_c2ws
        poses = []
        for c2w in c2ws:
            pose = c2w @ np.diag([1, -1, -1, 1])
            poses.append(pose)
        poses = np.stack(poses, 0)
        return poses

    def get_render_masks(self, poses):
        masks = []
        novel_views = self.sample_novel_views(0, len(poses))
        _, _, rendered_alphas = self.get_render_results(novel_views)
        print(f"rendered_alphas: {rendered_alphas.shape}")
        for i in range(rendered_alphas.shape[1]):
            mask = rendered_alphas[:, i, :, :] <= 0.999
            masks.append(mask)
        print(f"masks: {len(masks)}", masks[0].shape)
        return masks

    def add_trainset(self, novel_views, repaired_rgb, repaired_depth, start_add_index):
        reg_data = []
        for i in range(start_add_index, len(novel_views)):
            cam_info = novel_views[i]
            image = repaired_rgb[:, i].permute(1, 2, 0).float()
            mono_depth = repaired_depth[:, i].permute(1, 2, 0).float()
            data = {
                "cam_info": cam_info,
                "image": image.detach().cpu(),
                "mono_depth": mono_depth.detach().cpu(),
            }
            reg_data.append(data)
        self.scene.regset.populate(reg_data)
        print("length of regset", len(self.scene.regset))

    def run(self, iteration):
        start_add_index = 0
        start_rotation_cam_index = np.random.randint(
            0, len(self.scene.trainset) - self.args.num_frames
        )

        if self.args.camera_path_file is not None:
            self.random_pose_all = self.read_path_from_json()
            total_frames = len(self.random_pose_all)
            indices = np.linspace(0, total_frames - 1, 48, dtype=int)
            self.random_pose = self.random_pose_all[indices]
        else:
            self.random_pose = self.generate_path(self.scene.trainset, 48)
        start_sample_indices = [0, 16, 32]
        start_sample_index = start_sample_indices[
            iteration % len(start_sample_indices)
        ]  # random.choice(start_sample_indices)#random.randint(0, len(self.random_pose) - self.args.num_frames)
        self.random_pose = self.random_pose[
            start_sample_index : start_sample_index + self.args.num_frames
        ]
        orig_artifact_rgb = None
        orig_artifact_depth = None
        repaired_rgb = None
        repaired_depth = None
        novel_views = None
        rendered_alphas = None

        novel_views = self.sample_novel_views(0, len(self.random_pose))
        artifact_rgb, artifact_depth, rendered_alphas = self.get_render_results(
            novel_views
        )

        # Create videos directory if it doesn't exist
        ref_frames = self.get_ref_views(
            start_rotation_cam_index, start_rotation_cam_index + self.args.num_frames
        )
        orig_artifact_rgb = artifact_rgb.detach().clone()
        orig_artifact_depth = artifact_depth.detach().clone()
        artifact_rgb = self.rgb_preprocess(artifact_rgb)
        artifact_depth = self.depth_preprocess(artifact_depth)
        ref_frames = self.rgb_preprocess(ref_frames)

        repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth = (
            self.repair(artifact_rgb, artifact_depth, ref_frames)
        )  # rgb:[0,1], depth(depth):[0,]
        self.add_trainset(
            novel_views, repaired_rgb, repaired_depth, start_add_index=start_add_index
        )
        ##### add points #####
        add_gaussians(
            self.args,
            self.scene,
            orig_artifact_rgb,
            orig_artifact_depth,
            repaired_rgb,
            repaired_depth,
            novel_views,
            rendered_alphas,
            [0, 15],
        )
        # self.save_vis_results(iteration, artifact_rgb, artifact_depth, repaired_rgb, repaired_depth, 0)
