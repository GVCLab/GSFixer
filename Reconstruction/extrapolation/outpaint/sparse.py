import torch
import imageio
import copy
from .base import OutpaintBase
from utils.camera_utils import CameraInfo
import os
import cv2
import numpy as np
from extrapolation.outpaint.sample_utils import (
    generate_ellipse_path,
    interpolate_camera_path,
)


class OutpaintSparse(OutpaintBase):
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

    def get_ref_views(self, novel_views):
        novel_positions = torch.stack(
            [torch.from_numpy(np.linalg.inv(view.w2c)[:3, 3]) for view in novel_views]
        )  # [N, 3]
        novel_forwards = torch.stack(
            [torch.from_numpy(np.linalg.inv(view.w2c)[:3, 2]) for view in novel_views]
        )
        center_position = novel_positions.mean(dim=0)  # [3]
        mean_forward = torch.nn.functional.normalize(
            novel_forwards.mean(dim=0), dim=0
        )  # [3]

        min_score = float("inf")
        closest_idx = 0
        position_weight = 1.0
        direction_weight = 0.5

        for i, view in enumerate(self.scene.trainset):
            # c2w=np.linalg.inv(view["cam_info"].w2c)
            w2c_tensor = torch.from_numpy(view["cam_info"].w2c)  #
            inv_w2c = torch.linalg.inv(w2c_tensor)
            pos = inv_w2c[:3, 3]
            forward = inv_w2c[:3, 2]
            position_distance = torch.norm(pos - center_position)
            direction_difference = 1 - torch.dot(forward, mean_forward)
            score = (
                position_weight * position_distance
                + direction_weight * direction_difference
            )

            if score < min_score:
                min_score = score
                closest_idx = i
        ref_img_idx = self.scene.trainset.indices[closest_idx]
        ref_image = imageio.imread(self.scene.trainset.parser.image_paths[ref_img_idx])[
            ..., :3
        ]  # [0,255]
        ref_image = torch.from_numpy(ref_image).float() / 255.0
        ref_image = ref_image.unsqueeze(0).permute(3, 0, 1, 2)
        return ref_image  # [3,1,H,W]

    def generate_360_degree_poses(self, i):
        poses = []
        for view in self.scene.trainset:
            w2c = view["cam_info"].w2c
            c2w = np.linalg.inv(w2c)
            poses.append(c2w)
        poses = np.stack(poses, 0)
        z_variation = 1.5 - 0.20 * i
        z_phase = np.random.random()
        if z_variation <= 0.5:
            z_variation = 0.5
        random_poses = generate_ellipse_path(
            poses[:, :3], 85, z_variation=z_variation, z_phase=z_phase, scale=i
        )
        homogeneous_row = np.zeros((len(random_poses), 1, 4))
        homogeneous_row[:, 0, 3] = 1
        random_poses = np.concatenate([random_poses, homogeneous_row], axis=1)
        return random_poses

    def generate_interpolate_camera_path(self, i):
        poses = []
        z_variation = 1.5 - 0.20 * i
        z_phase = np.random.random()
        if z_variation <= 0.5:
            z_variation = 0.5
        for view in self.scene.trainset:
            w2c = view["cam_info"].w2c
            c2w = np.linalg.inv(w2c)
            poses.append(c2w)
        poses = np.stack(poses, 0)
        interpolate_poses = interpolate_camera_path(poses, 20)

        random_poses = generate_ellipse_path(
            interpolate_poses[:, :3],
            80,
            z_variation=z_variation,
            z_phase=z_phase,
            scale=i,
        )
        homogeneous_row = np.zeros((len(random_poses), 1, 4))
        homogeneous_row[:, 0, 3] = 1
        random_poses = np.concatenate([random_poses, homogeneous_row], axis=1)
        return random_poses

    def run(self, iteration):
        start_add_index = 0
        last_rgb_output = None
        self.random_pose = self.generate_360_degree_poses(iteration)
        start_indices, overlaps = self.generate_overlapping_indices(
            len(self.random_pose)
        )
        repaired_rgb = None
        repaired_depth = None
        novel_views = None
        for start_index, overlap in zip(start_indices, overlaps):
            end_index = min(start_index + self.args.num_frames, len(self.random_pose))
            novel_views = self.sample_novel_views(start_index, end_index)
            artifact_rgb, artifact_depth, rendered_alphas = self.get_render_results(
                novel_views
            )

            # Create videos directory if it doesn't exist
            ref_frames = self.get_ref_views(novel_views)
            artifact_rgb = self.rgb_preprocess(artifact_rgb)
            artifact_depth = self.depth_preprocess(artifact_depth)
            ref_frames = self.rgb_preprocess(ref_frames)

            repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth = (
                self.repair(artifact_rgb, artifact_depth, ref_frames)
            )  # rgb:[0,1], depth(depth):[0,]
            start_add_index = overlap if last_rgb_output is not None else 0
            self.add_trainset(
                novel_views, repaired_rgb, repaired_depth, start_add_index
            )
            last_rgb_output = orig_repaired_rgb
