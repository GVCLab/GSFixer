import torch
import imageio
import copy
from .base import OutpaintBase
from utils.camera_utils import CameraInfo
import os
import cv2
import numpy as np


class OutpaintCrop(OutpaintBase):
    def __init__(self, extrapolator, scene, args):
        super().__init__(extrapolator, scene, args)

    def sample_novel_views(self, start_index, end_index):
        num_train_views = len(self.scene.trainset)
        if num_train_views < self.args.num_frames:
            raise ValueError(
                "Not enough training views to sample 16 consecutive views."
            )
        novel_views = []
        new_x = 0
        new_y = 0
        orig_img = imageio.imread(self.scene.trainset.parser.image_paths[0])[..., :3]
        orig_img_height, orig_img_width = orig_img.shape[:2]
        new_x = np.random.randint(
            0, max(orig_img_width - self.args.diffusion_crop_width, 1)
        )
        new_y = np.random.randint(
            0, max(orig_img_height - self.args.diffusion_crop_height, 1)
        )
        print(f"start_index: {start_index}")
        for i in range(start_index, end_index):
            parser_index = self.scene.trainset.indices[i]
            train_view = self.scene.trainset[i]
            K = copy.deepcopy(train_view["cam_info"].K)
            w2c = copy.deepcopy(train_view["cam_info"].w2c)
            if len(self.scene.trainset.trajectories[0]) == 4:
                x_start, y_start, _, _ = (
                    self.scene.trainset.trajectories[parser_index]
                    if hasattr(self.scene.trainset, "trajectories")
                    else (0, 0)
                )
            else:
                x_start, y_start = (
                    self.scene.trainset.trajectories[parser_index]
                    if hasattr(self.scene.trainset, "trajectories")
                    else (0, 0)
                )
            K[0, 2] += x_start
            K[1, 2] += y_start
            K[0, 2] -= new_x
            K[1, 2] -= new_y
            # h, w = train_view["image"].shape[:2]
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
        ref_frame_index = torch.randint(0, len(self.scene.trainset), (1,)).item()
        ref_image = self.scene.trainset[ref_frame_index]["image"]
        ref_image = ref_image.unsqueeze(0).permute(3, 0, 1, 2)
        return ref_image  # [3,1,H,W]

    def run(self, iteration):
        start_add_index = 0
        last_rgb_output = None
        start_indices, overlaps = self.generate_overlapping_indices(
            len(self.scene.trainset)
        )
        repaired_rgb = None
        repaired_depth = None
        novel_views = None
        for start_index, overlap in zip(start_indices, overlaps):
            end_index = min(
                start_index + self.args.num_frames, len(self.scene.trainset)
            )

            novel_views = self.sample_novel_views(start_index, end_index)
            artifact_rgb, artifact_depth, artifact_alpha = self.get_render_results(
                novel_views
            )
            ref_frames = self.get_ref_views(start_index, end_index)
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
            artifact_rgb_video = (artifact_rgb + 1) / 2
            artifact_depth_video = (artifact_depth + 1) / 2
            compare_dir = os.path.join(self.args.model.model_path, "repair-compare")
            os.makedirs(compare_dir, exist_ok=True)

            # def save_video(frames, filename, is_depth=False):
            #     writer = imageio.get_writer(filename, fps=30)
            #     for frame in frames:
            #         frame = frame.detach().cpu().numpy()
            #         if is_depth:
            #             frame = np.repeat(frame, 3, axis=-1)  # Convert to RGB
            #         else:
            #             frame = np.clip(
            #                 frame, 0, 1
            #             )  # Ensure values are in [0, 1] range
            #         frame = (frame * 255).astype(np.uint8)
            #         writer.append_data(frame)
            #     writer.close()

            repaired_rgb = repaired_rgb.permute(1, 2, 3, 0)
            artifact_rgb_video = artifact_rgb_video.permute(1, 2, 3, 0)
            rgb_comparison = torch.cat([repaired_rgb, artifact_rgb_video], dim=2)
            # save_video(
            #     rgb_comparison,
            #     os.path.join(compare_dir, f"{iteration}_{start_index:04d}_rgb.mp4"),
            #     is_depth=False,
            # )

            repaired_depth = repaired_depth.permute(1, 2, 3, 0)
            # Create mask for valid depth values
            valid_mask = (repaired_depth != 0).float()
            # Convert depth to disparity while handling zeros
            repaired_depth = torch.where(
                valid_mask > 0,
                1 / (repaired_depth + 1e-6),
                torch.zeros_like(repaired_depth),
            )
            # Normalize only valid disparity values
            min_val = torch.min(repaired_depth * valid_mask + 1e6 * (1 - valid_mask))
            max_val = torch.max(repaired_depth * valid_mask)
            repaired_depth = torch.where(
                valid_mask > 0,
                (repaired_depth - min_val) / (max_val - min_val),
                torch.zeros_like(repaired_depth),
            )
            artifact_depth_video = artifact_depth_video.permute(1, 2, 3, 0)
            depth_comparison = torch.cat([repaired_depth, artifact_depth_video], dim=2)
            # save_video(
            #     depth_comparison,
            #     os.path.join(compare_dir, f"{iteration}_{start_index:04d}_depth.mp4"),
            #     is_depth=True,
            # )
            start_add_index = overlap if last_rgb_output is not None else 0
            last_rgb_output = orig_repaired_rgb
