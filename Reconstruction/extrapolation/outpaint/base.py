import torch
from gaussian_renderer import render
from utils.camera_utils import SampleCamera


class OutpaintBase:
    def __init__(self, extrapolator, scene, args):
        self.extrapolator = extrapolator
        self.scene = scene
        self.args = args

    def generate_overlapping_indices(self, total_length):
        print("total_length: ", total_length)
        num_frames = self.args.num_frames
        num_of_video = total_length // num_frames + 2
        num_of_overlap = num_of_video * num_frames - total_length
        if num_of_overlap < num_of_video:
            num_of_overlap += num_of_video
        start_indices = [0]
        overlap = [0]
        idx = 0
        average_overlap = num_of_overlap // (num_of_video - 1)
        idx += 1
        while idx < num_of_video - 1:
            curr_start_index = start_indices[idx - 1] + num_frames - average_overlap
            start_indices.append(curr_start_index)
            overlap.append(average_overlap)
            idx += 1
        last_overlap = num_of_overlap - sum(overlap)
        overlap.append(last_overlap)
        start_indices.append(start_indices[-1] + num_frames - last_overlap)
        return start_indices, overlap

    def sample_novel_views(self):
        pass

    def get_ref_views(self):
        pass

    def get_render_results(self, novel_views):
        artifact_rgb = []
        artifact_depth = []
        artifact_alpha = []
        with torch.no_grad():
            for viewpoint in novel_views:
                cam = SampleCamera(viewpoint)
                render_pkg = render(
                    cam, self.scene.gaussians, self.args.pipe, self.args.background
                )
                artifact_rgb.append(render_pkg["render"])
                artifact_depth.append(render_pkg["surf_depth"])
                artifact_alpha.append(render_pkg["rend_alpha"])
        artifact_rgb = torch.stack(artifact_rgb, dim=0).detach()
        artifact_depth = torch.stack(artifact_depth, dim=0).detach()
        artifact_alpha = torch.stack(artifact_alpha, dim=0).detach()
        artifact_rgb = artifact_rgb.permute(1, 0, 2, 3)
        artifact_depth = artifact_depth.permute(1, 0, 2, 3)
        artifact_alpha = artifact_alpha.permute(1, 0, 2, 3)
        return artifact_rgb, artifact_depth, artifact_alpha  # [C,T,H,W]

    def rgb_preprocess(self, artifact_rgb):
        artifact_rgb = (artifact_rgb - 0.5) * 2
        artifact_rgb = torch.nn.functional.interpolate(
            artifact_rgb.unsqueeze(0),
            size=(
                artifact_rgb.shape[1],
                self.args.diffusion_resize_height,
                self.args.diffusion_resize_width,
            ),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return artifact_rgb

    def depth_preprocess(self, artifact_depth):
        epsilon = 1e-10
        valid_mask = artifact_depth > 0
        disparity = torch.zeros_like(artifact_depth)
        disparity[valid_mask] = 1.0 / (artifact_depth[valid_mask] + epsilon)
        valid_disparities = torch.masked_select(disparity, valid_mask)

        if valid_disparities.numel() > 0:
            disp_min = valid_disparities.min()
            disp_max = valid_disparities.max()
            normalized_disparity = torch.zeros_like(disparity)
            normalized_disparity[valid_mask] = (disparity[valid_mask] - disp_min) / (
                disp_max - disp_min
            )

        else:
            print("Warning: No valid depth values found")
            normalized_disparity = torch.zeros_like(disparity)
        normalized_disparity = (normalized_disparity - 0.5) * 2
        normalized_disparity = torch.nn.functional.interpolate(
            normalized_disparity.unsqueeze(0),
            size=(
                normalized_disparity.shape[1],
                self.args.diffusion_resize_height,
                self.args.diffusion_resize_width,
            ),
            mode="nearest",
        ).squeeze(0)
        return normalized_disparity

    def repair(
        self, artifact_rgb, artifact_depth, ref_frames, init_rgb=None, init_depth=None
    ):
        repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth = (
            self.extrapolator.repair(
                artifact_rgb, artifact_depth, ref_frames, init_rgb, init_depth
            )
        )  # [3, 16, 320, 512], [1, 16, 320, 512]
        return repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth

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

    def add_points(
        self,
        orig_artifact_rgb,
        orig_artifact_depth,
        repaired_rgb,
        repaired_depth,
        novel_views,
        rendered_alphas,
        add_indices,
    ):
        self.extrapolation_utils.add_Gaussians(
            orig_artifact_rgb,
            orig_artifact_depth,
            repaired_rgb,
            repaired_depth,
            novel_views,
            rendered_alphas,
            add_indices,
        )

    def save_videos(
        self,
        orig_artifact_rgb,
        repaired_rgb,
        repaired_depth,
        orig_artifact_depth,
        current_step,
        start_index,
    ):
        # Save videos
        print(
            "orig_artifact_rgb: ",
            orig_artifact_rgb.shape,
            "repaired_rgb: ",
            repaired_rgb.shape,
            "repaired_depth: ",
            repaired_depth.shape,
        )
        video_depth = 1 / (repaired_depth + 1e-6)  # from depth to disparity
        video_depth = (video_depth - video_depth.min()) / (
            video_depth.max() - video_depth.min()
        )  # vis disparity is better
        video_artifact_depth = torch.zeros_like(orig_artifact_depth)
        mask = orig_artifact_depth > 0
        video_artifact_depth[mask] = 1 / (
            orig_artifact_depth[mask] + 1e-6
        )  # from depth to disparity
        video_artifact_depth = (video_artifact_depth - video_artifact_depth.min()) / (
            video_artifact_depth.max() - video_artifact_depth.min()
        )  # vis disparity is better
        self.runner.save_videos(
            orig_artifact_rgb.permute(1, 2, 3, 0),
            repaired_rgb.permute(1, 2, 3, 0),
            video_depth.permute(1, 2, 3, 0),
            video_artifact_depth.permute(1, 2, 3, 0),
            self.cfg,
            current_step,
            start_index,
        )

    def entry(self):
        pass
