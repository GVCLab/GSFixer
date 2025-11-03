import torch
from gaussian_renderer import render
from utils.camera_utils import Camera, SampleCamera
import torch.nn.functional as F


class OutpaintBase:
    def __init__(self, extrapolator, scene, args):
        self.extrapolator = extrapolator
        self.scene = scene
        self.args = args

    def generate_overlapping_indices(self, total_length):
        print("total_length: ", total_length)
        # num_frames = self.args.num_frames
        num_frames = int(self.args.num_frames / 3) + 1
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
        # train_views = self.scene.getTrainCameras().copy()
        train_views = self.scene.trainset.data_list
        artifact_rgb = []
        # artifact_depth = []
        # artifact_alpha = []
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            for idx, viewpoint in enumerate(novel_views):
                viewpoint_cam = SampleCamera(viewpoint)
                render_pkg = render(viewpoint_cam, self.scene.gaussians, self.args.pipe, background, use_trained_exp=self.args.train_test_exp, separate_sh=False)
                # if idx % (self.args.num_frames-1) == 0 or idx == 0:
                #     rendering = train_views[int(idx/(self.args.num_frames-1))]['image'].unsqueeze(0).permute(0, 3, 1, 2)
                #     rendering = F.interpolate(rendering, size=(self.args.diffusion_resize_height, self.args.diffusion_resize_width), mode='bilinear', align_corners=False).squeeze(0).cuda()
                #     artifact_rgb.append(rendering)
                # else:
                #     rendering = render_pkg["render"].detach().unsqueeze(0)
                #     rendering = F.interpolate(rendering, size=(self.args.diffusion_resize_height, self.args.diffusion_resize_width), mode='bilinear', align_corners=False).squeeze(0).cuda()
                #     artifact_rgb.append(rendering)

                rendering = render_pkg["render"].detach().unsqueeze(0)
                rendering = F.interpolate(rendering, size=(self.args.diffusion_resize_height, self.args.diffusion_resize_width), mode='bilinear', align_corners=False).squeeze(0).cuda()
                artifact_rgb.append(rendering)

        artifact_rgb = torch.stack(artifact_rgb, dim=0).detach()
        return artifact_rgb

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

    # def repair(
    #     self, artifact_rgb, artifact_depth, ref_frames, init_rgb=None, init_depth=None
    # ):
    #     repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth = (
    #         self.extrapolator.repair(
    #             artifact_rgb, artifact_depth, ref_frames, init_rgb, init_depth
    #         )
    #     )  # [3, 16, 320, 512], [1, 16, 320, 512]
    #     return repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth
    
    def repair(self, artifact_rgb):
        repaired_rgb = self.extrapolator.repair(artifact_rgb)
         # [3, 16, 320, 512], [1, 16, 320, 512]
        return repaired_rgb

    # def add_trainset(self, novel_views, repaired_rgb, repaired_depth, start_add_index):
    def add_trainset(self, novel_views, repaired_rgb):
        reg_data = []
        for i in range(len(novel_views)):
            cam_info = novel_views[i]
            image = repaired_rgb[i]
            # mono_depth = repaired_depth[:, i].permute(1, 2, 0).float()
            data = {
                "cam_info": cam_info,
                "image": image,
            }
            reg_data.append(data)
        self.scene.regset.populate(reg_data)
        # self.regset.populate(reg_data)
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
