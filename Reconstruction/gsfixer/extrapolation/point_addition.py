import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn

import torch


def add_points(scene, novel_view, repaired_rgb, aligned_depth, unreliable_map):
    print("Adding points...")
    worldtocams = torch.from_numpy(novel_view.w2c).cuda().double()
    camtoworlds = torch.inverse(worldtocams)
    Ks = torch.from_numpy(novel_view.K).cuda().double()
    unreliable_map = unreliable_map.squeeze(-1)
    y, x = torch.where(unreliable_map)

    if len(y) > 0:
        # Create new points
        new_points = []
        new_colors = []

        num_points_to_add = int(0.1 * len(y))
        indices = torch.randperm(len(y))[:num_points_to_add]

        for idx in indices:
            yi, xi = y[idx], x[idx]
            depth = aligned_depth[yi, xi]
            color = repaired_rgb[yi, xi]
            # Backproject to 3D space
            pixel = (
                torch.tensor([xi, yi, 1.0], device=torch.device("cuda"))
                .unsqueeze(0)
                .double()
            )
            cam_point = torch.linalg.inv(Ks) @ (pixel.T * depth)
            world_point = (
                camtoworlds[:3, :3] @ cam_point + camtoworlds[:3, 3:4]
            ).squeeze()
            new_points.append(world_point)
            new_colors.append(color)
        if len(new_points) > 10:
            new_points = torch.stack(new_points).cuda().float()
            new_colors = torch.stack(new_colors).cuda().float()
        else:
            return
        creat_gaussians_from_pcd(scene, new_points, new_colors)

        print("Finished adding points.")


def align_depth(repair_depth, artifact_depth, alphas):
    valid_mask = (repair_depth != 0) & (artifact_depth != 0) & (alphas > 0.4)
    x = repair_depth[valid_mask].flatten()
    y = artifact_depth[valid_mask].flatten()

    A = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(A, y).solution

    scale, shift = solution[0], solution[1]

    aligned_depth = repair_depth * scale + shift

    return scale, shift, aligned_depth


def get_unreliable_map(
    args, artifact_rgb, artifact_depth, alphas, aligned_depth, opacity_threshold=0.1
):
    unreliable_map = torch.zeros_like(alphas, dtype=torch.bool)
    unreliable_map = unreliable_map | (alphas < opacity_threshold)
    depth_diff = torch.abs(artifact_depth - aligned_depth)
    threshold = torch.quantile(depth_diff.flatten(), 0.9)
    depth_mask = artifact_depth == 0
    unreliable_map = unreliable_map | depth_mask
    # Count and print the number of 0s and 1s in the unreliable map
    num_zeros = torch.sum(unreliable_map == False).item()
    num_ones = torch.sum(unreliable_map == True).item()
    print(f"Unreliable map contains {num_zeros} zeros and {num_ones} ones")
    # Save unreliable map as image
    unreliable_map_vis = unreliable_map.float() * 255
    unreliable_map_vis = unreliable_map_vis.squeeze(-1).cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.join(args.model_path, "unreliable_maps"), exist_ok=True)

    # Save single unreliable map
    save_path = os.path.join(args.model_path, "unreliable_maps", "unreliable_map.png")
    imageio.imwrite(save_path, unreliable_map_vis)
    return unreliable_map


def creat_gaussians_from_pcd(scene, points, colors):
    fused_point_cloud = points
    fused_color = RGB2SH(colors)
    features = (
        torch.zeros((fused_color.shape[0], 3, (scene.gaussians.max_sh_degree + 1) ** 2))
        .float()
        .cuda()
    )
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    dist2 = torch.clamp_min((distCUDA2(points).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
    rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

    opacities = inverse_sigmoid(
        0.1
        * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
    )

    new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    new_features_dc = nn.Parameter(
        features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
    )
    new_features_rest = nn.Parameter(
        features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
    )
    new_scaling = nn.Parameter(scales.requires_grad_(True))
    new_rotation = nn.Parameter(rots.requires_grad_(True))
    new_opacity = nn.Parameter(opacities.requires_grad_(True))
    # new_max_radii2D = torch.zeros((new_xyz.shape[0]), device="cuda")

    scene.gaussians.densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        selected_pts_mask=None
    )
    print(
        "Point addition finished. Now we have ",
        scene.gaussians.get_xyz.shape[0],
        " points.",
    )


def add_gaussians(
    args,
    scene,
    artifact_rgbs,
    artifact_depths,
    repaired_rgbs,
    repaired_depths,
    novel_views,
    rendered_alphas,
    add_indices,
):
    artifact_rgbs = artifact_rgbs.permute(1, 2, 3, 0).double()
    artifact_depths = artifact_depths.permute(1, 2, 3, 0).double()
    repaired_rgbs = repaired_rgbs.permute(1, 2, 3, 0).double()
    repaired_depths = repaired_depths.permute(1, 2, 3, 0).double()
    rendered_alphas = rendered_alphas.permute(1, 2, 3, 0).double()
    for idx in add_indices:
        repaired_depth = repaired_depths[idx].cuda()
        repaired_rgb = repaired_rgbs[idx].cuda()
        depth_for_align = repaired_depths[0].cuda()
        repaired_depth = (
            torch.nn.functional.interpolate(
                repaired_depth.unsqueeze(0).permute(0, 3, 1, 2),
                size=(repaired_rgbs.shape[1], repaired_rgbs.shape[2]),
                mode="nearest",
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        depth_for_align = (
            torch.nn.functional.interpolate(
                depth_for_align.unsqueeze(0).permute(0, 3, 1, 2),
                size=(repaired_rgbs.shape[1], repaired_rgbs.shape[2]),
                mode="nearest",
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        repaired_rgb = (
            torch.nn.functional.interpolate(
                repaired_rgb.unsqueeze(0).permute(0, 3, 1, 2),
                size=(repaired_rgbs.shape[1], repaired_rgbs.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        print("Adding points for index: ", idx)
        novel_view = novel_views[idx]
        artifact_rgb = artifact_rgbs[idx]
        artifact_depth = artifact_depths[idx]
        repaired_depth = repaired_depths[idx]
        alphas = rendered_alphas[idx]
        repaired_rgb = repaired_rgbs[idx]
        scale, shift, aligned_repaired_depth = align_depth(
            repaired_depth, artifact_depth, alphas
        )

        unreliable_map = get_unreliable_map(
            args, artifact_rgb, artifact_depth, alphas, aligned_repaired_depth
        )
        add_points(
            scene, novel_view, repaired_rgb, aligned_repaired_depth, unreliable_map
        )
    torch.cuda.empty_cache()
