# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
from utils.camera_utils import Camera

import torch


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform


def generate_ellipse_path(
    args,
    poses: np.ndarray,
    n_frames: int = 120,
    const_speed: bool = True,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = args.path_scale * (-sc + offset)
    high = args.path_scale * (sc + offset)
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:

    # # Resample theta angles so that the velocity is closer to constant.
    # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    # positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]
    direction = center - positions
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    distance = args.distance
    positions = positions - direction * distance
    positions[:, 2] += args.position_z_offset

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    angle_radians = np.radians(args.rotation_angle)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)],
        ]
    )

    for i in range(new_poses.shape[0]):
        new_poses[i, :, :3] = np.matmul(
            new_poses[i, :, :3], rotation_matrix
        )  # np.dot(, )
    return new_poses


def move_and_rotate_render(poses, cam_index, n_frames=480):
    """Generate a camera trajectory that follows the original path while rotating around the up axis."""
    angles = np.linspace(0, 360, n_frames)
    rotated_camtoworlds = []

    for i, angle in enumerate(angles):
        angle_radians = np.radians(angle)
        # Create rotation matrix around Y axis (up direction)
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
                [0, 0, 0, 1],
            ]
        )
        # Apply rotation to the camera pose
        rotated_cam = np.dot(poses, rotation_matrix)
        rotated_camtoworlds.append(rotated_cam)

    return np.array(rotated_camtoworlds)


def generate_path(meta_data, args, n_frames=480):
    if not isinstance(meta_data[0], Camera):
        viewpoint_cameras = [Camera(cam["cam_info"]) for cam in meta_data]
    else:
        viewpoint_cameras = meta_data

    if args.camera_path_file is not None:
        import json

        camera_path_file = args.camera_path_file
        with open(camera_path_file, "r") as f:
            camera_path_data = json.load(f)
        sample_c2ws = []
        for i in range(len(camera_path_data["camera_path"])):
            c2w = np.array(camera_path_data["camera_path"][i]["camera_to_world"])
            c2w = c2w.reshape(4, 4)
            sample_c2ws.append(c2w)
        sample_c2ws = np.array(sample_c2ws)
        c2ws = sample_c2ws
        print(c2ws.shape)

        traj = []
        for c2w in c2ws:
            c2w = c2w @ np.diag([1, -1, -1, 1])
            cam = copy.deepcopy(viewpoint_cameras[0])
            image_height = 512
            image_width = 960
            cam.image_height = int(image_height / 2) * 2
            cam.image_width = int(image_width / 2) * 2
            cam.world_view_transform = (
                torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
            )
            cam.full_proj_transform = (
                cam.world_view_transform.unsqueeze(0).bmm(
                    cam.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]
            traj.append(cam)
    else:
        c2ws = np.array(
            [
                np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy()))
                for cam in viewpoint_cameras
            ]
        )
        pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

        new_poses = generate_ellipse_path(
            args=args, poses=pose_recenter, n_frames=n_frames, z_variation=0.5
        )
        new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

        traj = []
        for c2w in new_poses:
            c2w = c2w @ np.diag([1, -1, -1, 1])
            cam = copy.deepcopy(viewpoint_cameras[0])
            cam.image_height = int(cam.image_height / 2) * 2
            cam.image_width = int(cam.image_width / 2) * 2
            cam.world_view_transform = (
                torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
            )
            cam.full_proj_transform = (
                cam.world_view_transform.unsqueeze(0).bmm(
                    cam.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]
            traj.append(cam)
    return traj


def generate_path_rotation(meta_data, n_frames=480, cam_index=20):
    if not isinstance(meta_data[0], Camera):
        viewpoint_cameras = [Camera(cam["cam_info"]) for cam in meta_data]
    else:
        viewpoint_cameras = meta_data
    c2ws = np.array(
        [
            np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy()))
            for cam in viewpoint_cameras
        ]
    )
    pose = c2ws[cam_index]
    new_poses = move_and_rotate_render(pose, cam_index, n_frames)

    traj = []
    for c2w in new_poses:
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(
                cam.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        traj.append(cam)

    return traj


def load_img(pth: str) -> np.ndarray:
    """Load an image and cast to float32."""
    with open(pth, "rb") as f:
        image = np.array(Image.open(f), dtype=np.float32)
    return image


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open(pth, "wb") as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(np.uint8)
        ).save(f, "PNG")


def save_img_f32(depthmap, pth):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    with open(pth, "wb") as f:
        Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, "TIFF")
