import numpy as np
import scipy
from typing import List, Mapping, Optional, Text, Tuple, Union
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


def interpolate_camera_path(poses, num_views):
    positions = poses[:, :3, 3]  # shape: (N, 3)

    rotations = R.from_matrix(poses[:, :3, :3])
    quats = rotations.as_quat()  # shape: (N, 4)

    t = np.linspace(0, 1, len(poses))
    t_new = np.linspace(0, 1, num_views)

    if len(poses) < 3:
        kind = "linear"
    elif len(poses) < 4:
        kind = "quadratic"
    else:
        kind = "cubic"
    pos_interpolator = interp1d(t, positions, axis=0, kind=kind)
    new_positions = pos_interpolator(t_new)

    key_rots = R.from_quat(quats)
    slerp = Slerp(t, key_rots)
    new_rots = slerp(t_new)

    new_poses = np.zeros((num_views, 4, 4))
    new_poses[:, :3, :3] = new_rots.as_matrix()
    new_poses[:, :3, 3] = new_positions
    back_dir = new_poses[:, :3, 2]
    trans = np.random.uniform(0, 0.5)
    new_poses[:, :3, 3] = new_poses[:, :3, 3] + back_dir * trans
    new_poses[:, 3, 3] = 1.0

    return new_poses


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
    """Construct lookat view matrix in COLMAP convention.

    Args:
        lookdir: Looking direction (will be aligned with +Z axis)
        up: Up direction (will be aligned close to +Y axis)
        position: Camera position
    Returns:
        4x4 view matrix where:
        - Z axis is the looking direction (forward)
        - Y axis is up
        - X axis is right
    """
    vec2 = normalize(-lookdir)
    vec1 = normalize(up)
    vec0 = normalize(np.cross(vec1, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


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

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


def generate_ellipse_path(
    poses: np.ndarray,
    n_frames: int = 120,
    const_speed: bool = False,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = (-sc + offset) * 1.2
    high = (sc + offset) * 1.2
    # Optional height variation need not be symmetric
    # z_low = scale_z*np.percentile((poses[:, :3, 3]), 10, axis=0)
    # z_high = scale_z*np.percentile((poses[:, :3, 3]), 90, axis=0)
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
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # center = center + np.array([0, 0, -0.5])
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    angle_radians = np.radians(2 * scale)
    sign = np.random.choice([-1, 1])
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )

    for i in range(new_poses.shape[0]):
        new_poses[i, :, :3] = np.matmul(
            new_poses[i, :, :3], rotation_matrix
        )  # np.dot(, )

    return new_poses
