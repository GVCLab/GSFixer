import torch
import imageio
import copy
from .base import OutpaintBase
from utils.camera_utils import CameraInfo
import numpy as np
import scipy
from gsfixer.extrapolation.outpaint.sample_utils import (
    generate_ellipse_path,
    interpolate_camera_path,
)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def interp_traj(c2ws: torch.Tensor, n_inserts: int = 25, device='cuda') -> torch.Tensor:
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        interpolated_path = generate_interpolated_path(np.stack([start_pose, end_pose]), n_inserts)
        interpolated_path = interpolated_path[:-1]
        interpolated_poses.append(interpolated_path)

    interpolated_poses.append(c2ws[-1:])
    full_path = np.concatenate(interpolated_poses, axis=0)
    return full_path

def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    def viewmatrix(lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    new_points = points_to_poses(new_points)
    new_points = np.concatenate(
            [
                new_points,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(new_points), axis=0
                ),
            ],
            axis=1,
        ) 
    return new_points

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
                width=width,
                height=height,
            )
            novel_views.append(cam_info)
        return novel_views


    def get_ref_views(self, novel_views, k=2):
        novel_positions = torch.stack(
            [torch.from_numpy(np.linalg.inv(view.w2c)[:3, 3]) for view in novel_views]
        )  # [N, 3]
        novel_forwards = torch.stack(
            [torch.from_numpy(np.linalg.inv(view.w2c)[:3, 2]) for view in novel_views]
        )  # [N, 3]

        center_position = novel_positions.mean(dim=0)  # [3]
        mean_forward = torch.nn.functional.normalize(novel_forwards.mean(dim=0), dim=0)  # [3]

        position_weight = 1.0
        direction_weight = 0.5

        all_scores = []
        for i, view in enumerate(self.scene.trainset):
            w2c_tensor = torch.from_numpy(view["cam_info"].w2c)
            inv_w2c = torch.linalg.inv(w2c_tensor)
            pos = inv_w2c[:3, 3]
            forward = inv_w2c[:3, 2]
            position_distance = torch.norm(pos - center_position)
            direction_difference = 1 - torch.dot(forward, mean_forward)
            score = position_weight * position_distance + direction_weight * direction_difference

            all_scores.append((score.item(), i))

        all_scores.sort(key=lambda x: x[0])
        top_k_scores = all_scores[:k]

        ref_images = []
        ref_c2ws = []
        for score, idx in top_k_scores:
            w2c = self.scene.trainset[idx]["cam_info"].w2c
            ref_c2ws.append(np.linalg.inv(w2c))

            img_idx = self.scene.trainset.indices[idx]
            image_path = self.scene.trainset.parser.image_paths[img_idx]
            img = imageio.imread(image_path)[..., :3]
            img_tensor = torch.from_numpy(img).float() / 255.0  # [H, W, 3]
            img_tensor = img_tensor.permute(2, 0, 1)
            ref_images.append(img_tensor)

        return torch.stack(ref_images), ref_c2ws


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
        self.random_pose = self.generate_360_degree_poses(iteration)
        start_indices, overlaps = self.generate_overlapping_indices(
            len(self.random_pose)
        )
        repaired_rgb = None
        novel_views = None
        K = copy.deepcopy(self.scene.trainset[0]["cam_info"].K)
        height, width = self.scene.trainset[0]["image"].shape[:2]
        for start_index, overlap in zip(start_indices, overlaps):
            end_index = min(start_index + int(self.args.num_frames / 3) + 1, len(self.random_pose))
            novel_views = self.sample_novel_views(start_index, end_index)

            ref_frames0, ref_c2ws0 = self.get_ref_views([novel_views[0]], k=1)
            ref2ellipse_c2ws = interp_traj(np.stack([ref_c2ws0[0], np.linalg.inv(novel_views[0].w2c)]), n_inserts=int(self.args.num_frames / 3), device=self.args.device)

            ref_frames1, ref_c2ws1 = self.get_ref_views([novel_views[-1]], k=1)
            ellipse2ref_c2ws = interp_traj(np.stack([np.linalg.inv(novel_views[-1].w2c), ref_c2ws1[-1]]), n_inserts=int(self.args.num_frames / 3), device=self.args.device)

            ref2ellipse_c2ws_views = []
            for i in range(len(ref2ellipse_c2ws)):
                cam_info = CameraInfo(
                    uid=None,
                    colmapid=None,
                    K=K,
                    w2c=np.linalg.inv(ref2ellipse_c2ws[i]),
                    image_name=None,
                    image_path=None,
                    width=width,
                    height=height,
                )
                ref2ellipse_c2ws_views.append(cam_info)

            ellipse2ref_c2ws_views = []
            for i in range(len(ellipse2ref_c2ws)):
                cam_info = CameraInfo(
                    uid=None,
                    colmapid=None,
                    K=K,
                    w2c=np.linalg.inv(ellipse2ref_c2ws[i]),
                    image_name=None,
                    image_path=None,
                    width=width,
                    height=height,
                )
                ellipse2ref_c2ws_views.append(cam_info)

            novel_views = ref2ellipse_c2ws_views + novel_views + ellipse2ref_c2ws_views
            # novel_views = ref2ellipse_c2ws_views
            artifact_rgb = self.get_render_results(novel_views)
           
            ref_frames0 = torch.nn.functional.interpolate(
                    ref_frames0,
                    size=(artifact_rgb[0].shape[1], artifact_rgb[0].shape[2]),
                    mode="bilinear")    
            artifact_rgb[0] = ref_frames0

            ref_frames1 = torch.nn.functional.interpolate(
                    ref_frames1,
                    size=(artifact_rgb[0].shape[1], artifact_rgb[0].shape[2]),
                    mode="bilinear")    
            artifact_rgb[-1] = ref_frames1

            repaired_rgb = self.repair(artifact_rgb)
            
            self.add_trainset(novel_views, repaired_rgb)
