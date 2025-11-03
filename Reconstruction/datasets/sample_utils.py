import random
import copy
import numpy as np
import torch
import imageio


def rotate_camera_detailed(camera_to_world, angle_degrees):
    position = camera_to_world[:3, 3]

    forward = camera_to_world[:3, 2]

    up = camera_to_world[:3, 1]

    angle_radians = torch.tensor(np.radians(angle_degrees), dtype=torch.float32)
    rotation_3x3 = torch.tensor(
        [
            [torch.cos(angle_radians), 0, torch.sin(angle_radians)],
            [0, 1, 0],
            [-torch.sin(angle_radians), 0, torch.cos(angle_radians)],
        ],
        dtype=torch.float32,
    )

    new_forward = rotation_3x3 @ forward.float()

    right = torch.cross(new_forward, up)
    right = right / torch.norm(right)  # Normalize

    new_up = torch.cross(right, new_forward)
    new_up = new_up / torch.norm(new_up)  # Normalize

    new_camera_to_world = torch.eye(4)
    new_camera_to_world[:3, 0] = right
    new_camera_to_world[:3, 1] = new_up
    new_camera_to_world[:3, 2] = new_forward
    new_camera_to_world[:3, 3] = position

    return new_camera_to_world


def sample_novel_views(
    trainset, current_step, max_steps, start_index, rotation_degree=0
):
    num_train_views = len(trainset.indices)
    # Ensure we have at least 16 views to sample from
    if num_train_views < 16:
        raise ValueError("Not enough training views to sample 16 consecutive views.")

    # Select 16 consecutive views starting from the calculated start index
    # perturb the views
    novel_views = []
    orig_img_width, orig_img_height = list(trainset.parser.imsize_dict.values())[0]
    new_x = 0  # np.random.randint(0, max(orig_img_width - 512, 1))
    new_y = 0  # np.random.randint(0, max(orig_img_height - 320, 1))
    scale = min(current_step / max_steps, 1.0)

    print(f"start_index: {start_index}", scale)

    for i in range(start_index, start_index + 16):
        train_view = trainset[i]
        novel_view = {}
        for key in train_view.keys():
            if key == "reg_data":
                continue
            if isinstance(train_view[key], int):
                novel_view[key] = train_view[key]
            else:
                novel_view[key] = copy.deepcopy(train_view[key].detach())

        camtoworld = novel_view["camtoworld"]

        new_camtoworld = rotate_camera_detailed(camtoworld, rotation_degree)

        novel_view["camtoworld"] = new_camtoworld

        K = novel_view["K"]

        x_start, y_start = (
            trainset.trajectories[i] if hasattr(trainset, "trajectories") else (0, 0)
        )
        K[0, 2] += x_start
        K[1, 2] += y_start

        K[0, 2] -= new_x
        K[1, 2] -= new_y
        novel_view["K"] = K
        novel_views.append(novel_view)
    ref_frames = get_reference_frames(
        trainset, start_index, num_train_views, new_y, new_x
    )
    return novel_views, ref_frames


def get_reference_frames(trainset, start_index, num_train_views, new_y, new_x):
    # Select two reference frames
    ref_frame1 = torch.randint(0, max(1, start_index), (1,)).item()
    ref_frame2 = torch.randint(
        min(start_index + 16, num_train_views - 2), num_train_views - 1, (1,)
    ).item()

    # Load and process reference images
    ref_frames = []
    for frame in [ref_frame1, ref_frame2]:
        ref_img_idx = trainset.indices[frame]
        ref_image = imageio.imread(trainset.parser.image_paths[ref_img_idx])[..., :3]
        # ref_image = ref_image[new_y : new_y + 320, new_x : new_x + 512]
        ref_image = torch.from_numpy(ref_image).float()
        ref_frames.append(ref_image)

    # Stack reference frames
    ref_frames = torch.stack(ref_frames)  # Shape: [2, 320, 512, 3]

    return ref_frames
