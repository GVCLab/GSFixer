import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser(description="Test for each scene")
parser.add_argument("--device_id", type=str, default="0")
parser.add_argument("--scene_root", type=str, default="./mipnerf360")
parser.add_argument("--output_root", type=str, default="./output_test")
parser.add_argument("--n_views_list", nargs="+", type=int, default=[3, 6, 9])
parser.add_argument("--blip_path", type=str, default="./checkpoints/blip2-opt-2.7b")
parser.add_argument("--dinov2_ckpt", type=str, default="./checkpoints/dinov2-with-registers-large")
parser.add_argument("--vggt_ckpt", type=str, default="./checkpoints/vggt/VGGT-1B/model.pt")
parser.add_argument("--model_name", type=str, default="./checkpoints/CogVideoX-5b-I2V")
parser.add_argument("--transformer_path", type=str, default="./checkpoints/GSFixer")
parser.add_argument("--port", type=str, default="6000")

args = parser.parse_args()

scene_root = args.scene_root
scene_path_list = glob.glob(os.path.join(scene_root, '*'))
scene_path_list = sorted(scene_path_list)
n_views_list = args.n_views_list
output_root = args.output_root

for n_views in n_views_list:
    for scene_path in scene_path_list:
        scene_name = scene_path.split('/')[-1]
        result_dir = os.path.join(output_root, scene_name, str(n_views))
        train_cmd = (
            f"CUDA_VISIBLE_DEVICES={args.device_id} python train.py "
            f"--data_dir {scene_path} "
            f"-m {result_dir} "
            f"--blip_path {args.blip_path} "
            f"--dinov2_ckp {args.dinov2_ckpt} "
            f"--vggt_ckpt {args.vggt_ckpt} "
            f"--model_name {args.model_name} "
            f"--transformer_path {args.transformer_path} "
            f"--outpaint_type sparse "
            f"--depth_loss "
            f"--sparse_view {n_views} "
            f"--diffusion_resize_width 720 "
            f"--diffusion_resize_height 480 "
            f"--diffusion_crop_width 720 "
            f"--diffusion_crop_height 480 "
            f"--iterations 30_000 "
            f"--test_iterations 30_000 "
            f"--start_diffusion_iter 7000 "
            f"--diffusion_until 30000 "
            f"--diffusion_every 10000 "
            f"--opacity_reset_interval 3100 "
            f"--repair "
            f"--port {args.port} "
            f"--save_dir ./output_temp/{scene_name}/{n_views} "
        )
        print(train_cmd)
        os.system(train_cmd)
        print("finish train cmd", scene_name, n_views)


        render_cmd = f"CUDA_VISIBLE_DEVICES=0 python render.py -m {result_dir} --data_dir {scene_path}"
        print(render_cmd)
        os.system(render_cmd)
        print("finish render cmd", scene_name, n_views)


        metrics_cmd = f"CUDA_VISIBLE_DEVICES=0 python metrics.py -m {result_dir}"
        print(metrics_cmd)
        os.system(metrics_cmd)
        print("finish metrics_cmd", scene_name, n_views)
    
