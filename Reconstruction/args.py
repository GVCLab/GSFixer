from argparse import ArgumentParser
from arguments import DataParams, ModelParams, PipelineParams, OptimizationParams
import sys
import torch


def config_args():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    dp = DataParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6012)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--repair", action="store_true")
    # parser.add_argument("--mono_depth", action="store_true")
    # parser.add_argument("--diffusion_ckpt", type=str, default=None)
    parser.add_argument("--diffusion_config", type=str, default=None)
    parser.add_argument("--diffusion_resize_width", type=int, default=512)
    parser.add_argument("--diffusion_resize_height", type=int, default=320)
    parser.add_argument("--start_diffusion_iter", type=int, default=1000)
    parser.add_argument("--diffusion_every", type=int, default=1000)
    parser.add_argument("--diffusion_until", type=int, default=7000)
    parser.add_argument(
        "--outpaint_type",
        type=str,
        choices=["crop", "segment", "sparse", "rotation"],
        default="crop",
    )
    parser.add_argument("--add_indices", type=int, nargs="+", default=[])
    parser.add_argument("--initize_points", action="store_true")
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--wo_crop", action="store_true")
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--position_z_offset", type=float, default=0.0)
    parser.add_argument("--rotation_angle", type=float, default=16.0)
    parser.add_argument("--path_scale", type=float, default=0.7)
    parser.add_argument("--camera_path_file", type=str, default=None)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=3.2)
    # parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--start_dist_iter", type=int, default=5000)

    # gsfixer
    parser.add_argument('--blip_path', type=str, default="./gsfixer_vivo/checkpoints/blip2-opt-2.7b")
    parser.add_argument('--dinov2_ckpt', type=str, default="./gsfixer_vivo/dinov2-with-registers-large")
    parser.add_argument('--vggt_ckpt', type=str, default="./gsfixer_vivo/vggt/VGGT-1B/model.pt")
    parser.add_argument(
        '--model_name',
        type=str,
        default='./gsfixer_vivo/CogVideoX-5b-I2V',
        help='Path to the model',
    )
    parser.add_argument(
        '--transformer_path',
        type=str,
        default="./gsfixer_vivo/CogVideoX-5b-I2V/transformer",
        help='Path to the pretrained transformer model',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[480, 720],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--diffusion_guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--diffusion_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps',
    )
    parser.add_argument(
        '--sampler_name',
        type=str,
        choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default='DDIM_Origin',
        help='Choose the sampler',
    )
    parser.add_argument(
        '--low_gpu_memory_mode',
        type=bool,
        default=False,
        help='Enable low GPU memory mode',
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--weight_dtype', default=torch.bfloat16, help='weight_dtype to use'
    )
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--diffusion_crop_width", type=int, default=720)
    parser.add_argument("--diffusion_crop_height", type=int, default=480)
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    parser.add_argument(
        '--seed', type=int, default=43, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument('--save_dir', type=str, default="/data/vjuicefs_ai_camera_3drg/public_data/yxyl/ObjectCrafter/gaussian-splatting_genfusion_fixcrafter/output_temp")
    parser.add_argument(
        '--lambda_reg', type=float, default=1.0, help='weight of generation image'
    )
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.dataset = dp.extract(args)
    args.model = lp.extract(args)
    args.opt = op.extract(args)
    args.pipe = pp.extract(args)
    return args
