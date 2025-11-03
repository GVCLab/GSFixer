#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import lpips
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui

from datasets import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import Namespace
from utils.camera_utils import Camera, SampleCamera
import numpy as np
import torchvision.utils as vutils

## gsfixer
from gsfixer.cogvideo.inference import GSFixer

from args import config_args
from gsfixer.extrapolation.outpaint.crop import OutpaintCrop
from gsfixer.extrapolation.outpaint.sparse import OutpaintSparse
from gsfixer.extrapolation.outpaint.rotation import OutpaintRotation
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class TrainManager:
    def __init__(self, args):
        self.args = args
        self.tb_writer = self.init_logger()
        if self.args.repair:
            self.extrapolator = GSFixer(self.args)
        else:
            self.extrapolator = None
        self.scene = self.init_scene()
        self.init_constants()
        self.init_lpips_loss()
        self.init_ema_log()
        print(self.print_args())

    def print_args(self):
        output = []
        output.append("\nArguments:")
        output.append("-" * 50)

        # Dataset args
        output.append("\nDataset Parameters:")
        for key, value in vars(self.args.dataset).items():
            output.append(f"{key}: {value}")

        # Model args
        output.append("\nModel Parameters:")
        for key, value in vars(self.args.model).items():
            output.append(f"{key}: {value}")

        # Optimization args
        output.append("\nOptimization Parameters:")
        for key, value in vars(self.args.opt).items():
            output.append(f"{key}: {value}")

        # Pipeline args
        output.append("\nPipeline Parameters:")
        for key, value in vars(self.args.pipe).items():
            output.append(f"{key}: {value}")

        # Other args
        output.append("\nOther Parameters:")
        for key, value in vars(self.args).items():
            if key not in ["dataset", "model", "opt", "pipe"]:
                output.append(f"{key}: {value}")

        output.append("-" * 50)

        print("\n".join(output))

    def init_logger(self):
        tb_writer = prepare_output_and_logger(self.args)
        return tb_writer

    def init_scene(self):
        if self.args.outpaint_type == "crop":
            gaussians = GaussianModel(self.args.model.sh_degree, sparse_aware=True)
        else:
            gaussians = GaussianModel(self.args.model.sh_degree)
        scene = Scene(
            self.args.model.model_path, self.args, gaussians, self.extrapolator
        )
        gaussians.training_setup(self.args.opt)
        if self.args.start_checkpoint:
            (model_params, first_iter) = torch.load(self.args.start_checkpoint)
            gaussians.restore(model_params, self.args.opt)
        return scene

    def init_constants(self):
        bg_color = [1, 1, 1] if self.args.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.args.background = self.background

    def init_lpips_loss(self):
        self.lpips_loss_fun = lpips.LPIPS(net="vgg").cuda()

    def init_ema_log(self):
        self.ema_loss_for_log = 0.0
        self.ema_Ll1depth_for_log = 0.0

    # train from start_iter to end_iter (inclusive)
    def train(self, start_iter, end_iter):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        use_sparse_adam = SPARSE_ADAM_AVAILABLE 
        depth_l1_weight = get_expon_lr_func(self.args.opt.depth_l1_weight_init, self.args.opt.depth_l1_weight_final, max_steps=self.args.opt.iterations)

        progress_bar = tqdm(
            range(start_iter, end_iter),
            initial=start_iter,
            total=end_iter,
            desc="Training progress",
        )
        start_iter += 1

        for iteration in range(start_iter, end_iter + 1):
            iter_start.record()
            self.scene.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.scene.gaussians.oneupSHdegree()

            viewpoint = self.scene.getTrainInstant()
            viewpoint_cam = Camera(viewpoint["cam_info"])

            render_pkg = render(viewpoint_cam, self.scene.gaussians, self.args.pipe, self.background, use_trained_exp=self.args.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]           
            gt_image = viewpoint["image"][0].permute(2, 0, 1).cuda()

            Ll1 = l1_loss(image, gt_image)
            psnr_value = 20 * torch.log10(1.0 / torch.sqrt(Ll1**2))
            ssim_value = ssim(image, gt_image)

            loss = (1.0 - self.args.opt.lambda_dssim) * Ll1 + self.args.opt.lambda_dssim * (1.0 - ssim_value)
            # loss
            total_loss = loss

            # Depth regularization
            Ll1depth_pure = 0.0
            viewpoint_cam.depth_reliable = False
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0


            # try:
            if (
                self.args.repair
                and iteration > self.args.start_diffusion_iter
                and iteration < self.args.diffusion_until
            ):
                reg_data = next(self.scene.regset)
                reg_gt_image = reg_data["image"].permute(2, 0, 1).cuda()
                cam_info = reg_data["cam_info"]
                reg_cam = SampleCamera(cam_info)

                reg_render_pkg = render(reg_cam, self.scene.gaussians, self.args.pipe, self.background, use_trained_exp=self.args.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                reg_image = reg_render_pkg["render"]
                viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                reg_image = torch.nn.functional.interpolate(
                    reg_image.unsqueeze(0),
                    size=(reg_gt_image.shape[1], reg_gt_image.shape[2]),
                    mode="bilinear",
                ).squeeze(0)

                Ll1 = l1_loss(reg_image, reg_gt_image)
                psnr_value = 20 * torch.log10(1.0 / torch.sqrt(Ll1**2))
                ssim_value = ssim(image, gt_image)
                reg_loss = (1.0 - self.args.opt.lambda_dssim) * Ll1 + self.args.opt.lambda_dssim * (1.0 - ssim_value)
                reg_weight = self.args.lambda_reg * np.sin(
                    (iteration - self.args.start_diffusion_iter)
                    / (self.args.diffusion_until - self.args.start_diffusion_iter)
                    * np.pi
                )
                total_loss += reg_weight * reg_loss

            total_loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                self.ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * self.ema_Ll1depth_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}", "PSNR": f"{psnr_value:.{2}f}", "Depth Loss": f"{self.ema_Ll1depth_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.args.opt.iterations:
                    progress_bar.close()

                if (iteration in self.args.save_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification
                if iteration < self.args.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.scene.gaussians.max_radii2D[visibility_filter] = torch.max(self.scene.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.scene.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.args.densify_from_iter and iteration % self.args.densification_interval == 0:
                        size_threshold = 20 if iteration > self.args.opacity_reset_interval else None
                        self.scene.gaussians.densify_and_prune(self.args.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold, radii)
                    
                    if iteration % self.args.opacity_reset_interval == 0 or (self.args.white_background and iteration == self.args.densify_from_iter):
                        self.scene.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.args.iterations:
                    self.scene.gaussians.exposure_optimizer.step()
                    self.scene.gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        self.scene.gaussians.optimizer.step(visible, radii.shape[0])
                        self.scene.gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        self.scene.gaussians.optimizer.step()
                        self.scene.gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in self.args.checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.scene.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    def run_with_repair(self):
        sample_steps = [0] + list(
            range(
                self.args.start_diffusion_iter,
                self.args.diffusion_until + 1,
                self.args.diffusion_every,
            )
        )
        print("sample steps", sample_steps)
        print(f"Sample steps: {sample_steps}")
        for i in range(len(sample_steps) - 1):
            start_iter = sample_steps[i]
            end_iter = sample_steps[i + 1]
            if start_iter >= self.args.diffusion_until:
                self.scene.save(start_iter)
                break

            self.train(start_iter, min(end_iter, self.args.iterations))
            print(f"Finished training from {start_iter} to {end_iter}")
            if end_iter >= self.args.iterations:
                self.scene.save(end_iter)
                break
            self.scene.regset.clear()
            outpaint = None
            if self.args.outpaint_type == "crop":
                outpaint = OutpaintCrop(self.extrapolator, self.scene, self.args)
            elif self.args.outpaint_type == "sparse":
                outpaint = OutpaintSparse(self.extrapolator, self.scene, self.args)
            elif self.args.outpaint_type == "rotation":
                outpaint = OutpaintRotation(self.extrapolator, self.scene, self.args)

            outpaint.run(i)
        if sample_steps[-1] < self.args.iterations:
            self.train(sample_steps[-1], self.args.iterations)

    def run(self):
        if not self.args.repair:
            self.train(0, self.args.iterations)
        else:
            self.run_with_repair()


def save_args_to_file(args, filepath):
    with open(filepath, "wb") as f:  # Note: using 'wb' for binary write mode
        pickle.dump(args, f)


def prepare_output_and_logger(args):
    if not args.model.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model.model_path))
    os.makedirs(args.model.model_path, exist_ok=True)
    print(args)
    save_args_to_file(args, os.path.join(args.model.model_path, "cfg_args"))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = config_args()
    print("Optimizing " + args.model.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    train_manager = TrainManager(args)
    train_manager.run()

    # All done
    print("\nTraining complete.")
