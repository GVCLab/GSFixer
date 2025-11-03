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

from argparse import ArgumentParser, Namespace
import sys
import os
import pickle


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                elif t == list:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, nargs="+"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class DataParams(ParamGroup):
    def __init__(self, parser):
        self.data_dir = "data/360_v2/garden"
        self.data_factor = 4
        self.test_every = 8
        self.global_scale = 1.0
        self.patch_size = []
        self.depth_loss = False
        self.sparse_view = 0
        self.batch_size = 1

        self.init_type = "sfm"
        self.init_num_pts = 100_000
        self.init_extent = 3.0  # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm

        super().__init__(parser, "Loading Parameters")


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._model_path = ""
        self._images = "../images_4"
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = True
        self.n_views = 0
        # self.render_items = ["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"]
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        # self.opacity_reset_interval = 3100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        # self.densify_from_iter = 30_000
        # self.densify_until_iter = 30_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")


def load_args_from_file(filepath):
    """Load arguments from a file using pickle deserialization.

    Args:
        filepath: Path to the arguments file

    Returns:
        Namespace object containing the loaded arguments
    """
    try:
        with open(filepath, "rb") as f:  # Note: using 'rb' for binary read mode
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]

    args_cmdline = parser.parse_args(cmdlne_string)

    cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")

    args_cfgfile = load_args_from_file(cfgfilepath)
    print(args_cfgfile)
    if args_cfgfile is None:
        return args_cmdline
    else:
        merged_dict = {}
        for k, v in vars(args_cmdline).items():
            merged_dict[k] = v
        for k, v in vars(args_cfgfile).items():
            if k != "model" and k != "dataset" and k != "opt" and k != "pipe":
                merged_dict[k] = v

        for k, v in vars(args_cfgfile.model).items():
            merged_dict[k] = v
        for k, v in vars(args_cfgfile.dataset).items():
            merged_dict[k] = v
        for k, v in vars(args_cfgfile.opt).items():
            merged_dict[k] = v
        for k, v in vars(args_cfgfile.pipe).items():
            merged_dict[k] = v
        print(merged_dict)
        return Namespace(**merged_dict)
