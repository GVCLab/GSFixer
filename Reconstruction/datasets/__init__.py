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

import os, torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from datasets.gaussian_model import GaussianModel
from arguments import DataParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from datasets.colmap_debug import Dataset, Parser
from datasets.normalize import get_center_and_diag
from utils.graphics_utils import BasicPointCloud
from utils.camera_utils import Camera
from datasets.regset import RegSet


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        model_path,
        args,
        gaussians: GaussianModel,
        extrapolator=None,
        outpaint_type="crop",
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = model_path
        self.gaussians = gaussians
        self.outpaint_type = args.outpaint_type
        self.loaded_iter = None
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.parser = Parser(
            data_dir=args.dataset.data_dir,
            factor=args.dataset.data_factor,
            normalize=True,
            test_every=args.dataset.test_every,
        )
        self.trainset = Dataset(
            args,
            self.parser,
            split="train",
            patch_size=args.dataset.patch_size,
            load_depths=args.dataset.depth_loss,
            sparse_view=args.dataset.sparse_view,
            # extrapolator=extrapolator,
        )
        self.regset = RegSet()
        self.valset = Dataset(
            args, self.parser, split="val", sparse_view=args.dataset.sparse_view
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * args.dataset.global_scale

        # gs init
        if args.dataset.init_type == "sfm":
            points = torch.from_numpy(self.parser.points).float()
            rgbs = torch.from_numpy(self.parser.points_rgb).float() / 255.0
        elif args.dataset.init_type == "random":
            points = (
                args.dataset.init_extent
                * self.scene_scale
                * (torch.rand((args.dataset.init_num_pts, 3)) * 2 - 1)
            )
            rgbs = torch.rand((args.dataset.init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sfm or random")

        # estimate scene scale
        cam_o = self.parser.camtoworlds[self.trainset.indices][:, :3, 3]
        center, diagonal = get_center_and_diag(cam_o)
        self.cameras_extent = diagonal * 1.1

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            point_cloud = BasicPointCloud(points, rgbs)
            self.gaussians.create_from_pcd(point_cloud, self.trainset.data_list, self.cameras_extent)

        # init dataloader
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=args.dataset.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        self.renderloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=args.dataset.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        self.valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        self.trainloader_iter = iter(self.trainloader)
        self.valloader_iter = iter(self.valloader)
        self.renderloader_iter = iter(self.renderloader)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainInstant(self, scale=1.0):
        try:
            return next(self.trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(self.trainloader)
            return next(trainloader_iter)

    def getRenderInstant(self, scale=1.0):
        try:
            return next(self.renderloader_iter)
        except StopIteration:
            renderloader_iter = iter(self.renderloader_iter)
            return next(renderloader_iter)

    def getTestInstant(self, scale=1.0):
        try:
            return next(self.valloader_iter)
        except StopIteration:
            return None
