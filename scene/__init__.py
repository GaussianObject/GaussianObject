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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import time


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], extra_opts=None, load_ply=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # type: ignore
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration and load_ply is None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.render_cameras = {}

        if hasattr(extra_opts, 'use_dust3r') and extra_opts.use_dust3r: # type: ignore
            scene_info = sceneLoadTypeCallbacks["DUSt3R"](args.source_path, args.images, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "sparse")): # type: ignore
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "transforms_alignz_train.json")): # type: ignore
            print("Found transforms_alignz_train.json file, assuming OpenIllumination data set!")
            scene_info = sceneLoadTypeCallbacks["OpenIllumination"](args.source_path, args.white_background, args.eval, extra_opts=extra_opts) # type: ignore
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and load_ply is None:
            # NOTE :this dump use the file name, we dump the SceneInfo.pcd as the input.ply
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.render_cameras:
                camlist.extend(scene_info.render_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            init_time = time.time()
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, mode="train")
            init_time2 = time.time()
            print("Loading training cameras with {}s".format(init_time2 - init_time))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            init_time3 = time.time()
            print("Loading test cameras with {}s".format(time.time() - init_time2))
            self.render_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.render_cameras, resolution_scale, args)
            print("Loading render cameras with {}s".format(time.time() - init_time3))

        if self.loaded_iter:
            load_name = "point_cloud.ply"
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), load_name))
        elif load_ply:
            self.gaussians.load_ply(load_ply)
            # in this case, we need it to be trainable, so we need to make sure the spatial_lr_scale is not 0
            self.gaussians.spatial_lr_scale = self.cameras_extent
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.save_ply(os.path.join(self.model_path, "input.ply"))

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getAllCameras(self, scale=1.0):
        return self.train_cameras[scale] + self.test_cameras[scale]

    def getRenderCameras(self, scale=1.0):
        return self.render_cameras[scale]
