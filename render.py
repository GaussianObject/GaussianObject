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

import json
import os
import subprocess
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render, render_w_pose
import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix
from scene.cameras import Camera
from utils.pose_utils import get_loss_tracking, update_pose


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                render = renders[idx].cuda()
                gt = gts[idx].cuda()

                ssims.append(ssim(render, gt))
                psnrs.append(psnr(render, gt))
                lpipss.append(lpips_fn(render, gt))

            print("==FROM 3DGS==")
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

def render_set(model_path, name, iteration, views: List[Camera], gaussians, pipeline, background, save_images=True, not_generate_video=False, refine_iters=0, extra_opts=None):
    if extra_opts and extra_opts.use_dust3r:
        from gaussian_renderer import render_w_pose as render
    else:
        from gaussian_renderer import render

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ssims = 0.
    psnrs = 0.
    lpipss = 0.
    depths = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if extra_opts and extra_opts.use_dust3r:
            opt_params = []
            opt_params.append(
                {
                    "params": [view.cam_rot_delta],
                    "lr": 0.0003,
                    "name": "rot_{}".format(view.uid),
                }
            )
            opt_params.append(
                {
                    "params": [view.cam_trans_delta],
                    "lr": 0.0001,
                    "name": "trans_{}".format(view.uid),
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)
            for _ in range(refine_iters):
                render_pkg = render(view, gaussians, pipeline, background)
                image, opacity = render_pkg["render"], render_pkg["rendered_alpha"]
                pose_optimizer.zero_grad()
                # image_ab = (torch.exp(view.exposure_a)) * image + view.exposure_b
                # image_ab = image
                loss_tracking = get_loss_tracking(
                    image, opacity, view
                )
                # print(loss_tracking)
                loss_tracking.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    converged = update_pose(view)
                if converged:
                    break

        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depths.append(render_pkg["rendered_depth"].detach().cpu().numpy()[0])
        gt = view.original_image[0:3, :, :]
        ssims += ssim(rendering, gt).mean().item()
        psnrs += psnr(rendering, gt).mean().item()
        lpipss += lpips_fn(rendering, gt).item() # NCHW
        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    print(f'{name} SSIM: {ssims / len(views)}')
    print(f'{name} PSNR: {psnrs / len(views)}')
    print(f'{name} LPIPS: {lpipss / len(views)}')

    # since the eval is done in the render function, just dump the results to json
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "results.json"), 'w') as fp:
        json.dump({"SSIM": ssims / len(views), "PSNR": psnrs / len(views), "LPIPS": lpipss / len(views)}, fp, indent=True)

    # Use ffmpeg to output video
    if not not_generate_video:
        renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
        gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt.mp4")
        combined_path = os.path.join(model_path, name, "ours_{}".format(iteration), "combined.mp4")
        # Use ffmpeg to output video
        subprocess.run(["ffmpeg", "-y", 
                    "-framerate", "24",
                    "-i", os.path.join(render_path, "%05d.png"), 
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", 
                    "-crf", "23", 
                    # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
                    renders_path], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
        subprocess.run(["ffmpeg", "-y", 
                    "-framerate", "24",
                    "-i", os.path.join(gts_path, "%05d.png"), 
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", 
                    "-crf", "23", 
                    # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
                    gt_path], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
        # Concatenate the videos vertically using the `concat` filter
        command = [
            "ffmpeg","-y",
            "-i",renders_path,
            "-i",gt_path,
            "-filter_complex","[0:v][1:v]hstack=inputs=2[v]",
            "-map","[v]",
            "-c:v","libx264",
            "-crf","23",
            "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
            combined_path
        ]
        # Run the command
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL

        # Remove the original videos
        os.remove(renders_path)
        os.remove(gt_path)

        # use opencv generate depth video
        # import pdb
        # pdb.set_trace()
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth.mp4")
        depth_video = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (depths[0].shape[1], depths[0].shape[0]), False)
        for depth in depths:
            # opencv need to convert to uint8
            if depth.max() > 0:
                depth[depth <= 0] = depth[depth>0].min()
                depth_normalized = cv2.normalize(depth, depth, 0.0, 1.0, cv2.NORM_MINMAX)
            else:
                depth_normalized = np.zeros_like(depth)
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = np.uint8(depth_normalized)
            depth_video.write(depth_uint8)
        depth_video.release()
        # due to some bug, we need to use ffmpeg to convert the depth video to mp4
        subprocess.run(["ffmpeg", "-y", "-i", depth_path, "-c:v", "libx264", "-crf", "23", depth_path.replace(".mp4", "_compressed.mp4")], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(depth_path)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_all : bool, extra_opts=None):
    # with torch.no_grad():
    load_ply = None if extra_opts.load_ply == 'origin' else extra_opts.load_ply
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, extra_opts=extra_opts, load_ply=load_ply)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages, refine_iters=0, extra_opts=extra_opts)

    if not skip_test and len(scene.getTestCameras()) > 0:
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages, refine_iters=extra_opts.refine_iters, extra_opts=extra_opts)

    if not skip_all:
        render_set(dataset.model_path, "all", scene.loaded_iter, scene.getAllCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages, refine_iters=0, extra_opts=extra_opts)

@torch.no_grad()
def render_path(dataset : ModelParams, iteration : int, pipeline : PipelineParams, extra_opts=None):
    load_ply = None if extra_opts.load_ply == 'origin' else extra_opts.load_ply
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, extra_opts=extra_opts, load_ply=load_ply)

    iteration = scene.loaded_iter

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    model_path = dataset.model_path
    name = "render"

    views = scene.getRenderCameras()

    # print(len(views))
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    depths = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if args.render_resize_method == 'crop':
            image_size = 512
        elif args.render_resize_method == 'pad':
            image_size = max(view.image_width, view.image_height)
        else:
            raise NotImplementedError
        view.original_image = torch.zeros((3, image_size, image_size), device=view.original_image.device)
        focal_length_x = fov2focal(view.FoVx, view.image_width)
        focal_length_y = fov2focal(view.FoVy, view.image_height)
        view.image_width = image_size
        view.image_height = image_size
        view.FoVx = focal2fov(focal_length_x, image_size)
        view.FoVy = focal2fov(focal_length_y, image_size)
        view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)

        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depths.append(render_pkg["rendered_depth"].cpu().numpy()[0])
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    # Use ffmpeg to output video
    renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
    # Use ffmpeg to output video
    subprocess.run(["ffmpeg", "-y", 
                "-framerate", "24",
                "-i", os.path.join(render_path, "%05d.png"), 
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", 
                "-crf", "23", 
                # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
                renders_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_all", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--not_saveimages", action="store_true")
    parser.add_argument("--not_generate_video", "-ng", action="store_true")
    parser.add_argument("--is_eval", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--render_resize_method", default="crop", type=str)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument('--use_dust3r', action='store_true', default=False,
                        help='use dust3r estimated poses')
    parser.add_argument('--dust3r_json', type=str, default=None)
    parser.add_argument('--refine_iters', type=int, default=0)
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument("--load_ply", default="origin", type=str, help="Load other ply as init")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # sometimes we only want to render the images, and do not want to evaluate the metrics
    if args.is_eval:
        with torch.no_grad():
            evaluate([args.model_path])
        exit()

    if args.render_path:
        render_path(model.extract(args), args.iteration, pipeline.extract(args), extra_opts = args)
        exit()

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_all, extra_opts = args)
