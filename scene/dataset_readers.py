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

import sys
import json
import os.path as osp
from typing import NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getWorld2View2, transform_pcd
from utils.image_utils import load_meshlab_file
from utils.camera_utils import transform_cams, CameraInfo, generate_ellipse_path_from_camera_infos


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, extra_opts=None):
    cam_infos = []

    # direct load resized images, not the original ones
    if extra_opts.resolution in [1, 2, 4, 8]:
        tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
        if not osp.exists(tmp_images_folder):
            print(f"The {tmp_images_folder} is not found, use original resolution images")
        else:
            print(f"Using resized images in {tmp_images_folder}...")
            images_folder = tmp_images_folder
    else:
        print("use original resolution images")

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE": 
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = osp.join(images_folder, osp.basename(extr.name))
        image_name = osp.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        ### load masks
        mask_path_png = osp.join(osp.dirname(images_folder), "masks", osp.basename(
            image_path).replace(osp.splitext(osp.basename(image_path))[-1], '.png'))

        if osp.exists(mask_path_png) and hasattr(extra_opts, "use_mask") and extra_opts.use_mask:
            mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = None

        mono_depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, mask=mask, mono_depth=mono_depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readDust3rCamerasWithCamInfos(dust3r_frames, ori_cam_infos):
    cam_infos = []
    for frame in dust3r_frames:
        id = frame['id']
        R = np.array(frame['rotation'])
        T = np.array(frame['position'])
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        focal_length_y = frame['fy']
        focal_length_x = frame['fx']
        FovY = focal2fov(focal_length_y, ori_cam_infos[id].height)
        FovX = focal2fov(focal_length_x, ori_cam_infos[id].width)
        cam_info = CameraInfo(
            uid=ori_cam_infos[id].uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=ori_cam_infos[id].image,
            image_path=ori_cam_infos[id].image_path,
            image_name=ori_cam_infos[id].image_name,
            width=ori_cam_infos[id].width,
            height=ori_cam_infos[id].height,
            mask=ori_cam_infos[id].mask,
            mono_depth=ori_cam_infos[id].mono_depth
        )
        cam_infos.append(cam_info)
    return cam_infos

def readColmapSceneInfo(path, images, eval, llffhold=8, extra_opts=None):
    try:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=osp.join(path, reading_dir), extra_opts=extra_opts)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "sparse/0/points3D.ply")
    bin_path = osp.join(path, "sparse/0/points3D.bin")
    txt_path = osp.join(path, "sparse/0/points3D.txt")
    if not osp.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if hasattr(extra_opts, 'sparse_view_num') and extra_opts.sparse_view_num > 0: # means sparse setting
        assert eval == False
        assert osp.exists(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt")), "sparse_id.txt not found!"
        ids = np.loadtxt(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt"), dtype=np.int32)
        ids_test = np.loadtxt(osp.join(path, f"sparse_test.txt"), dtype=np.int32)
        test_cam_infos = [train_cam_infos[i] for i in ids_test]
        train_cam_infos = [train_cam_infos[i] for i in ids]
        print("Sparse view, only {} images are used for training, others are used for eval.".format(len(ids)))
    if hasattr(extra_opts, 'use_dust3r') and extra_opts.use_dust3r:
        print('use dust3r estimated camera poses...')
        if hasattr(extra_opts, 'dust3r_json') and extra_opts.dust3r_json:
            with open(extra_opts.dust3r_json) as f:
                dust3r_frames = json.load(f)
        else:
            assert osp.exists(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}.json")), f"dust3r_{str(extra_opts.sparse_view_num)}.json not found!"
            with open(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}.json")) as f:
                dust3r_frames = json.load(f)
        train_cam_infos = readDust3rCamerasWithCamInfos(dust3r_frames, train_cam_infos)
        if osp.exists(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}_test.json")):
            with open(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}_test.json")) as f:
                test_dust3r_frames = json.load(f)
            test_cam_infos = readDust3rCamerasWithCamInfos(test_dust3r_frames, test_cam_infos)
        render_cam_infos = generate_ellipse_path_from_camera_infos(train_cam_infos)
        nerf_normalization = getNerfppNorm(train_cam_infos)
    # else:
    #     render_cam_infos = generate_ellipse_path_from_camera_infos(train_cam_infos)
    #     nerf_normalization = getNerfppNorm(train_cam_infos)

    # NOTE in sparse condition, we may use random points to initialize the gaussians
    if hasattr(extra_opts, 'init_pcd_name'):
        if extra_opts.init_pcd_name == 'origin':
            pass # None just skip, use better init.
        elif extra_opts.init_pcd_name == 'random':
            raise NotImplementedError
        else:
            # use specific pointcloud, direct load it
            pcd = fetchPly(osp.join(path, extra_opts.init_pcd_name if extra_opts.init_pcd_name.endswith(".ply") 
                                        else extra_opts.init_pcd_name + ".ply"))


    if hasattr(extra_opts, 'transform_the_world') and extra_opts.transform_the_world:
        """
            a experimental feature, we use the transform matrix to transform the pointcloud and the camera poses
        """
        assert osp.exists(osp.join(path, "pcd_transform.txt")), "pcd_transform.txt not found!"
        print("*"*10 , "The world is transformed!!!", "*"*10)
        MLMatrix44 = load_meshlab_file(osp.join(path, "pcd_transform.txt"))
        # this is a 4x4 matrix for transform the pointcloud, new_pc_xyz = (MLMatrix44 @ (homo_xyz.T)).T
        # First, we transform the input pcd, only accept BasicPCD
        assert isinstance(pcd, BasicPointCloud)
        pcd = transform_pcd(pcd, MLMatrix44)
        # then, we need to rotate all the camera poses
        train_cam_infos = transform_cams(train_cam_infos, MLMatrix44)
        test_cam_infos = transform_cams(test_cam_infos, MLMatrix44) if len(test_cam_infos) > 0 else []

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readDust3rCameras(dust3r_frames, images_folder, depths_npy=None, confidences_npy=None):
    cam_infos = []
    depths = np.load(depths_npy) if depths_npy else None
    confidences = np.load(confidences_npy) if confidences_npy else None
    for frame in dust3r_frames:
        id = frame['id']
        R = np.array(frame['rotation'])
        T = np.array(frame['position'])
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        focal_length_y = frame['fy']
        focal_length_x = frame['fx']
        image_height = frame['height']
        image_width = frame['width']
        FovY = focal2fov(focal_length_y, image_height)
        FovX = focal2fov(focal_length_x, image_width)

        image_path = osp.join(images_folder, frame['img_name'])
        image_name = frame['img_name'].split(".")[0]
        image = Image.open(image_path)

        mask_path_png = osp.join(osp.dirname(images_folder), "masks", osp.basename(
            image_path).replace(osp.splitext(osp.basename(image_path))[-1], '.png'))
        mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        mask = mask.astype(np.float32) / 255.0

        mono_depth = depths[id] if depths is not None else None
        confidence = confidences[id] / confidences[id].max() if confidences is not None else None

        cam_info = CameraInfo(
            uid=frame['id'],
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=image_width,
            height=image_height,
            mask=mask,
            mono_depth=mono_depth,
            confidence=confidence,
            is_dust3r=True
        )
        cam_infos.append(cam_info)
    return cam_infos

def readDust3rSceneInfo(path, images, eval, extra_opts=None):
    images_folder = osp.join(path, "images")
    if extra_opts.resolution in [1, 2, 4, 8]:
        tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
        if not osp.exists(tmp_images_folder):
            print(f"The {tmp_images_folder} is not found, use original resolution images")
        else:
            print(f"Using resized images in {tmp_images_folder}...")
            images_folder = tmp_images_folder
    else:
        print("use original resolution images")

    ids = np.loadtxt(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt"), dtype=np.int32)

    if hasattr(extra_opts, 'dust3r_json') and extra_opts.dust3r_json:
        with open(extra_opts.dust3r_json) as f:
            dust3r_frames = json.load(f)
    else:
        with open(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}.json")) as f:
            dust3r_frames = json.load(f)
    train_cam_infos = readDust3rCameras(
        dust3r_frames, images_folder,
        osp.join(path, f"dust3r_depth_{str(extra_opts.sparse_view_num)}.npy"),
        osp.join(path, f"dust3r_confidence_{str(extra_opts.sparse_view_num)}.npy")
    )
    if osp.exists(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}_test.json")):
        with open(osp.join(path, f"dust3r_{str(extra_opts.sparse_view_num)}_test.json")) as f:
            test_dust3r_frames = json.load(f)
        test_cam_infos = readDust3rCameras(test_dust3r_frames, images_folder)
    else:
        test_cam_infos = []
    render_cam_infos = generate_ellipse_path_from_camera_infos(train_cam_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, extra_opts.init_pcd_name if extra_opts.init_pcd_name.endswith(".ply") 
                                    else extra_opts.init_pcd_name + ".ply")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "DUSt3R": readDust3rSceneInfo,
}
