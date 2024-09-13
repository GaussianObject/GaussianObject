import math
import os
import random
import json

import torch
import numpy as np
import cv2
import pytorch_lightning as pl
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, rotmat2qvec
from utils.camera_utils import resize_mask_image, load_raw_depth
from utils.graphics_utils import getWorld2View2, focal2fov
from .random_camera_sampler import RandomCameraSampler


def getNerfppNorm(cam_centers):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate.astype(np.float32), "radius": float(radius)}


@dataclass
class LooDataModuleConfig:
    batch_size: int = 1
    data_dir: str = ''
    eval_camera_distance: float = 6.
    resolution: int = 1
    prompt: str = ''
    sparse_num: int = 0
    bg_white: bool = False
    length: int = 1500
    around_gt_steps: int = 750
    refresh_interval: int = 100
    refresh_size: int = 20
    use_dust3r: bool = False
    json_path: str = ''


@register("loo-dataset")
class LooDataset(Dataset):
    def __init__(self, cfg: LooDataModuleConfig, split: str = 'train', sparse_num: int = 0):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.data_dir = self.cfg.data_dir
        self.resolution = self.cfg.resolution
        self.sparse_num = sparse_num
        self.length = self.cfg.length
        self.around_gt_steps = self.cfg.around_gt_steps
        self.refresh_interval = self.cfg.refresh_interval
        self.refresh_size = self.cfg.refresh_size

        self.sparse_ids = []
        if self.sparse_num != 0:
            if self.split == 'train' or cfg.use_dust3r:
                with open(os.path.join(self.data_dir, f'sparse_{self.sparse_num}.txt')) as f:
                    self.sparse_ids = sorted([int(id) for id in f.readlines()])
            else:
                with open(os.path.join(self.data_dir, f'sparse_test.txt')) as f:
                    self.sparse_ids = sorted([int(id) for id in f.readlines()])

        images_folder=os.path.join(self.data_dir, 'images')
        if self.resolution in [1, 2, 4, 8]:
            tmp_images_folder = images_folder + f'_{str(self.resolution)}' if self.resolution != 1 else images_folder
            if not os.path.exists(tmp_images_folder):
                threestudio.warn(f"The {tmp_images_folder} is not found, use original resolution images")
            else:
                threestudio.info(f"Using resized images in {tmp_images_folder}...")
                images_folder = tmp_images_folder
        else:
            threestudio.info("use original resolution images")
        masks_folder = os.path.join(self.data_dir, 'masks')

        self.Rs, self.Ts, self.heights, self.widths, self.fovxs, self.fovys, self.images, self.masks, self.depths, self.confidences = [], [], [], [], [], [], [], [], [], []
        cam_c = []

        if cfg.use_dust3r:
            if len(cfg.json_path):
                with open(cfg.json_path) as f:
                    dust3r_frames = json.load(f)
            else:
                with open(os.path.join(self.data_dir, f"dust3r_{self.sparse_num}.json")) as f:
                    dust3r_frames = json.load(f)
            self.Rs = []
            self.Ts = []
            self.fovxs = []
            self.fovys = []
            all_Rs = None
            all_Ts = None
            cam_c = []
            dust3r_depths = np.load(os.path.join(self.data_dir, f"dust3r_depth_{self.sparse_num}.npy"))
            dust3r_confidences = np.load(os.path.join(self.data_dir, f"dust3r_confidence_{self.sparse_num}.npy"))
            for frame in dust3r_frames:
                id = frame['id']

                image_path = os.path.join(images_folder, frame['img_name'])
                image_name = os.path.basename(image_path).split(".")[0]
                image = Image.open(image_path)

                mask_path = os.path.join(masks_folder, image_name + '.png')
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                mask = mask.astype(np.float32) / 255.0
                resized_mask = resize_mask_image(mask, image.size)
                loaded_mask = torch.from_numpy(resized_mask).unsqueeze(0)

                if split == 'train':
                    depth = dust3r_depths[id]
                    confidence = dust3r_confidences[id] / dust3r_confidences[id].max()
                else:
                    depth_path = os.path.join(os.path.dirname(images_folder), "zoe_depth", os.path.basename(
                        image_path).replace(os.path.splitext(os.path.basename(image_path))[-1], '.png'))
                    depth = np.ones_like(resized_mask)
                    confidence = np.ones_like(resized_mask)
                resized_depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_NEAREST)
                loaded_depth = torch.from_numpy(resized_depth).unsqueeze(0)
                loaded_depth[loaded_mask <= 0.5] = 0.
                resized_confidence = cv2.resize(confidence, image.size, interpolation=cv2.INTER_NEAREST)
                loaded_confidence = torch.from_numpy(resized_confidence).unsqueeze(0)
                loaded_confidence[loaded_mask <= 0.5] = 0.

                # mask image
                image = (torch.from_numpy(np.array(image))/255.).permute(2, 0, 1) # C, H, W
                image[(loaded_mask <= 0.5).expand_as(image)] = 1.0 if self.cfg.bg_white else 0.0

                self.images.append(image)
                self.masks.append(loaded_mask.squeeze())
                self.depths.append(loaded_depth.squeeze())
                self.confidences.append(loaded_confidence.squeeze())

                self.heights.append(image.shape[-2])
                self.widths.append(image.shape[-1])

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
                height = frame['height']
                width = frame['width']
                FovY = focal2fov(focal_length_y / self.resolution, self.heights[id])
                FovX = focal2fov(focal_length_x / self.resolution, self.widths[id])
                cam_c.append(np.linalg.inv(getWorld2View2(R, T))[:3, 3:4])
                self.Rs.append(R)
                self.Ts.append(T)
                self.fovxs.append(FovX)
                self.fovys.append(FovY)

        else:
            cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            cam_extrinsics_unsorted = list(cam_extrinsics.values())
            cam_extrinsics = sorted(cam_extrinsics_unsorted.copy(), key = lambda x : x.name)

            for idx, extr in enumerate(cam_extrinsics):
                if idx in self.sparse_ids:
                    intr = cam_intrinsics[extr.camera_id]
                    height = intr.height
                    width = intr.width

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

                    cam_c.append(np.linalg.inv(getWorld2View2(R, T))[:3, 3:4])
                    self.Rs.append(R)
                    self.Ts.append(T)
                    self.fovxs.append(FovX)
                    self.fovys.append(FovY)

                    image_path = os.path.join(images_folder, os.path.basename(extr.name))
                    image_name = os.path.basename(image_path).split(".")[0]
                    image = Image.open(image_path)

                    mask_path = os.path.join(masks_folder, image_name + '.png')
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    mask = mask.astype(np.float32) / 255.0
                    resized_mask = resize_mask_image(mask, image.size)
                    loaded_mask = torch.from_numpy(resized_mask).unsqueeze(0)

                    depth_path = os.path.join(os.path.dirname(images_folder), "zoe_depth", os.path.basename(
                        image_path).replace(os.path.splitext(os.path.basename(image_path))[-1], '.png'))
                    depth = load_raw_depth(depth_path)
                    resized_depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_NEAREST)
                    loaded_depth = torch.from_numpy(resized_depth).unsqueeze(0)
                    loaded_depth[loaded_mask <= 0.5] = 0.

                    # mask image
                    image = (torch.from_numpy(np.array(image))/255.).permute(2, 0, 1) # C, H, W
                    image[(loaded_mask <= 0.5).expand_as(image)] = 1.0 if self.cfg.bg_white else 0.0

                    self.images.append(image)
                    self.masks.append(loaded_mask.squeeze())
                    self.depths.append(loaded_depth.squeeze())

                    self.heights.append(image.shape[-2])
                    self.widths.append(image.shape[-1])

            all_Rs = []
            all_Ts = []
            cam_c = []
            for extr in cam_extrinsics:
                R = np.transpose(qvec2rotmat(extr.qvec))
                T = np.array(extr.tvec)

                cam_c.append(np.linalg.inv(getWorld2View2(R, T))[:3, 3:4])
                all_Rs.append(R)
                all_Ts.append(T)

        self.cameras_extent = getNerfppNorm(cam_c)
        self.camera_sampler = RandomCameraSampler(self.Rs, self.Ts, all_Rs, all_Ts)

        self.cnt = 0
        self.random_poses = []

    def refresh_random_poses(self):
        self.random_poses = []
        dis_from_gt = 0.8
        threestudio.info(f'refresh random poses with dis_drom_gt={dis_from_gt} at step {self.cnt}')
        self.random_poses = []
        while len(self.random_poses) < self.refresh_size:
            self.random_poses.extend(self.camera_sampler.sample_away_from_gt(dis_from_gt))
        self.random_poses = self.random_poses[:self.refresh_size]

    def __len__(self):
        if self.split == 'train':
            return self.cfg.length
        elif len(self.sparse_ids):
            return len(self.sparse_ids)
        else:
            return len(self.Rs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == 'train':
            idx = random.randint(0, len(self.sparse_ids) - 1)
            random_index = random.randint(0, self.refresh_size - 1)
            if self.cnt < self.around_gt_steps:
                if self.cnt % self.cfg.refresh_interval == 0:
                    self.refresh_random_poses()
                random_R, random_T = self.random_poses[random_index]
            else:
                random_R, random_T = self.camera_sampler.sample(None)
            self.cnt += 1
        else:
            theta = 2 * math.pi * idx / len(self)
            random_index = idx
            random_R, random_T = self.camera_sampler.sample(theta)
        ret = {
            "index": idx,
            "R": self.Rs[idx],
            "T": self.Ts[idx],
            "height": self.heights[idx],
            "width": self.widths[idx],
            "fovx": self.fovxs[idx],
            "fovy": self.fovys[idx],
            "image": self.images[idx],
            "mask": self.masks[idx],
            "depth": self.depths[idx],
            "txt": self.cfg.prompt,
            "random_index": random_index,
            "random_R": random_R,
            "random_T": random_T,
            "random_poses": self.random_poses,
            "gt_images": self.images,
            "gt_Ts": self.Ts,
        }
        return ret

    def get_scene_extent(self):
        return self.cameras_extent

    def norm_to_pc(self, center):
        self.Ts = [(T - center) for T in self.Ts]


@register("loo-datamodule")
class LooDataModuleFromConfig(pl.LightningDataModule):
    cfg: LooDataModuleConfig
    train_dataset: Optional[LooDataset] = None
    val_dataset: Optional[LooDataset] = None
    test_dataset: Optional[LooDataset] = None

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(LooDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = LooDataset(self.cfg, "train", sparse_num=self.cfg.sparse_num)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LooDataset(self.cfg, "val", sparse_num=self.cfg.sparse_num)
        if stage in [None, "test", "predict"]:
            self.test_dataset = LooDataset(self.cfg, "test", sparse_num=self.cfg.sparse_num)

    def norm_to_pc(self, center):
        if self.train_dataset is not None:
            self.train_dataset.norm_to_pc(center)
        if self.val_dataset is not None:
            self.val_dataset.norm_to_pc(center)
        if self.test_dataset is not None:
            self.test_dataset.norm_to_pc(center)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )
