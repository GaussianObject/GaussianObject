import os
import cv2
import json
import pickle
import torch
import numpy as np
from random import random, choice
from typing import List, Union
from PIL import Image
from argparse import ArgumentParser
from torch.utils.data import Dataset
from gaussian_renderer import render
from scene import GaussianModel
from arguments import PipelineParams
from scene.cameras import Camera
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import focal2fov, fov2focal


imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]


def load_statistics_info(info_path):
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    return info

class GSCacheDataset(Dataset):
    def __init__(
        self,
        gaussian_dir: str,
        data_dir: str,
        loo_dir: str,
        image_size: int = 512,
        resolution: int = 4,
        sparse_num: int = 5,
        noise_scale_min: float = 0.6,
        noise_scale_max: float = 0.8,
        noise_dropout_min: float = 0.6,
        noise_dropout_max: float = 0.8,
        manual_noise_reduce_start: int = 20,
        manual_noise_reduce_gamma: float = 0.98,
        prompt: str = '',
        bg_white: bool = False,
        sh_degree: int = 0,
        use_prompt_list: bool = False,
        cache_max_iter: int = 240,
        train: bool = True,
        use_dust3r: bool = False
    ):
        super().__init__()
        self.gaussian_dir = gaussian_dir
        self.data_dir = data_dir
        self.loo_dir = loo_dir
        self.image_size = image_size
        self.resolution = resolution
        self.sparse_num = sparse_num
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max
        self.noise_dropout_min = noise_dropout_min
        self.noise_dropout_max = noise_dropout_max
        self.current_step = 0
        self.manual_noise_prob = 1.0
        self.manual_noise_reduce_start = manual_noise_reduce_start
        self.manual_noise_reduce_gamma = manual_noise_reduce_gamma
        self.prompt = prompt
        self.train = train
        self.use_dust3r = use_dust3r
        self.bg_white = bg_white
        self.bg_color = torch.tensor([1., 1., 1.] if bg_white else [0., 0., 0.] , dtype=torch.float32, device='cuda')
        self.use_prompt_list = use_prompt_list
        self.cache_max_iter = cache_max_iter

        self.iter = max([int(iter.split('_')[-1]) for iter in os.listdir(os.path.join(self.gaussian_dir, 'point_cloud'))
                         if os.path.isdir(os.path.join(self.gaussian_dir, 'point_cloud', iter)) and iter.split('_')[-1].isdigit()])

        ply_path = os.path.join(self.gaussian_dir, 'point_cloud', f'iteration_{self.iter}', 'point_cloud.ply')
        self.gaussian = GaussianModel(sh_degree=sh_degree)
        self.gaussian.load_ply(ply_path, False)
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        sparse_ids = []
        with open(os.path.join(self.data_dir, f'sparse_{self.sparse_num}.txt')) as f:
            sparse_ids = [int(id) for id in f.readlines()]

        images_folder = os.path.join(self.data_dir, 'images')
        if self.resolution in [1, 2, 4, 8]:
            tmp_images_folder = images_folder + f'_{str(self.resolution)}' if self.resolution != 1 else images_folder
            if not os.path.exists(tmp_images_folder):
                print(f"The {tmp_images_folder} is not found, use original resolution images")
            else:
                print(f"Using resized images in {tmp_images_folder}...")
                images_folder = tmp_images_folder
        else:
            print("use original resolution images")
        masks_folder = os.path.join(self.data_dir, 'masks')

        self.Rs: List[np.ndarray] = []
        self.Ts: List[np.ndarray] = []
        self.heights: List[float] = []
        self.widths: List[float] = []
        self.fovxs: List[float] = []
        self.fovys: List[float] = []
        self.images: List[np.ndarray] = []
        self.noisys: List[List[np.ndarray]] = []
        self.statistics_info = []

        if self.use_dust3r:
            # with open(os.path.join(self.data_dir, f"dust3r_{self.sparse_num}.json")) as f:
            with open(os.path.join(self.gaussian_dir, 'refined_cams.json')) as f:
                dust3r_frames = json.load(f)
            for idx, frame in zip(sparse_ids, dust3r_frames):
                image_path = os.path.join(images_folder, frame['img_name'])
                image_name = os.path.basename(image_path).split(".")[0]
                image = Image.open(image_path)
                image = np.array(image)

                mask_path = os.path.join(masks_folder, image_name + '.png')
                mask = Image.open(mask_path)
                mask = mask.resize((image.shape[1], image.shape[0]))
                mask = np.array(mask)
                image[mask < 127] = 255 if bg_white else 0

                R = np.array(frame['rotation'])
                T = np.array(frame['position'])
                c2w = np.eye(4)
                c2w[:3, :3] = R
                c2w[:3, 3] = T
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3].T
                T = w2c[:3, 3]
                height = frame['height']
                width = frame['width']
                focal_length_y = frame['fy']
                focal_length_x = frame['fx']
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)

                self.Rs.append(R)
                self.Ts.append(T)
                self.heights.append(height)
                self.widths.append(width)
                self.fovxs.append(FovX)
                self.fovys.append(FovY)
                self.images.append(image)

                noisy_paths = os.listdir(os.path.join(self.loo_dir, f'leave_{idx}', 'left_image'))
                its = sorted([int(path.replace('sample_', '').replace('.png', '')) for path in noisy_paths])
                min_it = its[0]
                noisys = [np.array(Image.open(os.path.join(self.loo_dir, f'leave_{idx}', 'left_image', f'sample_{it}.png'))) for it in its if it < min_it + self.cache_max_iter]
                self.noisys.append(noisys)
                print(f'Load {len(noisys)} images for leave {idx}')
                if os.path.exists(os.path.join(self.loo_dir, f'leave_{idx}', "diffs.pkl")):
                    self.statistics_info.append(load_statistics_info(os.path.join(self.loo_dir, f'leave_{idx}', "diffs.pkl")))

        else:
            cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            cam_extrinsics_unsorted = list(cam_extrinsics.values())
            cam_extrinsics = sorted(cam_extrinsics_unsorted.copy(), key = lambda x : x.name)

            for idx, extr in enumerate(cam_extrinsics):
                if (self.train and idx in sparse_ids) or (not self.train and idx not in sparse_ids):
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
                    
                    image_path = os.path.join(images_folder, os.path.basename(extr.name))
                    image_name = os.path.basename(image_path).split(".")[0]
                    image = Image.open(image_path)
                    image = np.array(image)

                    mask_path = os.path.join(masks_folder, image_name + '.png')
                    mask = Image.open(mask_path)
                    mask = mask.resize((image.shape[1], image.shape[0]))
                    mask = np.array(mask)
                    image[mask < 127] = 255 if bg_white else 0

                    self.Rs.append(R)
                    self.Ts.append(T)
                    self.heights.append(height)
                    self.widths.append(width)
                    self.fovxs.append(FovX)
                    self.fovys.append(FovY)
                    self.images.append(image)

                    if self.train:
                        noisy_paths = os.listdir(os.path.join(self.loo_dir, f'leave_{idx}', 'left_image'))
                        its = sorted([int(path.replace('sample_', '').replace('.png', '')) for path in noisy_paths])
                        min_it = its[0]
                        noisys = [np.array(Image.open(os.path.join(self.loo_dir, f'leave_{idx}', 'left_image', f'sample_{it}.png'))) for it in its if it < min_it + self.cache_max_iter]
                        self.noisys.append(noisys)
                        print(f'Load {len(noisys)} images for leave {idx}')
                        if os.path.exists(os.path.join(self.loo_dir, f'leave_{idx}', "diffs.pkl")):
                            self.statistics_info.append(load_statistics_info(os.path.join(self.loo_dir, f'leave_{idx}', "diffs.pkl")))

    def __len__(self):
        return len(self.images)

    def center_pad(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        H, W, _ = image.shape
        pad_l = max((self.image_size - W) // 2, 0)
        pad_r = max(self.image_size - W - pad_l, 0)
        pad_u = max((self.image_size - H) // 2, 0)
        pad_d = max(self.image_size - H - pad_u, 0)
        if isinstance(image, torch.Tensor):
            return torch.nn.functional.pad(image, (0, 0, pad_l, pad_r, pad_u, pad_d), mode='constant', value=1. if self.bg_white else 0.)
        else:
            return np.pad(image, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)), mode='constant', constant_values=1. if self.bg_white else 0.)

    def center_crop(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        image = self.center_pad(image)
        H, W, _ = image.shape
        up = H // 2 - self.image_size // 2
        down = up + self.image_size
        left = W // 2 - self.image_size // 2
        right = left + self.image_size
        return image[up:down, left:right, :]

    def resize_image(self, input_image: np.ndarray) -> np.ndarray:
        H, W, _ = input_image.shape
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_AREA)
        return img

    @torch.no_grad()
    def __getitem__(self, idx):
        image = self.images[idx]
        cam = Camera(
            colmap_id = 0,
            R = self.Rs[idx],
            T = self.Ts[idx],
            FoVx = self.fovxs[idx],
            FoVy = self.fovys[idx],
            image = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0).permute(2, 0, 1),
            gt_alpha_mask = None,
            mono_depth = None,
            image_name = 'tmp.png',
            uid = 0
        )

        if not self.train:
            render_pkg = render(cam, self.gaussian, self.pipe, self.bg_color)
            noisy = render_pkg['render']
            source = self.center_crop(noisy.clamp(0., 1.).permute(1, 2, 0))
        elif random() < self.manual_noise_prob:
            noise_dropout = random() * (self.noise_dropout_max - self.noise_dropout_min) + self.noise_dropout_min
            noise_scale = random() * (self.noise_scale_max - self.noise_scale_min) + self.noise_scale_min
            self.gaussian.add_statistics_noise(self.statistics_info, noise_dropout, noise_scale)
            render_pkg = render(cam, self.gaussian, self.pipe, self.bg_color)
            noisy = render_pkg['render']
            self.gaussian.restore_noise()
            source = self.center_crop(noisy.clamp(0., 1.).permute(1, 2, 0))
        else:
            noisy = choice(self.noisys[idx])
            source = self.center_crop(noisy.astype(np.float32) / 255)

        target = self.center_crop(image.astype(np.float32) / 127.5 - 1.0)

        self.current_step += 1
        if self.current_step >= self.manual_noise_reduce_start:
            self.manual_noise_prob = self.manual_noise_prob * self.manual_noise_reduce_gamma

        return {
            'jpg': target,
            'txt': choice(imagenet_templates_small).format(self.prompt) if self.use_prompt_list else f'a photo of a {self.prompt}',
            'hint': source
        }
