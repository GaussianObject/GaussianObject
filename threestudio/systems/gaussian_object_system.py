import os
import io
import threestudio
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import cv2
import einops
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from PIL import Image
import clip
from gaussian_renderer import render
from scene import GaussianModel
from arguments import PipelineParams, OptimizationParams
from scene.cameras import Render_Camera
from utils.sh_utils import SH2RGB
from utils.loss_utils import l1_loss, l2_loss, ssim, monodisp
from utils.graphics_utils import focal2fov, fov2focal
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import save_image, make_grid
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM, LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional.regression import pearson_corrcoef
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from minlora import add_lora, LoRAParametrization
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *


@torch.no_grad()
def process(
    model,
    ddim_sampler: DDIMSampler,
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str = '',
    n_prompt: str = '',
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 50,
    guess_mode: bool = False,
    strength: float = 1.0,
    scale: float = 1.0,
    eta: float = 1.0,
    denoise_strength: float = 1.0
):
    input_image = HWC3(input_image)
    detected_map = input_image.copy()

    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
    img = torch.stack([img for _ in range(num_samples)], dim=0)
    img = einops.rearrange(img, 'b h w c -> b c h w').clone()

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

    ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)
    t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
    z = model.get_first_stage_encoding(model.encode_first_stage(img))
    z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]


    alphas = ddim_sampler.alphas_cumprod.cuda()
    sds_w = (1 - alphas[t_enc]).view(-1, 1)

    return results, sds_w


def compute_tv_norm(values: torch.Tensor, losstype='l2') -> torch.Tensor:
    v00 = values[:, :-1, :-1]
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')
    return loss


def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)


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


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        sh = np.random.random((vertices.count, 3)) / 255.0
        colors = SH2RGB(sh)
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
    

@threestudio.register("gaussian-object-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        sparse_num: int = 5
        model_name: str = "control_v11f1e_sd15_tile"
        exp_name: str = ""
        lora_name: str = "lora-step=1799.ckpt"
        lora_rank: int = 64
        add_diffusion_lora: bool = True
        add_control_lora: bool = True
        add_clip_lora: bool = True
        around_gt_steps: int = 0
        scene_extent: float = 5.0
        min_strength: float = 0.1
        max_strength: float = 1.0
        novel_image_size: int = 512
        refresh_interval: int = 100
        refresh_size: int = 20
        controlnet_num_samples: int = 1
        sh_degree: int = 2

        ctrl_steps: int = 1000
        ctrl_loss_ratio_begin: float = 1.0
        ctrl_loss_ratio_final: float = 0.5

    cfg: Config
    def configure(self) -> None:
        self.gaussian = GaussianModel(sh_degree = self.cfg.sh_degree)
        self.cameras_extent = self.cfg.scene_extent
        self.bg_color = [1, 1, 1] if True else [0, 0, 0]
        self.background_tensor = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.init_dreamer = self.cfg.init_dreamer
        self.point_cloud = self.init_pointcloud(self.init_dreamer)

        # metrics
        self.psnr = PSNR().to("cuda")
        self.ssim = SSIM().to("cuda")
        self.lpips = LPIPS('vgg').to("cuda")
        self.lpips_loss = LPIPS('vgg').to("cuda")

        # data type align
        self.pil_to_tensor = ToTensor()
        self.tensor_to_pil = ToPILImage()

        # controlnet cache
        self.controlnet_outs: List[torch.Tensor] = []
        self.sds_ws: List[torch.Tensor] = []
        self.all_T: torch.Tensor = torch.zeros((0, 3))
        self.max_cam_dis: float = 0.

        # clip model
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.gt_features_all = []

        # lr scheduler
        self.novel_image_size = self.cfg.novel_image_size
        self.ctrl_steps = self.cfg.ctrl_steps
        self.ctrl_loss_ratio_begin = self.cfg.ctrl_loss_ratio_begin
        self.ctrl_loss_ratio_final = self.cfg.ctrl_loss_ratio_final
        self.ctrl_loss_ratio = self.ctrl_loss_ratio_begin


    def save_gif_to_file(self, images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())


    def update_learning_rate(self):
        if self.global_step < self.ctrl_steps:
            self.ctrl_loss_ratio = self.ctrl_loss_ratio_begin + (self.ctrl_loss_ratio_final - self.ctrl_loss_ratio_begin) * self.global_step / self.ctrl_steps
        else:
            self.ctrl_loss_ratio = 0.0
        self.log("train/ctrl_loss_ratio", self.ctrl_loss_ratio)


    def cal_loss(self, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid"):
        """
        Calculate the loss of the image, contains l1 loss and ssim loss.
        l1 loss: Ll1 = l1_loss(image, gt_image)
        ssim loss: Lssim = 1 - ssim(image, gt_image)
        Optional: [silhouette loss, monodepth loss]
        """
        gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
        if self.opt.random_background:
            gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
        Ll1 = torch.nan_to_num(l1_loss(image, gt_image))
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if silhouette_loss_type == "bce":
            silhouette_loss = torch.nan_to_num(F.binary_cross_entropy(render_pkg["rendered_alpha"][0], viewpoint_cam.mask))
        elif silhouette_loss_type == "mse":
            silhouette_loss = torch.nan_to_num(F.mse_loss(render_pkg["rendered_alpha"][0], viewpoint_cam.mask))
        else:
            raise NotImplementedError
        loss = loss + self.opt.lambda_silhouette * silhouette_loss

        if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:
            if mono_loss_type == "mid":
                # we apply masked monocular loss
                gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
                render_mask = torch.where(render_pkg["rendered_alpha"][0] > 0.5, True, False)
                mask = torch.logical_and(gt_mask, render_mask)
                if mask.sum() < 10:
                    depth_loss = 0.0
                else:
                    disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                    disp_render = 1 / render_pkg["rendered_depth"][0][mask].clamp(1e-6) # shape: [N]
                    depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
            elif mono_loss_type == "pearson":
                zoe_depth = viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6)
                rendered_depth = render_pkg["rendered_depth"][0][viewpoint_cam.mask > 0.5].clamp(1e-6)
                depth_loss = torch.nan_to_num(min(
                    (1 - pearson_corrcoef( -zoe_depth, rendered_depth)),
                    (1 - pearson_corrcoef(1 / (zoe_depth + 200.), rendered_depth))
                ))
            elif mono_loss_type == "dust3r":
                gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
                render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
                mask = torch.logical_and(gt_mask, render_mask)
                if mask.sum() < 10:
                    depth_loss = 0.0
                else:
                    disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                    disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                    depth_loss = torch.abs((disp_render - disp_mono)).mean()
                depth_loss *= (self.opt.iterations - self.global_step) / self.opt.iterations # linear scheduler
            else:
                raise NotImplementedError

            loss = loss + args.mono_rate * depth_loss

        else:
            depth_loss = 0.

        return {
            'loss': loss,
            'l1_loss': Ll1,
            'ssim_loss': 1.0 - ssim(image, gt_image),
            'silhouette_loss': silhouette_loss,
            'depth_loss': depth_loss
        }


    def render_gs(self, batch: Dict[str, Any], renderbackground=None, need_loss=False) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = torch.rand((3), device="cuda") if self.opt.random_background else self.background_tensor

        images, depths, alphas = [], [], []
        self.viewspace_point_list, self.radii = [], None # register one empty list for each image rendered
        loss_all = {
            'loss': 0.,
            'l1_loss': 0.,
            'ssim_loss': 0.,
            'silhouette_loss': 0.,
            'depth_loss': 0.
        }
        for id in range(batch['index'].shape[0]):
            viewpoint_cam = Render_Camera(
                batch['R'][id],
                batch['T'][id],
                batch['fovx'][id],
                batch['fovy'][id],
                batch['image'][id],
                batch['mask'][id],
                batch['depth'][id],
                white_background = (self.bg_color == [1, 1, 1])
            )
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            self.viewspace_point_list.append(render_pkg["viewspace_points"])
            self.radii = render_pkg["radii"] if id == 0 else torch.max(render_pkg["radii"], self.radii)
            images.append(render_pkg["render"]) # CHW
            depths.append(render_pkg["rendered_depth"][0])
            alphas.append(render_pkg["rendered_alpha"][0])
            if need_loss:
                loss = self.cal_loss(self.opt, render_pkg["render"], render_pkg, viewpoint_cam, renderbackground)
                for k, v in loss.items():
                    loss_all[k] += v
        self.visibility_filter = self.radii > 0.0 # update visibility filter
        return {
            "images": torch.stack(images, 0),
            "depths": torch.stack(depths, 0),
            "alphas": torch.stack(alphas, 0),
            "loss": loss_all
        }


    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.controlnet = create_model(f'models/{self.cfg.model_name}.yaml').cpu()
        self.controlnet.load_state_dict(load_state_dict('models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        self.controlnet.load_state_dict(load_state_dict(f'models/{self.cfg.model_name}.pth', location='cuda'), strict=False)
        lora_config = {
            nn.Embedding: {
                "weight": partial(LoRAParametrization.from_embedding, rank=self.cfg.lora_rank)
            },
            nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=self.cfg.lora_rank)
            },
            nn.Conv2d: {
                "weight": partial(LoRAParametrization.from_conv2d, rank=self.cfg.lora_rank)
            }
        }
        if self.cfg.add_diffusion_lora:
            for name, module in self.controlnet.model.diffusion_model.named_modules():
                if name.endswith('transformer_blocks'):
                    add_lora(module, lora_config=lora_config)
        if self.cfg.add_control_lora:
            for name, module in self.controlnet.control_model.named_modules():
                if name.endswith('transformer_blocks'):
                    add_lora(module, lora_config=lora_config)
        if self.cfg.add_clip_lora:
            add_lora(self.controlnet.cond_stage_model, lora_config=lora_config)
        self.controlnet.load_state_dict(load_state_dict(f'{self.cfg.exp_name}/ckpts-lora/{self.cfg.lora_name}', location='cuda'), strict=False)
        self.controlnet = self.controlnet.cuda()
        self.ddim_sampler = DDIMSampler(self.controlnet)

    def get_dis_from_ts(self, T):
        return torch.sort(torch.sqrt(torch.sum((T - self.all_T) ** 2, dim=-1)))[0]


    def get_random_view_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        focal_length_x = fov2focal(batch['fovx'], batch['width'])
        focal_length_y = fov2focal(batch['fovy'], batch['height'])
        return {
            'index': batch['random_index'],
            'R': batch['random_R'],
            'T': batch['random_T'],
            'height': torch.tensor([self.novel_image_size]),
            'width': torch.tensor([self.novel_image_size]),
            'fovx': torch.tensor([focal2fov(focal_length_x, self.novel_image_size)]),
            'fovy': torch.tensor([focal2fov(focal_length_y, self.novel_image_size)]),
            'image': torch.zeros((batch['image'].shape[0], batch['image'].shape[1], self.novel_image_size, self.novel_image_size), device=batch['image'].device),
            'mask': torch.zeros((batch['mask'].shape[0], self.novel_image_size, self.novel_image_size), device=batch['mask'].device),
            'depth': torch.zeros((batch['depth'].shape[0], self.novel_image_size, self.novel_image_size), device=batch['depth'].device),
            'txt': batch['txt']
        }

    def training_step(self, batch, batch_idx):
        if self.max_cam_dis == 0.:
            Ts = batch['gt_Ts']
            self.all_T = torch.cat(Ts)
            for T in Ts:
                distances = self.get_dis_from_ts(T)
                self.max_cam_dis = max(self.max_cam_dis, distances[2].cpu().item())
            # TODO: magic number here
            self.max_cam_dis *= 1.2
        if self.global_step % self.cfg.refresh_interval == 0 and self.global_step <= self.cfg.around_gt_steps and self.global_step < self.ctrl_steps:
            if self.global_step > 0:
                for idx, controlnet_out in enumerate(self.controlnet_outs):
                    if controlnet_out is None:
                        controlnet_out = torch.zeros((1, 3, self.novel_image_size, self.novel_image_size), device='cuda')
                        self.controlnet_outs[idx] = controlnet_out
                controlnet_outs_image = make_grid(torch.cat(self.controlnet_outs, dim=0), nrow=5)
                save_image(controlnet_outs_image, self.get_save_path(f"controlnet_out/it{self.true_global_step}.png"))
            self.controlnet_outs = []
            if self.global_step < self.cfg.around_gt_steps:
                if len(self.gt_features_all) == 0:
                    for gt_image in batch['gt_images']:
                        with torch.no_grad():
                            gt_features = self.clip_model.encode_image(self.clip_preprocess(self.tensor_to_pil(gt_image[0])).unsqueeze(0).to(self.device))
                        self.gt_features_all.append(gt_features)
                for R, T in batch['random_poses']:
                    controlent_batch = batch.copy()
                    controlent_batch['random_R'] = R
                    controlent_batch['random_T'] = T
                    controlent_batch = self.get_random_view_batch(controlent_batch)
                    render_results = self.render_gs(controlent_batch, renderbackground=self.background_tensor, need_loss=False)
                    images = render_results['images']
                    image = images[0]
                    image_np = np.array(self.tensor_to_pil(image))
                    denoise_strength = random.random() * (self.cfg.max_strength - self.cfg.min_strength) + self.cfg.min_strength
                    controlnet_outs, sds_w = process(
                        self.controlnet,
                        self.ddim_sampler,
                        image_np,
                        prompt = batch['txt'][0],
                        a_prompt = 'best quality',
                        n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality',
                        num_samples = self.cfg.controlnet_num_samples,
                        image_resolution = min(image_np.shape[0], image_np.shape[1]),
                        ddim_steps = 50,
                        guess_mode = False,
                        strength = 1.0,
                        scale = 1.0,
                        eta = 1.0,
                        denoise_strength = denoise_strength
                    )
                    best_controlnet_out = controlnet_outs[0]
                    best_controlnet_out_score = 0.
                    for controlnet_out in controlnet_outs:
                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(self.clip_preprocess(Image.fromarray(controlnet_out)).unsqueeze(0).to(self.device))
                        score = sum([torch.cosine_similarity(image_features, gt_features, dim=-1).mean() for gt_features in self.gt_features_all])
                        if score > best_controlnet_out_score:
                            best_controlnet_out = controlnet_out
                            best_controlnet_out_score = score
                    self.controlnet_outs.append(self.pil_to_tensor(best_controlnet_out).to(torch.float32).unsqueeze(0).cuda())
                    self.sds_ws.append(sds_w)

        self.gaussian.update_learning_rate(self.true_global_step)

        render_results = self.render_gs(batch, need_loss=True)

        for k, v in render_results['loss'].items():
            self.log(f"retrain/{k}", v)

        gs_loss = render_results['loss']['loss']
        self.log("retrain/gs_loss", gs_loss)

        ctrl_loss = 0.
        if self.ctrl_loss_ratio > 0.0:
            batch = self.get_random_view_batch(batch)

            render_results = self.render_gs(batch, renderbackground=self.background_tensor, need_loss=False)
            images = render_results['images']

            controlnet_outs = []
            sds_ws = []
            if self.global_step < self.cfg.around_gt_steps:
                for idx, image in enumerate(images):
                    cached_controlnet_out = self.controlnet_outs[batch['index'][idx]]
                    if cached_controlnet_out is not None:
                        controlnet_out = cached_controlnet_out
                        sds_w = self.sds_ws[batch['index'][idx]]
                    else:
                        image_np = np.array(self.tensor_to_pil(image))
                        denoise_strength = self.cfg.max_strength
                        controlnet_out, sds_w = process(
                            self.controlnet,
                            self.ddim_sampler,
                            image_np,
                            prompt = batch['txt'][idx],
                            a_prompt = 'best quality',
                            n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality',
                            num_samples = 1,
                            image_resolution = min(image_np.shape[0], image_np.shape[1]),
                            ddim_steps = 50,
                            guess_mode = False,
                            strength = 1.0,
                            scale = 1.0,
                            eta = 1.0,
                            denoise_strength = denoise_strength
                        )
                        controlnet_out = self.pil_to_tensor(controlnet_out).to(torch.float32).unsqueeze(0).cuda()
                        self.controlnet_outs[batch['index'][idx]] = controlnet_out
                        self.sds_ws[batch['index'][idx]] = sds_w
                    controlnet_outs.append(controlnet_out)
                    sds_ws.append(sds_w)
            else:
                for idx, image in enumerate(images):
                    image_np = np.array(self.tensor_to_pil(image))
                    denoise_strength = random.random() * (self.cfg.max_strength - self.cfg.min_strength) + self.cfg.min_strength
                    controlnet_samples, sds_w = process(
                        self.controlnet,
                        self.ddim_sampler,
                        image_np,
                        prompt = batch['txt'][idx],
                        a_prompt = 'best quality',
                        n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality',
                        num_samples = self.cfg.controlnet_num_samples,
                        image_resolution = min(image_np.shape[0], image_np.shape[1]),
                        ddim_steps = 50,
                        guess_mode = False,
                        strength = 1.0,
                        scale = 1.0,
                        eta = 1.0,
                        denoise_strength = denoise_strength
                    )
                    best_controlnet_out = controlnet_samples[0]
                    best_controlnet_out_score = 0.
                    for controlnet_out in controlnet_samples:
                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(self.clip_preprocess(Image.fromarray(controlnet_out)).unsqueeze(0).to(self.device))
                        score = min([torch.cosine_similarity(image_features, gt_features, dim=-1).mean() for gt_features in self.gt_features_all])
                        if score > best_controlnet_out_score:
                            best_controlnet_out = controlnet_out
                            best_controlnet_out_score = score
                    controlnet_outs.append(self.pil_to_tensor(best_controlnet_out).to(torch.float32).unsqueeze(0).cuda())
                    sds_ws.append(sds_w)
                if self.global_step % self.cfg.refresh_interval == 0 and self.global_step != self.cfg.around_gt_steps:
                    controlnet_outs_image = make_grid(torch.cat(controlnet_outs, dim=0), nrow=min(5, len(controlnet_outs)))
                    save_image(controlnet_outs_image, self.get_save_path(f"controlnet_out/it{self.true_global_step}.png"))

            distances = self.get_dis_from_ts(batch['T'])
            distance_weight = min(1., 2 * distances[0].cpu().item() / self.max_cam_dis)
            self.log("train/distance_weight", distance_weight)

            # TODO: only works for batch size 1
            controlnet_outs = torch.cat(controlnet_outs, dim=0)
            sds_ws = sds_ws[0].cpu().item()
            self.log("train/sds_ws", sds_ws)

            loss_l1 = torch.nan_to_num(l1_loss(controlnet_outs, images))
            self.log("train/loss_l1", loss_l1)
            ctrl_loss += sds_ws * loss_l1 * self.C(self.cfg.loss['lambda_l1']) * distance_weight

            loss_l2 = torch.nan_to_num(l2_loss(controlnet_outs, images))
            self.log("train/loss_l2", loss_l2)
            ctrl_loss += sds_ws * loss_l2 * self.C(self.cfg.loss['lambda_l2']) * distance_weight

            loss_lpips = torch.nan_to_num(self.lpips_loss(controlnet_outs, images))
            self.log("train/loss_lpips", loss_lpips)
            ctrl_loss += sds_ws * loss_lpips * self.C(self.cfg.loss['lambda_lpips']) * distance_weight

            loss_tv = torch.nan_to_num(compute_tv_norm(render_results['depths'], losstype='l2').sqrt().mean())
            self.log("train/loss_tv", loss_tv)
            ctrl_loss += sds_ws * loss_tv * self.C(self.cfg.loss['lambda_tv']) * distance_weight

            self.log("train/loss", ctrl_loss)

        self.update_learning_rate()
        loss = gs_loss * (1.0 - self.ctrl_loss_ratio) + ctrl_loss * self.ctrl_loss_ratio

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}


    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.opt.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.gaussian.add_densification_stats_no_grad(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step >= self.opt.densify_from_iter and self.true_global_step % self.opt.densification_interval == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 300 else None # 3000
                    before_num_gauss = len(self.gaussian._xyz)
                    if before_num_gauss < self.opt.max_num_splats:
                        self.gaussian.densify(self.opt.densify_grad_threshold, self.cameras_extent)
                    if before_num_gauss > self.opt.min_num_splats:
                        self.gaussian.prune(self.opt.prune_opacity_threshold, self.cameras_extent, size_threshold)
                    torch.cuda.empty_cache()
                    after_num_gauss = len(self.gaussian._xyz)
                    threestudio.info(f'Run densification at step: {self.true_global_step}, before: {before_num_gauss}, after: {after_num_gauss}')
                    self.log('gaussian/num_gauss', torch.tensor(after_num_gauss, dtype=torch.float32))
                if self.true_global_step > 0 and self.true_global_step % self.opt.opacity_reset_interval == 0:
                    self.gaussian.reset_opacity()


    def on_train_epoch_end(self):
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)


    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        self.opt = OptimizationParams(self.parser)
        for k, v in self.cfg.gaussian_opt_params.items():
            self.opt.__setattr__(k, v)
        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(self.opt)
        ret = {
            "optimizer": self.gaussian.optimizer,
        }
        return ret


    def init_pointcloud(self, path):
        if path == 'random':
            pcb = self.pcb()
            self.gaussian.create_from_pcd(pcb, self.cameras_extent)
            return self.pcb()
        max_num = 0
        ply_path = ''
        for it in os.listdir(os.path.join(path, 'point_cloud')):
            if not os.path.isdir(os.path.join(path, 'point_cloud', it)):
                continue
            num = int(it.split('_')[-1])
            if num > max_num:
                max_num = num
                ply_path = os.path.join(path, 'point_cloud', it, 'point_cloud.ply')
        threestudio.info(f'init ply file from iter {max_num}')

        self.point_cloud = fetchPly(ply_path)
        self.num_pts = self.point_cloud.points.shape[0]
        self.gaussian.load_ply(ply_path)
        self.gaussian.update_spatial_lr_scale(self.cameras_extent)
        return self.point_cloud
