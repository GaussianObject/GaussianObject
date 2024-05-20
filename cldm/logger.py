import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from lightning_utilities.core.rank_zero import rank_zero_only
from minlora import name_is_lora


class ImageLogger(Callback):
    def __init__(self, exp_dir='./ImageLogger', every_n_train_steps=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.exp_dir = exp_dir
        self.rescale = rescale
        self.every_n_train_steps = every_n_train_steps
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.every_n_train_steps]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.exp_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training # is train
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        skip_batch = self.every_n_train_steps < 1 or ((step + 1) % self.every_n_train_steps != 0)
        if not self.disabled and not skip_batch:
            self.log_img(pl_module, batch, batch_idx, split="train")


class LoraCheckpoint(Callback):
    def __init__(self, exp_dir='./LoraCheckpoint', every_n_train_steps=2000):
        super().__init__()
        self.exp_dir = exp_dir
        self.every_n_train_steps = every_n_train_steps
        os.makedirs(os.path.join(self.exp_dir, 'ckpts-lora'), exist_ok=True)

    def save_lora(self, pl_module, step):
        state_dict = pl_module.state_dict()
        lora_state_dict = {k: v for k, v in state_dict.items() if name_is_lora(k)}
        torch.save(lora_state_dict, os.path.join(self.exp_dir, 'ckpts-lora', f'lora-step={step}.ckpt'))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        skip_batch = self.every_n_train_steps < 1 or ((step + 1) % self.every_n_train_steps != 0)
        if not skip_batch:
            self.save_lora(pl_module, step)
