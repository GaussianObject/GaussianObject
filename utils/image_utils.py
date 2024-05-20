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

import torch
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_meshlab_file(meshlab_file):
    ### in default, the meshlab file named pcd_transform.txt
    with open(meshlab_file) as f:
        for num, line in enumerate(f, 1):
            if num == 6:
                MLMatrix44 = line
                break
    return np.array(MLMatrix44.split()).reshape(4, 4).astype(np.float32)