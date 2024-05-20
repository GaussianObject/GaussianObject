### used for pred monocular depth from images
import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib
from tqdm import tqdm

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def init_model(model_name='ZoeD_N', device = "cuda"):
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384")

    # Error(s) in loading state_dict for ZoeDepth:
    # https://github.com/isl-org/ZoeDepth/issues/82
    cpu_model = torch.hub.load("isl-org/ZoeDepth", model_name, pretrained=False)
    pretrained_dict = torch.hub.load_state_dict_from_url(f'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/{model_name.replace("_", "_M12_")}.pt', map_location='cpu')
    cpu_model.load_state_dict(pretrained_dict['model'], strict=False)
    for b in cpu_model.core.core.pretrained.model.blocks:
        b.drop_path = torch.nn.Identity()

    model = cpu_model.to(device)
    return model


def listimage_to_batchedimage(listimage):
    batched_image = torch.stack([ToTensor()(image) for image in listimage])
    return batched_image


def pred_depth_batch(model, batched_image):
    depth_preds = model.infer(batched_image)
    return depth_preds


def save_raw_depth(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 1000  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)


def load_raw_depth(fpath="raw.png"):
    depth = Image.open(fpath)
    depth = np.array(depth)
    depth = (depth / 1000).astype(np.float32)
    return depth



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', type=str, default=f'data/mip360/kitchen')
    args = parser.parse_args()
    cur_dir = args.source_path

    zoe_model = init_model()
    zoe_depth_folder = os.path.join(cur_dir, 'zoe_depth')
    zoe_depth_colored_folder = os.path.join(cur_dir, 'zoe_depth_colored')
    os.makedirs(zoe_depth_folder, exist_ok=False)
    os.makedirs(zoe_depth_colored_folder, exist_ok=False)

    for file in tqdm(sorted(os.listdir(os.path.join(cur_dir, 'images')))):
        image = Image.open(os.path.join(cur_dir, 'images', file)).convert("RGB")
        depth_numpy = zoe_model.infer_pil(image)
        save_raw_depth(depth_numpy, os.path.join(zoe_depth_folder, file.replace('.JPG', '.png').replace('.jpg', '.png')))
        colored_depth = colorize(depth_numpy, cmap='magma_r')
        Image.fromarray(colored_depth).save(os.path.join(zoe_depth_colored_folder, file.replace('.JPG', '.png').replace('.jpg', '.png')))
