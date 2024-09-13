import os
import cv2
import json
import tempfile
import numpy as np
import trimesh
import torch
import einops
import argparse
from PIL import Image
from tqdm import trange
from torch.nn import functional as F
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.viz import pts3d_to_trimesh, cat_meshes
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from scene.colmap_loader import rotmat2qvec


def qvec2rvec(q):
    w, x, y, z = q
    theta = 2 * np.arccos(w)
    sin_theta_over_two = np.sin(theta / 2)
    if sin_theta_over_two > 0:
        vx = x / sin_theta_over_two
        vy = y / sin_theta_over_two
        vz = z / sin_theta_over_two
        return theta * np.array([vx, vy, vz])
    else:
        print('zeros')
        return np.array([0, 0, 0])

def points2homopoints(points):
    assert points.shape[-1] == 3
    bottom = torch.ones_like(points[...,0:1])
    return torch.cat([points, bottom], dim=-1)

def batch_projection(Ks, Ts, points):
    '''
    Ks: B, 3, 3
    Ts: B, 4, 4
    points: B, N, 3
    '''
    pre_fix = points.shape[:-1] # [100, 100]
    points = points.reshape(-1, 3) # [M, 3]

    Ts = torch.stack(Ts, dim=0) # [N, 4, 4]
    Ks = torch.stack(Ks, dim=0).to(Ts.device) # [N, 3, 3]
    camera_num = Ks.shape[0]
    homopts = points2homopoints(points) # [M, 4]
    # world to camera # [N, M, 4] @ [N, 4, 4] = [N, M, 4]
    homopts_cam = torch.bmm(homopts.unsqueeze(0).repeat_interleave(Ts.shape[0], dim=0), Ts.transpose(1,2)) 
    # camera to image space  # [N, M, 4] @ [N, 4, 3] = [N, M, 3]
    homopts_img = torch.bmm(homopts_cam[...,:3], Ks.transpose(1,2))
    # normalize
    homopts_img = homopts_img / (homopts_img[...,2:] + 1e-6)
    # reshape back
    homopts_img = homopts_img.reshape(camera_num, *pre_fix, 3)
    homopts_cam = homopts_cam.reshape(camera_num, *pre_fix, 4)
    return homopts_img[...,0:2], homopts_cam[...,2]

@torch.no_grad()
def get_visual_hull(N, scale, Ks, Ts, original_images, original_masks):
    pcs = []
    color = []
    all_pts = []

    [xs, ys, zs], [xe, ye, ze] = [-scale, -scale, -scale], [scale, scale, scale]

    images = torch.stack([torch.tensor(np.array(image, dtype=np.float64) / 255) for image in original_images]).cuda()
    images = einops.rearrange(images, 'b h w c -> b c h w')
    masks = torch.stack([torch.tensor(mask) for mask in original_masks]).cuda()
    masks = einops.rearrange(masks, 'b h w -> b 1 h w')
    image_height, image_width = images.shape[-2:]

    for h_id in trange(N):
        i, j = torch.meshgrid(torch.linspace(xs, xe, N).cuda(),
                              torch.linspace(ys, ye, N).cuda())
        i, j = i.t(), j.t()
        pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1)
        pts[...,2] = h_id / N * (ze - zs) + zs # 100, 100, 3

        all_pts.append(pts)

        # now we have the pts, we need to project them to the image plane
        # batched projection
        uv, z = batch_projection(Ks, Ts, pts) # [N, 100, 100, 2], [N, 100, 100]
        valid_z_mask = z > 0
        valid_x_y_mask = (uv[...,0] > 0) & (uv[...,0] < image_width) & (uv[...,1] > 0) & (uv[...,1] < image_height)
        valid_pt_mask = valid_z_mask & valid_x_y_mask

        # simple resize the uv to [-1, 1]
        uv[...,0] = uv[...,0] / image_width * 2 - 1
        uv[...,1] = uv[...,1] / image_height * 2 - 1

        # now we have the uv, we use grid_sample to sample the image to get the color
        result = F.grid_sample(images.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 3
        # sample mask
        result_mask = F.grid_sample(masks.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 1

        valid_pt_mask = result_mask.squeeze() > 0 & valid_pt_mask

        pcs.append(valid_pt_mask.float().sum(0) >= (images.shape[0] - 1)) # [100, 100]
        color.append(result.mean(0)) # [100, 100, 3]
    
    pcs = torch.stack(pcs, -1)
    color = torch.stack(color, -1)

    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    idx = torch.where(pcs > 0)

    color = torch.stack((r[idx] * 255, g[idx] * 255, b[idx] * 255), -1)

    idx = torch.stack([idx[1], idx[0], idx[2]], -1) # note the order is hwz -> xyz
    # turn the idx to the point position used in batch_projection
    idx = idx.float() / N
    idx[...,0] = idx[...,0] * (xe - xs) + xs
    idx[...,1] = idx[...,1] * (ye - ys) + ys
    idx[...,2] = idx[...,2] * (ze - zs) + zs

    return idx.cpu().numpy(), color.cpu().numpy() / 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', type=str, default='data/realcap/rabbit')
    parser.add_argument('--sparse_num', type=int, default=4)
    args = parser.parse_args()

    model_path = 'models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

    sparse_num = args.sparse_num
    rescale = 1.

    scene_path = args.source_path
    ids = np.loadtxt(os.path.join(scene_path, f'sparse_{sparse_num}.txt'), dtype=np.int32)
    images = sorted(os.listdir(os.path.join(scene_path, 'images')))
    images = [os.path.join(scene_path, 'images', images[id]) for id in ids]
    original_images = [Image.open(image) for image in images]
    masks = sorted(os.listdir(os.path.join(scene_path, 'masks')))
    masks = [os.path.join(scene_path, 'masks', masks[id]) for id in ids]
    original_masks = [np.array(Image.open(mask).resize(image.size))[:, :, 0] / 255.0 for mask, image in zip(masks, original_images)]

    loaded_images = load_images(images, size=512)
    pairs = make_pairs(loaded_images, scene_graph='complete', prefilter=None, symmetrize=True)
    cache_dir = tempfile.mkdtemp()
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(images, pairs, cache_dir, model, device=device)

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d, depths, confidences = scene.get_dense_pts3d()

    pts3d_mesh = to_numpy(pts3d)
    meshes = cat_meshes([pts3d_to_trimesh(imgs[i], pts3d_mesh[i].reshape(imgs[i].shape)) for i in range(len(imgs))])

    used_verts = set()
    for i, j, k in meshes['faces']:
        used_verts.add(i)
        used_verts.add(j)
        used_verts.add(k)
    used_verts = np.array(list(used_verts))

    vertices =  meshes['vertices'][used_verts]
    colors = meshes['vertice_colors'][used_verts]

    visibility = np.ones(vertices.shape[0], dtype=bool)
    for pose, focal, image, mask in zip(poses, focals, original_images, original_masks):
        width, height = image.size
        K = np.array([[focal.item() / 512 * max(height, width), 0, width / 2], [0, focal.item() / 512 * max(height, width), height / 2], [0, 0, 1]])

        c2w = pose.detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        R = rotmat2qvec(np.transpose(R))
        R = qvec2rvec(R)
        points_2d, _ = cv2.projectPoints(vertices, R, T, K, distCoeffs=None)
        h, w = mask.shape

        visibility[points_2d[:, 0, 1] < 0] = 0
        visibility[points_2d[:, 0, 1] >= h] = 0
        visibility[points_2d[:, 0, 0] < 0] = 0
        visibility[points_2d[:, 0, 0] >= w] = 0
        coords = points_2d.astype(np.int32)
        coords[coords < 0] = 0
        coords[:, 0, 1][coords[:, 0, 1] >= h] = h - 1
        coords[:, 0, 0][coords[:, 0, 0] >= w] = w - 1
        visibility[mask[coords[:, 0, 1], coords[:, 0, 0]] < 0.5] = 0

    vertices = vertices[visibility]
    colors = colors[visibility]

    center = np.mean(vertices, axis=0)
    vertices -= center
    max_bbox = np.abs(vertices).max()
    vertices = vertices / max_bbox * rescale

    poses[:, :3, 3] = (poses[:, :3, 3] - torch.tensor(center).to(device)) / max_bbox * rescale

    depths = np.array([depth.detach().cpu().numpy() for depth in depths])
    depths = depths / max_bbox * rescale
    np.save(os.path.join(scene_path, f'dust3r_depth_{sparse_num}.npy'), depths)
    confidences = np.array([confidence.detach().cpu().numpy() for confidence in confidences])
    np.save(os.path.join(scene_path, f'dust3r_confidence_{sparse_num}.npy'), confidences)

    cameras = []
    for i in range(sparse_num):
        cameras.append({
            'id': i,
            'img_name': os.path.basename(images[i]),
            'width': original_images[i].size[0],
            'height': original_images[i].size[1],
            'position': poses[i, :3, 3].tolist(),
            'rotation': poses[i, :3, :3].tolist(),
            'fy': focals[i].item() / 512 * max(original_images[i].size),
            'fx': focals[i].item() / 512 * max(original_images[i].size),
        })
    with open(os.path.join(scene_path, f'dust3r_{sparse_num}.json'), 'w') as f:
        json.dump(cameras, f, indent=4)

    cloud = trimesh.PointCloud(vertices, colors)
    cloud.export(os.path.join(scene_path, f'dust3r_{sparse_num}.ply'))
