<div align="center">
<img src='assets/logo.png' style="height:100px"></img>
</div>

# GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting

## SIGGRAPH Asia 2024 (ACM Transactions on Graphics)

### [Project Page](https://gaussianobject.github.io/) | [Paper](https://arxiv.org/abs/2402.10259) | [Video](https://www.youtube.com/watch?v=s5arAXdgdZQ) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WIZgM--tJ3aq25t9g238JAuAoXrQYVMs?usp=sharing#scrollTo=TlrxF62GNePB)

[GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting](https://gaussianobject.github.io/)    
[Chen Yang](https://scholar.google.com/citations?hl=zh-CN&user=StdXTR8AAAAJ)<sup>1*</sup>, [Sikuang Li](https://gaussianobject.github.io/)<sup>1*</sup>, [Jiemin Fang](https://jaminfong.cn/)<sup>2†</sup>, [Ruofan Liang](https://nexuslrf.github.io/)<sup>3</sup>, [Lingxi Xie](http://lingxixie.com/Home.html)<sup>2</sup>, [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/)<sup>2</sup>, [Wei Shen](https://shenwei1231.github.io/)<sup>1✉</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>2</sup>    
<sup>1</sup>MoE Key Lab of Artificial Intelligence, AI Institute, SJTU &emsp; <sup>2</sup>Huawei Inc. &emsp; <sup>3</sup>University of Toronto    
<sup>*</sup>Equal contribution. &emsp; <sup>†</sup>Project lead. &emsp; <sup>✉</sup>Corresponding author.

##  🚩 News 
- 🤖 We provide a [step-by-step guideline](#-try-your-casually-captured-data) for COLMAP-free GaussianObject. Now you can use GaussianObject to reconstruct arbitary captured objects!
- 🔥 GaussianObject has been accepted by [ACM TOG (SIGGRAPH Asia 2024)!](https://asia.siggraph.org/2024/) See you in Tokyo!

---

https://github.com/user-attachments/assets/a388150a-2f90-4ced-ad90-d4aac48c39dc

We propose GaussianObject, a framework to represent and render the 3D object with Gaussian splatting, that achieves high rendering quality with only **4 input images** even under **COLMAP-free** conditions.

We first introduce techniques of visual hull and floater elimination which explicitly inject structure priors into the initial optimization process for helping build multi-view consistency, yielding a coarse 3D Gaussian representation. Then we construct a Gaussian repair model based on diffusion models to supplement the omitted object information, where Gaussians are further refined. We design a self-generating strategy to obtain image pairs for training the repair model. Our GaussianObject achives strong reconstruction results from only 4 views and significantly outperforms previous state-of-the-art methods.

![pipeline](assets/pipe.png)

- We initialize 3D Gaussians by constructing a visual hull with camera parameters and masked images, optimizing them with the $\mathcal{L}_{\text{gs}}$ and refining through floater elimination.
- We use a novel `leave-one-out' strategy and add 3D noise to Gaussians to generate corrupted Gaussian renderings. These renderings, paired with their corresponding reference images, facilitate the training of the Gaussian repair model employing $\mathcal{L}_{\text{tune}}$.
- Once trained, the Gaussian repair model is frozen and used to correct views that need to be rectified. These views are identified through distance-aware sampling. The repaired images and reference images are used to further optimize 3D Gaussians with $`\mathcal{L}_{\text{rep}}`$ and $`\mathcal{L}_{\text{gs}}`$.

## ⚡ Colab

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WIZgM--tJ3aq25t9g238JAuAoXrQYVMs?usp=sharing#scrollTo=TlrxF62GNePB)

[Sang Han](https://github.com/jjangsangy) provides a Colab script for GaussianObject in [#9](https://github.com/GaussianObject/GaussianObject/issues/9). Thanks for the contribution of the community! If you are experiencing issues with insufficient GPU VRAM, try this.

## 🚀 Setup

### CUDA

GaussianObject is tested with CUDA 11.8. If you are using a different version, you can choose to install [nvidia/cuda](https://anaconda.org/nvidia/cuda) in a local conda environment or modify the version of [PyTorch](https://pytorch.org/get-started/previous-versions/) in section [Python Environment](#python-environment).

### Cloning the Repository

The repository contains submodules. Please clone it with

```sh
git clone https://github.com/GaussianObject/GaussianObject.git --recursive
```

or update submodules in `GaussianObject` directory with

```sh
git submodule update --init --recursive
```

### Dataset

You can try GaussianObject with the Mip-NeRF360 dataset and OmniObject3D dataset. The data can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1DUOxFybdsSYJHI5p79O_QH87TIODiJ8h).
<details>
<summary>
The directory structure of the dataset is as follows:</summary>

```text
GaussianObject
├── data
│   ├── mip360
│   │   ├── bonsai
│   │   │   ├── images
│   │   │   ├── images_2
│   │   │   ├── images_4
│   │   │   ├── images_8
│   │   │   ├── masks
│   │   │   ├── sparse
│   │   │   ├── zoe_depth
│   │   │   ├── zoe_depth_colored
│   │   │   ├── sparse_4.txt
│   │   │   ├── sparse_6.txt
│   │   │   ├── sparse_9.txt
│   │   │   └── sparse_test.txt
│   │   ├── garden
│   │   └── kitchen
│   └── omni3d
└── ...
```
`images`, `images_2`, `images_4`, `images_8` and `sparse` are from the original dataset. `masks` is the object mask generated with [segment-anything](https://github.com/facebookresearch/segment-anything). `zoe_depth` and `zoe_depth_colored` are the depth maps and colored depth maps. `sparse_4.txt`, `sparse_6.txt` and `sparse_9.txt` are train set image ids and `sparse_test.txt` is the test set.

</details>



To test GaussianObject with your own dataset, you can manually prepare the dataset with the same directory structure. The depth maps and colored depth maps are generated with

```sh
python preprocess/pred_monodepth.py -s <YOUR_DATA_DIR>
```

### Python Environment

GaussianObject is tested with Python 3.11. All the required packages are listed in `requirements.txt`. You can install them with

```sh
# install pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# setup pip packages
pip install -r requirements.txt

# (Optional) setup croco for DUSt3R
cd submodules/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

### Pretrained ControlNet Model

Pretrained weights of Stable Diffusion v1.5 and ControlNet Tile need to be put in `models/` following the instruction of [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) with our given script:

```sh
cd models
python download_hf_models.py
cd ..
```

## 💪 Run the Code

Taking the scene `kitchen` from `mip360` dataset as an example, GaussianObject generate the visual hull of it, train a coarse 3DGS representation, analyze the statistical regularity of the coarse model with leave-one-out strategy, fine-tune the Gaussian Repair Model with LoRA and repair the 3DGS representation step by step.

### Visual Hull
<details>
<summary>
Train script:</summary>

```sh
python visual_hull.py \
    --sparse_id 4 \
    --data_dir data/mip360/kitchen \
    --reso 2 --not_vis
```
The visual hull is saved in `data/mip360/kitchen/visual_hull_4.ply`.
</details>


### Coarse 3DGS
<details>
<summary>
Train script:</summary>
```sh
python train_gs.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
```

You can render the coarse model it with

```sh
# render the test set
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --skip_all --skip_train

# render the path
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --render_path
```

The rendering results are saved in `output/gs_init/kitchen/test/ours_10000` and `output/gs_init/kitchen/render/ours_10000`.
</details>

### Leave One Out
<details>
<summary>
Train script:</summary>

```sh
python leave_one_out_stage1.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background

python leave_one_out_stage2.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
```

</details>

### LoRA Fine-Tuning
<details>
<summary>
Train script:</summary>

```sh
python train_lora.py --exp_name controlnet_finetune/kitchen \
    --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num 4 \
    --data_dir data/mip360/kitchen \
    --gs_dir output/gs_init/kitchen \
    --loo_dir output/gs_init/kitchen_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora
```
</details>

### Gaussian Repair
<details>
<summary>
Train script:</summary>

```sh
python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 0 \
    tag="kitchen" \
    system.init_dreamer="output/gs_init/kitchen" \
    system.exp_name="output/controlnet_finetune/kitchen" \
    system.refresh_size=8 \
    data.data_dir="data/mip360/kitchen" \
    data.resolution=4 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2
```

The final 3DGS representation is saved in `output/gaussian_object/kitchen/save/last.ply`. You can render it with

```sh
# render the test set
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --skip_all --skip_train \
    --load_ply output/gaussian_object/kitchen/save/last.ply

# render the path
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --render_path \
    --load_ply output/gaussian_object/kitchen/save/last.ply
```

The rendering results are saved in `output/gs_init/kitchen/test/ours_None` and `output/gs_init/kitchen/render/ours_None`.

</details>

## 📸 Try Your Casually Captured Data
GaussianObject can work without accurate camera poses (usually from COLMAP) and masks, which we term it as CF-GaussianObject. 

<details>
<summary>
Here is the guideline for CF-GaussianObject:</summary>

To use CF-GaussianObject (COLMAP-free GaussianObject), you need to download [SAM](https://github.com/facebookresearch/segment-anything) and [DUSt3R](https://github.com/naver/dust3r) or [MASt3R](https://github.com/naver/mast3r) checkpoints. 

```sh
cd models
sh download_preprocess_models.sh
cd ..
```

Assume you have a dataset with 4 images, it should be put in `./data` as the following structure

```text
GaussianObject
├── data
│   ├── <your dataset name>
│   │   ├── images
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── 0003.png
│   │   │   └── 0004.png
│   │   ├── sparse_4.txt
│   │   └── sparse_test.txt
│   └── ...
└── ...
```

where `sparse_4.txt` and `sparse_test.txt` contain the same sequence numbers of the input images, starting from 0. If all images are used for training, the files should be

```text
0
1
2
3
```

To downsampling the images, you can use

```sh
python preprocess/downsample.py -s data/realcap/rabbit
```


### Generate Masks

`segment_anything.ipynb` uses SAM to generate masks. Please refer to the file and [segment-anything](https://github.com/facebookresearch/segment-anything) for more details.

### Generate Coarse Poses

[DUSt3R](https://github.com/naver/dust3r) is used to estimate coarse poses for input images. You can get the poses with

```sh
python pred_poses.py -s data/realcap/rabbit --sparse_num 4
```

An alternative [MASt3R](https://github.com/naver/mast3r) script is provided in `pred_poses_mast3r.py`.

### Gaussian Repair
Once the data is prepared, the later steps are similar to standard GaussianObject.
You can refer to the [Run the Code](#-run-the-code) section for more details. Here is an example script.

<!-- <details>

<summary>Training script</summary> -->

```sh
python train_gs.py -s data/realcap/rabbit \
    -m output/gs_init/rabbit \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --random_background --use_dust3r

python render.py \
    -m output/gs_init/rabbit \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/rabbit/refined_cams.json \
    --white_background --render_path --use_dust3r

python leave_one_out_stage1.py -s data/realcap/rabbit \
    -m output/gs_init/rabbit_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/rabbit/refined_cams.json \
    --white_background --random_background --use_dust3r

python leave_one_out_stage2.py -s data/realcap/rabbit \
    -m output/gs_init/rabbit_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/rabbit/refined_cams.json \
    --white_background --random_background --use_dust3r

python train_lora.py --exp_name controlnet_finetune/rabbit \
    --prompt xxy5syt00 --sh_degree 2 --resolution 8 --sparse_num 4 \
    --data_dir data/realcap/rabbit \
    --gs_dir output/gs_init/rabbit \
    --loo_dir output/gs_init/rabbit_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora --use_dust3r

python train_repair.py \
    --config configs/gaussian-object-colmap-free.yaml \
    --train --gpu 0 \
    tag="rabbit" \
    system.init_dreamer="output/gs_init/rabbit" \
    system.exp_name="output/controlnet_finetune/rabbit" \
    system.refresh_size=8 \
    data.data_dir="data/realcap/rabbit" \
    data.resolution=8 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    data.json_path="output/gs_init/rabbit/refined_cams.json" \
    data.refresh_size=8 \
    system.sh_degree=2

python render.py \
    -m output/gs_init/rabbit \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --render_path --use_dust3r \
    --load_ply output/gaussian_object/rabbit/save/last.ply
```

</details>

## 🌏 Citation

If you find GaussianObject useful for your work please cite:

```text
@article{yang2024gaussianobject,
  title   = {GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting},
  author  = {Chen Yang and Sikuang Li and Jiemin Fang and Ruofan Liang and
             Lingxi Xie and Xiaopeng Zhang and Wei Shen and Qi Tian},
  journal = {ACM Transactions on Graphics},
  year    = {2024}
}
```

## 🤗 Acknowledgement

Some code of GaussianObject is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [threestudio](https://github.com/threestudio-project/threestudio) and [ControlNet](https://github.com/lllyasviel/ControlNet). Thanks for their great work!
