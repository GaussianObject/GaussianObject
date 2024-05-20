from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    revision="1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9",
    filename="v1-5-pruned.ckpt",
    local_dir="models",
)

hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    revision="69fc48b9cbd98661f6d0288dc59b59a5ccb32a6b",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir="models",
)
