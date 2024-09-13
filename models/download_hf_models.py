from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="benjamin-paine/stable-diffusion-v1-5",
    revision="26a823710f75136819d791422b0b8686afbe784b",
    filename="v1-5-pruned.ckpt",
    local_dir=".",
)

hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    revision="69fc48b9cbd98661f6d0288dc59b59a5ccb32a6b",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir=".",
)
