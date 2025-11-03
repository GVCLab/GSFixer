from huggingface_hub import snapshot_download
import os


def download_model():
    snapshot_download(
        repo_id="flow666/GSFixer",
        local_dir="../checkpoints/GSFixer",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="zai-org/CogVideoX-5b-I2V",
        local_dir="../checkpoints/CogVideoX-5b-I2V",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="facebook/VGGT-1B",
        local_dir="../checkpoints/VGGT-1B",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="facebook/dinov2-with-registers-large",
        local_dir="../checkpoints/dinov2-with-registers-large",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="Salesforce/blip2-opt-2.7b",
        local_dir="../checkpoints/blip2-opt-2.7b",
        local_dir_use_symlinks=False,
    )


download_model()
