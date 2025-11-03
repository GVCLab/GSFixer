from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="flow666/DL3DV-Res_Benchmark", 
    repo_type="dataset",
    local_dir="../data/DL3DV-Res",
    local_dir_use_symlinks=False,
    )

