import os
import sys
import logging
import importlib

from huggingface_hub import hf_hub_download

def download_sam3d_pipeline(root_folder=".", hf_token=None, reload_libs=True):
    download_file_from_hf_hub('facebook/sam-3d-objects', 'pipeline.yaml', root_folder=root_folder, hf_token=hf_token, reload_libs=reload_libs)

def download_file_from_hf_hub(path_to_repo, file_path_dest, root_folder=".", hf_token=None, reload_libs=True):
    set_hf_content_environment(root_folder=root_folder, hf_token=hf_token, reload_libs=reload_libs)
    path = hf_hub_download(repo_id=path_to_repo, filename=file_path_dest)
    logging.info(f'downloaded {path_to_repo} : {file_path_dest} path {path}')
    return path

def set_hf_content_environment(root_folder=".", hf_token=None, reload_libs=True):
    # 1. Set env vars
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token

    os.environ["HF_HOME"] = f"{root_folder}/content/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{root_folder}/content/hf_cache/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{root_folder}/content/hf_cache/datasets"
    os.environ["HF_HUB_CACHE"] = f"{root_folder}/content/hf_cache/hub"

    # 2. Create dirs
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    logging.info(
        "HF env set | "
        f"HF_HOME={os.environ['HF_HOME']} | "
        f"TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}"
    )

    # 3. Reload HF-related libs if already imported
    if reload_libs:
        for module in [
            "transformers",
            "huggingface_hub",
            "datasets",
            "tokenizers",
        ]:
            if module in sys.modules:
                importlib.reload(sys.modules[module])
