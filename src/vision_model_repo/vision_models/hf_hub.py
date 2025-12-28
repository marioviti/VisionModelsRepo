import os
import sys
import logging
import importlib

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