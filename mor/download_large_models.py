"""
Download Gemma 4 larger models (26B-A4B, 31B) using huggingface_hub with resume.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def download_model(model_id: str, local_dir: Path = None):
    """Download model files to cache, resume if partial."""
    print(f"Downloading {model_id}...")
    try:
        # Use snapshot_download to get all files
        # This will resume partial downloads automatically
        cache_dir = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks="auto",
            resume_download=True,
            allow_patterns="*.safetensors",
            token=os.environ.get("HF_TOKEN")
        )
        print(f"Downloaded to: {cache_dir}")
        # List files
        safetensors = list(Path(cache_dir).glob("*.safetensors"))
        total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
        print(f"  {len(safetensors)} safetensor files, {total_gb:.2f} GB")
        return True
    except Exception as e:
        print(f"Error downloading {model_id}: {e}")
        return False

def main():
    models = [
        ("google/gemma-4-26B-A4B-it", "26B-A4B"),
        ("google/gemma-4-31B-it", "31B"),
    ]
    
    for model_id, name in models:
        print(f"\n=== {name} ===")
        # Check if already downloaded
        cache_root = Path.home() / ".cache/huggingface/hub"
        model_cache = cache_root / f"models--{model_id.replace('/', '--')}"
        if model_cache.exists():
            snapshots = [d for d in os.listdir(model_cache) if d not in ("blobs", "refs")]
            if snapshots:
                safetensors = list(Path(model_cache, snapshots[0]).glob("*.safetensors"))
                if safetensors:
                    total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
                    print(f"Already downloaded: {len(safetensors)} files, {total_gb:.2f} GB")
                    continue
        
        # Download
        success = download_model(model_id)
        if not success:
            print(f"Failed to download {name}")

if __name__ == "__main__":
    main()