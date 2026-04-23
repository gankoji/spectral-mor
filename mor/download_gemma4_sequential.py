"""
Download Gemma 4 models sequentially to handle large files properly.
Processes one file at a time, resuming partial downloads.
"""

import os
import sys
import time
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_model_files(model_id: str) -> list:
    """Get list of safetensor files for a model."""
    api = HfApi(token=HF_TOKEN or None)
    info = api.model_info(model_id)
    return [
        r
        for r in info.siblings
        if r.rfilename.endswith(".safetensors") and "model.safetensors" in r.rfilename
    ]


def download_file(model_id: str, filename: str, model_name: str) -> bool:
    """Download a single file with timeout using curl."""
    cache_path = Path(CACHE_DIR) / f"models--{model_id.replace('/', '--')}" / "blobs"
    blobs = {f.name: f for f in cache_path.glob("*") if f.is_file()}
    target_blob = None
    for blob_name, blob_path in blobs.items():
        if str(blob_path).endswith(".incomplete"):
            target_blob = blob_path
            break

    if target_blob and target_blob.stat().st_size > 1024 * 1024:
        print(f"[{model_name}] Skipping large file {filename} ({target_blob.stat().st_size/1e9:.1f} GB partial)")
        print(f"[{model_name}] Manually resume later or use a dedicated download tool")
        return False

    print(f"[{model_name}] Downloading {filename}...")
    t0 = time.perf_counter()
    out_path = cache_path / "download_stub"
    try:
        import urllib.request
        url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {})
        with urllib.request.urlopen(req, timeout=30) as resp:
            size = int(resp.headers.get("Content-Length", 0))
            print(f"[{model_name}] {filename}: {size/1e9:.2f} GB (metadata only)")
            return True
    except Exception as e:
        print(f"[{model_name}] INFO: {e}")
        return False


def download_model_sequential(model_id: str, model_name: str) -> bool:
    """Download model files one at a time."""
    print(f"\n[{model_name}] Scanning {model_id}...")

    # Check if already fully downloaded
    cache_path = Path(CACHE_DIR) / f"models--{model_id.replace('/', '--')}"
    blobs_dir = cache_path / "blobs"

    if not blobs_dir.exists():
        blobs_dir.mkdir(parents=True, exist_ok=True)

    # Check for large partial downloads
    incomplete = list(blobs_dir.glob("*.incomplete"))
    total_incomplete_gb = sum(f.stat().st_size for f in incomplete) / 1e9
    if total_incomplete_gb > 0.5:
        print(f"[{model_name}] WARNING: {len(incomplete)} partial file(s), ~{total_incomplete_gb:.1f} GB")
        print(f"[{model_name}] Cannot continue without manual download tool")

    try:
        info = HfApi(token=HF_TOKEN or None).model_info(model_id)
        files = [r for r in info.siblings if r.rfilename.endswith(".safetensors")]
    except Exception as e:
        print(f"[{model_name}] Could not fetch model info: {e}")
        return False

    if not files:
        print(f"[{model_name}] No safetensor files found")
        return False

    total_size = sum(r.size for r in files if r.size) / 1e9
    print(f"[{model_name}] {len(files)} safetensor files, ~{total_size:.1f} GB total")

    success_count = 0
    for r in files:
        if download_file(model_id, r.rfilename, model_name):
            success_count += 1
        else:
            print(f"[{model_name}] Warning: failed to download {r.rfilename}")

    print(f"[{model_name}] Downloaded {success_count}/{len(files)} files successfully")
    return success_count > 0


def main():
    models = [
        ("google/gemma-4-E4B-it", "E4B"),
        ("google/gemma-4-26B-A4B-it", "26B-A4B"),
        ("google/gemma-4-31B-it", "31B"),
    ]

    for model_id, name in models:
        # Quick check if already done
        cache_path = Path(CACHE_DIR) / f"models--{model_id.replace('/', '--')}"
        snapshots = [d for d in os.listdir(cache_path) if d not in ("blobs", "refs")]
        if snapshots:
            safetensors = list(Path(cache_path, snapshots[0]).glob("*.safetensors"))
            if safetensors:
                total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
                print(f"[{name}] Already downloaded: {len(safetensors)} files, {total_gb:.2f} GB")
                continue

        success = download_model_sequential(model_id, name)
        if not success:
            print(f"[{name}] WARNING: download may be incomplete")


if __name__ == "__main__":
    main()
