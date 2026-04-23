"""
Download Gemma 4 E4B, 26B-A4B, and 31B weights in memory-mapped safetensor format.
Uses transformers snapshot_download to avoid loading into RAM.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from huggingface_hub import snapshot_download

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

models = [
    ("google/gemma-4-E4B-it", "E4B", None),
    ("google/gemma-4-26B-A4B-it", "26B-A4B", None),
    ("google/gemma-4-31B-it", "31B", None),
]

for model_id, name, _ in models:
    # Check if already downloaded
    cache_path = os.path.join(CACHE_DIR, f"models--{model_id.replace('/', '--')}")
    snapshots = [d for d in os.listdir(cache_path) if d != "blobs" and d != "refs"]
    if snapshots:
        safetensors = list(Path(cache_path, snapshots[0]).glob("*.safetensors"))
        if safetensors:
            total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
            print(f"[{name}] Already downloaded: {len(safetensors)} files, {total_gb:.2f} GB")
            continue

    print(f"\n[{name}] Downloading {model_id}...")
    t0 = time.perf_counter()
    try:
        local_path = snapshot_download(
            model_id,
            cache_dir=CACHE_DIR,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.bin", "*.msgpack"],
        )
        elapsed = time.perf_counter() - t0
        print(f"[{name}] Downloaded to {local_path} in {elapsed:.1f}s")

        # Report size
        safetensors = list(Path(local_path).glob("*.safetensors"))
        total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
        print(f"[{name}] {len(safetensors)} safetensor files, {total_gb:.2f} GB")
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
