import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pgd_enrichment import pgd_decompose
from safetensors.torch import safe_open

E4B_PATH = "/Users/jacbaile/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c/model.safetensors"

with safe_open(E4B_PATH, framework='pt', device='cpu') as f:
    key = "model.language_model.layers.20.self_attn.q_proj.weight"
    W = f.get_tensor(key).float().numpy().astype('float32')
    print(f"Shape: {W.shape}")
    print("Running PGD with R=4...")
    t0 = time.perf_counter()
    modes, residual = pgd_decompose(W, num_modes=4, max_fixed_point_iters=10, seed=42)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s, got {len(modes)} modes")
    reconstructed = np.zeros(W.shape, dtype=np.float32)
    for mode in modes:
        reconstructed += np.outer(mode[0], mode[1])
    rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)
    print(f"Relative error: {rel_error:.4f}")