import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pgd_enrichment import pgd_decompose
from safetensors.torch import safe_open

def compute_spectral_entropy(S: np.ndarray, eps: float = 1e-12) -> float:
    S_sq = S ** 2
    p = S_sq / (np.sum(S_sq) + eps)
    return -np.sum(p * np.log(p + eps))

def spectral_entropy_of_matrix(W: np.ndarray) -> float:
    s = np.linalg.svd(W, compute_uv=False)
    return compute_spectral_entropy(s)

def effective_rank(W: np.ndarray) -> float:
    s = np.linalg.svd(W, compute_uv=False)
    return np.exp(compute_spectral_entropy(s))

path = "/Users/jacbaile/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/7d4c97e54145f8ffd1a4dd1b4986a5015a517842/model-00001-of-00002.safetensors"
with safe_open(path, framework='pt', device='cpu') as f:
    # find q_proj layer 0
    for k in f.keys():
        if 'q_proj' in k and 'layers.0.' in k:
            W = f.get_tensor(k).float().numpy().astype('float32')
            print(f"Loaded {k}: shape {W.shape}")
            break

print(f"  Spectral entropy H_s: {spectral_entropy_of_matrix(W):.3f}")
print(f"  Effective rank: {effective_rank(W):.1f}")
print(f"  Full rank min(m,n): {min(W.shape)}")
print(f"  Effective rank ratio: {effective_rank(W)/min(W.shape):.3f}")

# Run PGD for R=4,8,16
for R in [4, 8, 16]:
    np.random.seed(42 + R)
    t0 = time.perf_counter()
    modes, residual = pgd_decompose(W, num_modes=R, max_fixed_point_iters=10, seed=42+R)
    elapsed = time.perf_counter() - t0
    
    reconstructed = np.zeros(W.shape, dtype=np.float32)
    for mode in modes:
        reconstructed += np.outer(mode[0], mode[1])
    rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)
    
    n, m = W.shape
    full_params = n * m
    mor_params = sum(sum(f.shape[0] for f in mode) for mode in modes)
    compression = full_params / mor_params
    
    H_recon = spectral_entropy_of_matrix(reconstructed)
    H_full = spectral_entropy_of_matrix(W)
    gap = (H_full - H_recon) / H_full * 100
    
    print(f"\nR={R}:")
    print(f"  Compression: {compression:.1f}×")
    print(f"  Rel error: {rel_error:.4f}")
    print(f"  H_recon: {H_recon:.3f} (gap: {gap:.1f}%)")
    print(f"  Walltime: {elapsed:.2f}s")