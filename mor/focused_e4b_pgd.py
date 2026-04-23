"""
Focused PGD analysis on Gemma 4 E4B real weights.
Layers: 0, 20, 41
Tensors: q_proj, down_proj
Ranks: 4, 8, 16, 32
"""

import sys, time, json
print("[DEBUG] Starting script", file=sys.stderr)
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
print("[DEBUG] Before pgd import", file=sys.stderr)
from pgd_enrichment import pgd_decompose
print("[DEBUG] After pgd import", file=sys.stderr)

print("[DEBUG] Before safetensors import", file=sys.stderr)
from safetensors.torch import safe_open
print("[DEBUG] After safetensors import", file=sys.stderr)

# --- Spectral helpers ---
def compute_spectral_entropy(S: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of squared singular values (variance spectrum)."""
    S_sq = S ** 2
    p = S_sq / (np.sum(S_sq) + eps)
    return -np.sum(p * np.log(p + eps))

def effective_rank(W: np.ndarray) -> float:
    """Effective rank = exp(entropy) of singular spectrum."""
    s = np.linalg.svd(W, compute_uv=False)
    return np.exp(compute_spectral_entropy(s))

def spectral_entropy_of_matrix(W: np.ndarray) -> float:
    """Compute spectral entropy of matrix W."""
    s = np.linalg.svd(W, compute_uv=False)
    return compute_spectral_entropy(s)

# Paths
E4B_PATH = "/Users/jacbaile/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c/model.safetensors"
OUTPUT_DIR = Path(__file__).parent / "gemma4_weights" / "E4B"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Focused subset
LAYER_INDICES = [0, 20, 41]
TENSOR_PATHS = {
    "q_proj":  "model.language_model.layers.{i}.self_attn.q_proj.weight",
    "down_proj": "model.language_model.layers.{i}.mlp.down_proj.weight",
}
RANK_VALUES = [4, 8, 16, 32]
MAX_ITERS = 10  # reduced from 20 for speed

def load_tensors(sf_path, layer_indices, tensor_paths):
    """Load specified tensors from safetensor file."""
    results = {}
    with safe_open(sf_path, framework='pt', device='cpu') as f:
        for i in layer_indices:
            results[i] = {}
            for name, tmpl in tensor_paths.items():
                key = tmpl.format(i=i)
                t = f.get_tensor(key).float().numpy().astype('float32')
                results[i][name] = t
    return results

def run_pgd_on_tensors(tensor_data, ranks, max_iters=MAX_ITERS):
    """Run PGD decomposition on all loaded tensors."""
    results = {}
    for layer_i, layer_tensors in tensor_data.items():
        results[layer_i] = {}
        for name, W in layer_tensors.items():
            results[layer_i][name] = {}
            shape = W.shape
            # Compute full spectral entropy
            H_full = spectral_entropy_of_matrix(W)
            r_full = effective_rank(W)

            for R in ranks:
                np.random.seed(42 + R)
                t_start = time.perf_counter()
                modes, residual = pgd_decompose(
                    W, num_modes=R, max_fixed_point_iters=max_iters, seed=42+R
                )
                elapsed = time.perf_counter() - t_start

                # Reconstruct from CP modes
                reconstructed = np.zeros(shape, dtype=np.float32)
                for mode in modes:
                    # mode = [f_row, f_col] for 2D matrix
                    reconstructed += np.outer(mode[0], mode[1])

                rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)

                # Compression ratio
                n, m = shape
                full_params = n * m
                mor_params = sum(sum(f.shape[0] for f in mode) for mode in modes)
                compression = full_params / mor_params

                # Spectral entropy of reconstructed
                H_recon = spectral_entropy_of_matrix(reconstructed)

                results[layer_i][name][R] = {
                    "rank": R,
                    "shape": list(shape),
                    "compression": round(compression, 1),
                    "rel_error": round(float(rel_error), 5),
                    "H_full": round(float(H_full), 4),
                    "H_recon": round(float(H_recon), 4),
                    "H_gap_pct": round(float((H_full - H_recon) / H_full * 100), 2),
                    "eff_rank_full": round(float(r_full), 1),
                    "walltime_s": round(elapsed, 3),
                }
    return results

def main():
    print("=" * 70)
    print("Focused PGD on Gemma 4 E4B — Layers", LAYER_INDICES)
    print("=" * 70)

    # Load tensors
    print("\n[1] Loading E4B tensors...")
    t0 = time.perf_counter()
    tensor_data = load_tensors(E4B_PATH, LAYER_INDICES, TENSOR_PATHS)
    load_time = time.perf_counter() - t0
    print(f"    Loaded in {load_time:.1f}s")

    # Print shapes
    print("\n    Tensor shapes:")
    for i in LAYER_INDICES:
        print(f"    Layer {i}:")
        for name, W in tensor_data[i].items():
            n, m = W.shape
            print(f"      {name:12s}: {n:5d} × {m:6d}  ({n*m/1e6:.1f}M params)")

    # Run PGD
    print(f"\n[2] Running PGD at R = {RANK_VALUES}...")
    t0 = time.perf_counter()
    pgd_results = run_pgd_on_tensors(tensor_data, RANK_VALUES)
    pgd_time = time.perf_counter() - t0
    print(f"    Done in {pgd_time:.1f}s")

    # Print results
    print("\n[3] Results:")
    for layer_i in LAYER_INDICES:
        print(f"\n    Layer {layer_i}:")
        for name in TENSOR_PATHS:
            rd = pgd_results[layer_i][name]
            shape = rd[4]["shape"]
            H_full = rd[4]["H_full"]
            print(f"      {name:12s} ({shape[0]}×{shape[1]}, H={H_full:.2f})")
            print(f"      {'R':>4}  {'Comp':>7}  {'RelErr':>8}  {'H_recon':>8}  {'H_gap%':>7}  {'Time(s)':>7}")
            for R in RANK_VALUES:
                r = rd[R]
                print(f"      {R:>4}  {r['compression']:>7.1f}×  {r['rel_error']:>8.4f}  "
                      f"{r['H_recon']:>8.2f}  {r['H_gap_pct']:>6.1f}%  {r['walltime_s']:>7.2f}")

    # Save
    output_file = OUTPUT_DIR / "focused_pgd_results.json"
    with open(output_file, 'w') as f:
        json.dump({"meta": {"model": "gemma-4-E4B-it", "load_time_s": load_time,
                             "pgd_time_s": pgd_time, "layers": LAYER_INDICES,
                             "ranks": RANK_VALUES, "max_iters": MAX_ITERS},
                    "full_results": pgd_results}, f, indent=2)
    print(f"\n[4] Saved to {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()