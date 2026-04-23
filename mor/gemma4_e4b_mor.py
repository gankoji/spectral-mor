"""
PGD MOR analysis on Gemma 4 E4B real weights.
E4B: hidden=2560, intermediate=10240, 42 layers, 8 KV heads, head_dim=256

Key shapes:
  q_proj: (2048, 2560) - Q is 2048 wide (8 heads × 256)
  k_proj: (512, 2560)  - KV is 512 wide (8 KV heads × 64)
  v_proj: (512, 2560)
  o_proj: (2560, 2048) - output projection
  gate/up_proj: (10240, 2560)
  down_proj: (2560, 10240)
"""

import sys, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pgd_enrichment import pgd_decompose

from safetensors.torch import safe_open

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

# Key tensor paths (layer 0, mid-layer 20, last layer 41)
LAYER_INDICES = [0, 10, 20, 30, 41]
TENSOR_PATHS = {
    "q_proj":  "model.language_model.layers.{i}.self_attn.q_proj.weight",
    "k_proj":  "model.language_model.layers.{i}.self_attn.k_proj.weight",
    "v_proj":  "model.language_model.layers.{i}.self_attn.v_proj.weight",
    "o_proj":  "model.language_model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.language_model.layers.{i}.mlp.gate_proj.weight",
    "up_proj":   "model.language_model.layers.{i}.mlp.up_proj.weight",
    "down_proj":  "model.language_model.layers.{i}.mlp.down_proj.weight",
}
RANK_VALUES = [4, 8, 16, 32, 64]

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

def run_pgd_on_tensors(tensor_data, ranks, max_iters=20):
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

                # Reconstruct from CP modes: each mode is [row_factors, ..., col_factors]
                # For 2D matrix: each mode is [alpha*row_factor(1D), col_factor(1D)]
                reconstructed = np.zeros(shape, dtype=np.float32)
                for mode in modes:
                    # mode = [f_row, f_col] — both are 1D vectors
                    reconstructed += np.outer(mode[0], mode[1])

                rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)

                # Compression ratio
                n, m = shape
                full_params = n * m
                # Each mode contributes d factors of length shape[i]
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

def summarize_layer(results, layer_i):
    """Summarize results for a single layer."""
    summary = {"layer": layer_i, "tensors": {}}
    for name, rank_data in results[layer_i].items():
        summary["tensors"][name] = {}
        shape = rank_data[4]["shape"]
        summary["tensors"][name]["shape"] = shape
        summary["tensors"][name]["H_full"] = rank_data[4]["H_full"]

        # Best compression
        best = max(rank_data.items(), key=lambda x: x[1]["compression"])
        summary["tensors"][name]["best_R"] = best[0]
        summary["tensors"][name]["best_compression"] = best[1]["compression"]
        summary["tensors"][name]["best_rel_error"] = best[1]["rel_error"]
        summary["tensors"][name]["H_gap_pct"] = best[1]["H_gap_pct"]

        # Per-rank table
        summary["tensors"][name]["ranks"] = {
            R: {
                "compression": v["compression"],
                "rel_error": v["rel_error"],
                "walltime_s": v["walltime_s"],
            }
            for R, v in rank_data.items()
        }

    return summary

def main():
    print("=" * 70)
    print("PGD MOR on Gemma 4 E4B — Real Weights")
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

    # Summarize
    print("\n[4] Layer Summary:")
    all_results = {}
    for layer_i in LAYER_INDICES:
        summary = summarize_layer(pgd_results, layer_i)
        all_results[layer_i] = summary
        print(f"\n    Layer {layer_i}:")
        for name, ts in summary["tensors"].items():
            print(f"      {name:12s}: shape={ts['shape']}, H_full={ts['H_full']:.2f}")
            print(f"                 best: R={ts['best_R']}, {ts['best_compression']:.1f}× comp, "
                  f"rel_err={ts['best_rel_error']:.4f}, H_gap={ts['H_gap_pct']:.1f}%")

    # Compare with E2B
    e2b_results_path = OUTPUT_DIR.parent / "E2B" / "e2b_sample_results.json"
    print("\n[5] E4B vs E2B comparison:")
    if e2b_results_path.exists():
        e2b = json.loads(e2b_results_path.read_text())
        # E2B down_proj: 1536×12288
        # E4B down_proj: 2560×10240
        for layer_i in [0, 20, 41]:
            if layer_i in all_results and layer_i in e2b:
                print(f"\n    Layer {layer_i} — down_proj:")
                e4b_r = all_results[layer_i]["tensors"].get("down_proj", {})
                e2b_r = e2b.get(str(layer_i), {}).get("tensors", {}).get("down_proj", {})
                if e4b_r and e2b_r:
                    print(f"      E2B: {e2b_r['shape']}, best {e2b_r.get('best_compression','?')}× @ R={e2b_r.get('best_R','?')}")
                    print(f"      E4B: {e4b_r['shape']}, best {e4b_r['best_compression']}× @ R={e4b_r['best_R']}")
                    print(f"      Ratio: {e4b_r['best_compression'] / e2b_r.get('best_compression', 1):.1f}× better compression")

    # Save
    output_file = OUTPUT_DIR / "e4b_mor_results.json"
    with open(output_file, 'w') as f:
        json.dump({"meta": {"model": "gemma-4-E4B-it", "load_time_s": load_time,
                             "pgd_time_s": pgd_time, "layers": LAYER_INDICES,
                             "ranks": RANK_VALUES},
                    "full_results": pgd_results,
                    "summary": all_results}, f, indent=2)
    print(f"\n[6] Saved to {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()