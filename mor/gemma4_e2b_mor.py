"""
Gemma 4 E2B PGD MOR Analysis (Memory-Mapped Version)
====================================================

Uses safetensors.torch.safe_open to selectively load only needed tensors
without loading the entire 10GB file into memory.

Real model dimensions from Gemma 4 E2B:
- Language model hidden: 1536 (constant)
- MLP: 6144 (layers 0-15) → 12288 (layers 16-34)
- Q projection: 2048×1536 (layers 0-15) → 4096×1536 (layers 16-34)
- K projection: 256×1536 (constant)
- V projection: 256×1536 (constant)
- O projection: 1536×2048 (layers 0-15) → 1536×4096 (layers 16-34)
- Gate/Up: 6144×1536 / 1536×6144 (early) → 12288×1536 / 1536×12288 (late)
- Down: 1536×6144 (early) → 1536×12288 (late)
"""

import numpy as np
import torch
import time
import json
import sys
import os
from pathlib import Path
from safetensors.torch import safe_open
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# PGD
sys.path.insert(0, str(Path(__file__).parent))
from pgd_enrichment import pgd_decompose


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAFETENSOR_PATH = (
    "/Users/jacbaile/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf/"
    "model.safetensors"
)

# Layers to analyze (3 representative layers)
LAYERS = [0, 17, 34]

# Projection types
PROJ_TYPES_ATTN = ["q_proj", "o_proj"]  # attention
PROJ_TYPES_MLP = ["down_proj"]  # MLP — focus on down_proj (output projection)

# PGD rank budgets (trimmed for speed)
RANK_BUDGETS = [8, 32, 128]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_spectral_entropy(S: np.ndarray, eps: float = 1e-12) -> float:
    S_sq = S ** 2
    p = S_sq / (np.sum(S_sq) + eps)
    return -np.sum(p * np.log(p + eps))


def effective_rank(S: np.ndarray) -> float:
    return np.exp(compute_spectral_entropy(S))


def load_tensors_mmap(keys: List[str]) -> Dict[str, np.ndarray]:
    """Load only specified tensors using memory-mapped safe_open."""
    result = {}
    with safe_open(SAFETENSOR_PATH, framework="pt", device="cpu") as f:
        for key in keys:
            try:
                tensor = f.get_tensor(key)
                result[key] = tensor.float().numpy().astype(np.float32)
            except Exception as e:
                print(f"  WARNING: '{key}' not found ({e})")
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

@dataclass
class PGDResult:
    layer: int
    proj_type: str
    shape: Tuple[int, int]
    rank: int
    walltime_s: float
    explained_var: float
    compression_ratio: float
    modes_extracted: int
    spectral_entropy_init: float
    eff_rank_init: float
    eff_rank_ratio: float
    initial_norm: float


def analyze_projection(
    weight: np.ndarray,
    layer: int,
    proj_type: str,
    rank_budgets: List[int],
) -> List[PGDResult]:
    """Run PGD at multiple rank budgets for a single projection tensor."""

    m, n = weight.shape
    initial_norm = float(np.linalg.norm(weight))

    # Compute initial spectral properties
    initial_svd = np.linalg.svd(weight, compute_uv=False)
    H_init = compute_spectral_entropy(initial_svd)
    er_init = effective_rank(initial_svd)

    results = []

    for rank in rank_budgets:
        if rank > min(m, n):
            break

        t0 = time.perf_counter()
        modes, residual_norms = pgd_decompose(
            weight,
            num_modes=rank,
            max_fixed_point_iters=10,
            seed=42,
        )
        elapsed = time.perf_counter() - t0

        final_residual = residual_norms[-1]
        explained_var = 1.0 - (final_residual ** 2) / (initial_norm ** 2 + 1e-12)

        # Compression ratio
        full_params = m * n
        compressed_params = rank * (m + n)
        comp_ratio = full_params / compressed_params

        results.append(PGDResult(
            layer=layer,
            proj_type=proj_type,
            shape=(m, n),
            rank=rank,
            walltime_s=elapsed,
            explained_var=float(explained_var),
            compression_ratio=float(comp_ratio),
            modes_extracted=len(modes),
            spectral_entropy_init=float(H_init),
            eff_rank_init=float(er_init),
            eff_rank_ratio=float(er_init / min(m, n)),
            initial_norm=float(initial_norm),
        ))

    return results


def main():
    output_dir = Path(__file__).parent / "gemma4_weights"
    output_dir.mkdir(exist_ok=True)

    # Build key names
    keys_to_load = []
    for layer in LAYERS:
        for proj in PROJ_TYPES_ATTN:
            keys_to_load.append(f"model.language_model.layers.{layer}.self_attn.{proj}.weight")
        for proj in PROJ_TYPES_MLP:
            keys_to_load.append(f"model.language_model.layers.{layer}.mlp.{proj}.weight")

    print(f"Memory-mapping {len(keys_to_load)} tensors from safetensor...")
    t0 = time.perf_counter()
    tensors = load_tensors_mmap(keys_to_load)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    if tensors:
        print(f"Memory: {sum(v.nbytes for v in tensors.values()) / 1e9:.3f} GB")

    if not tensors:
        print("ERROR: No tensors loaded. Aborting.")
        return []

    # Print shapes
    print("\n--- Weight Shapes Loaded ---")
    for name, arr in sorted(tensors.items()):
        layer = name.split(".layers.")[1].split(".")[0] if ".layers." in name else "?"
        proj = name.split(".")[-2]
        print(f"  L{layer} {proj}: {arr.shape} ({arr.nbytes/1e6:.1f} MB FP32)")

    # Run PGD analysis
    print(f"\n{'='*80}")
    print("PGD MOR ANALYSIS: Gemma 4 E2B")
    print(f"{'='*80}")

    all_results = []

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")

        # Attention projections
        for proj in PROJ_TYPES_ATTN:
            key = f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
            if key not in tensors:
                continue
            weight = tensors[key]
            m, n = weight.shape
            print(f"\n  {proj} ({m}×{n})  ", end="", flush=True)
            proj_results = analyze_projection(weight, layer, proj, RANK_BUDGETS)
            all_results.extend(proj_results)
            best = max(proj_results, key=lambda r: r.compression_ratio)
            print(f"→ best: {best.compression_ratio:.1f}× @ R={best.rank} "
                  f"({best.explained_var:.1%} expl. var, {best.walltime_s:.1f}s)")

        # MLP projections
        for proj in PROJ_TYPES_MLP:
            key = f"model.language_model.layers.{layer}.mlp.{proj}.weight"
            if key not in tensors:
                continue
            weight = tensors[key]
            m, n = weight.shape
            print(f"\n  {proj} ({m}×{n})  ", end="", flush=True)
            proj_results = analyze_projection(weight, layer, proj, RANK_BUDGETS)
            all_results.extend(proj_results)
            best = max(proj_results, key=lambda r: r.compression_ratio)
            print(f"→ best: {best.compression_ratio:.1f}× @ R={best.rank} "
                  f"({best.explained_var:.1%} expl. var, {best.walltime_s:.1f}s)")

    # Summary tables
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLES")
    print(f"{'='*80}")

    # Table 1: compression ratio
    rb_str = " | ".join(f"R={rb}" for rb in RANK_BUDGETS)
    print(f"\n--- Compression ratio (×) ---")
    print(f"{'Proj':<12} | {'Shape':>14} | {rb_str}")
    print("-" * (30 + len(RANK_BUDGETS) * 9))

    seen = set()
    for r in all_results:
        key = (r.proj_type, r.shape)
        if key in seen:
            continue
        seen.add(key)
        row = {rb.rank: rb.compression_ratio for rb in all_results
               if rb.proj_type == r.proj_type and rb.shape == r.shape}
        shape_str = f"{r.shape[0]}×{r.shape[1]}"
        vals = " | ".join(f"{row.get(rb, 0):>5.1f}×" if rb in row else "  --- "
                          for rb in RANK_BUDGETS)
        print(f"{r.proj_type:<12} | {shape_str:>14} | {vals}")

    # Table 2: explained variance
    print(f"\n--- Explained variance ---")
    print(f"{'Proj':<12} | {'Shape':>14} | {rb_str}")
    print("-" * (30 + len(RANK_BUDGETS) * 9))

    seen = set()
    for r in all_results:
        key = (r.proj_type, r.shape)
        if key in seen:
            continue
        seen.add(key)
        row = {rb.rank: rb.explained_var for rb in all_results
               if rb.proj_type == r.proj_type and rb.shape == r.shape}
        shape_str = f"{r.shape[0]}×{r.shape[1]}"
        vals = " | ".join(f"{row.get(rb, 0):>6.0%}" if rb in row else "   --- "
                          for rb in RANK_BUDGETS)
        print(f"{r.proj_type:<12} | {shape_str:>14} | {vals}")

    # Table 3: walltime
    print(f"\n--- Walltime (seconds) ---")
    print(f"{'Proj':<12} | {'Shape':>14} | {rb_str}")
    print("-" * (30 + len(RANK_BUDGETS) * 9))

    seen = set()
    for r in all_results:
        key = (r.proj_type, r.shape)
        if key in seen:
            continue
        seen.add(key)
        row = {rb.rank: rb.walltime_s for rb in all_results
               if rb.proj_type == r.proj_type and rb.shape == r.shape}
        shape_str = f"{r.shape[0]}×{r.shape[1]}"
        vals = " | ".join(f"{row.get(rb, 0):>6.1f}s" if rb in row else "   --- "
                          for rb in RANK_BUDGETS)
        print(f"{r.proj_type:<12} | {shape_str:>14} | {vals}")

    # Spectral entropy table
    print(f"\n--- Spectral entropy (untouched Gemma 4 E2B weights) ---")
    print(f"{'Proj':<12} | {'Shape':>14} | {'Init H_s':>9} | {'Eff Rank':>9} | {'Ratio':>7}")
    print("-" * 60)

    seen = set()
    for r in all_results:
        key = (r.proj_type, r.shape)
        if key in seen:
            continue
        seen.add(key)
        shape_str = f"{r.shape[0]}×{r.shape[1]}"
        print(f"{r.proj_type:<12} | {shape_str:>14} | {r.spectral_entropy_init:>9.4f} | "
              f"{r.eff_rank_init:>9.1f} | {r.eff_rank_ratio:>7.1%}")

    # Save results
    results_serializable = []
    for r in all_results:
        d = asdict(r)
        d["shape"] = list(d["shape"])
        results_serializable.append(d)

    results_path = output_dir / "gemma4_e2b_pgd_results.json"
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    main()