"""
PGD Scaling Experiment
======================

Explores PGD enrichment at real model dimensions, testing whether the
compression efficiency discovered at 128-d persists at scales closer to
actual LLM weight tensors.

Key questions:
1. Does PGD maintain compression efficiency at 12k x 12k (GPT-2 class)?
2. How does runtime scale with dimension?
3. Does the entropy reduction behavior hold at larger scales?
4. What dimensions are needed for meaningful MOR exploration?
"""

import numpy as np
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mor"))
from pgd_enrichment import pgd_decompose, _reconstruct_rank1


# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

@dataclass
class ModelArch:
    name: str
    d_model: int       # hidden dim
    d_intermediate: int  # MLP intermediate dim
    n_layers: int


ARCHS = {
    "toy_128": ModelArch("toy_128", d_model=128, d_intermediate=512, n_layers=4),

    # GPT-2 class (124M)
    "gpt2_small": ModelArch("gpt2_small", d_model=768, d_intermediate=3072, n_layers=12),

    # GPT-2 medium class (345M)
    "gpt2_med": ModelArch("gpt2_med", d_model=1024, d_intermediate=4096, n_layers=24),

    # Full GPT-2 hidden dim (the actual Q/K/V projection dimension in GPT-2)
    # c_attn is (768, 2304) — effectively 3 × 768 slices along output
    # We test the full square projection: (d_model, d_model)
    "gpt2_full": ModelArch("gpt2_full", d_model=768, d_intermediate=3072, n_layers=12),

    # Gemma 3 270M
    "gemma3_270m": ModelArch("gemma3_270m", d_model=1024, d_intermediate=4096, n_layers=16),

    # Stress-test: what happens at near-12k square?
    "stress_4k": ModelArch("stress_4k", d_model=4096, d_intermediate=16384, n_layers=8),
    "stress_8k": ModelArch("stress_8k", d_model=8192, d_intermediate=32768, n_layers=4),
    "stress_12k": ModelArch("stress_12k", d_model=12288, d_intermediate=49152, n_layers=2),
}


@dataclass
class ScalingResult:
    arch_name: str
    tensor_shape: Tuple[int, int]
    dtype: str
    rank_budget: int
    walltime_s: float
    residual_ratio: float
    explained_variance: float
    modes_extracted: int
    modes_converged: bool
    initial_norm: float
    final_residual_norm: float
    compression_ratio: float  # how many params saved vs storing full matrix


# ---------------------------------------------------------------------------
# Singular-value utilities
# ---------------------------------------------------------------------------

def compute_spectral_entropy(S: np.ndarray, eps: float = 1e-12) -> float:
    S_sq = S ** 2
    p = S_sq / (np.sum(S_sq) + eps)
    return -np.sum(p * np.log(p + eps))


def effective_rank(S: np.ndarray) -> float:
    return np.exp(compute_spectral_entropy(S))


# ---------------------------------------------------------------------------
# Single-tensor PGD run
# ---------------------------------------------------------------------------

def run_pgd_on_tensor(
    tensor: np.ndarray,
    rank_budget: int,
    seed: int = 42,
    max_iters: int = 20,
    tol: float = 1e-6,
) -> ScalingResult:
    """
    Run PGD decomposition on a single tensor and record scaling metrics.
    """
    initial_norm = float(np.linalg.norm(tensor))
    shape = tensor.shape

    t0 = time.perf_counter()
    modes, residual_norms = pgd_decompose(
        tensor,
        num_modes=rank_budget,
        max_fixed_point_iters=max_iters,
        seed=seed,
        tol=tol,
    )
    walltime = time.perf_counter() - t0

    final_residual = residual_norms[-1]
    modes_extracted = len(modes)
    residual_ratio = final_residual / (initial_norm + 1e-12)
    explained_variance = 1.0 - (final_residual ** 2) / (initial_norm ** 2 + 1e-12)

    # Compression: each rank-1 mode costs sum(dim_i) params vs prod(dim_i) for full
    d = len(shape)
    full_params = int(np.prod(shape))
    mode_costs = [int(sum(len(m[i]) for i in range(d))) for m in modes]
    compressed_params = sum(mode_costs)
    compression_ratio = full_params / (compressed_params + 1e-12)

    # Check convergence: did we hit max_iters or early-exit?
    modes_converged = modes_extracted < rank_budget

    return ScalingResult(
        arch_name="",
        tensor_shape=shape,
        dtype=str(tensor.dtype),
        rank_budget=rank_budget,
        walltime_s=walltime,
        residual_ratio=residual_ratio,
        explained_variance=explained_variance,
        modes_extracted=modes_extracted,
        modes_converged=modes_converged,
        initial_norm=initial_norm,
        final_residual_norm=final_residual,
        compression_ratio=compression_ratio,
    )


def fill_arch_name(result: ScalingResult, arch_name: str) -> ScalingResult:
    result.arch_name = arch_name
    return result


# ---------------------------------------------------------------------------
# Weight generators
# ---------------------------------------------------------------------------

def synthetic_trained_weight(shape: Tuple[int, int], seed: int = 42, exponent: float = 0.5) -> np.ndarray:
    """
    Generate a weight matrix with power-law spectral structure
    (simulates trained LLM weight rather than random init).
    """
    np.random.seed(seed)
    m, n = shape
    r = min(m, n)
    # Random base for SVD
    base = np.random.randn(*shape).astype(np.float32)
    U_actual, s_mean, Vh_actual = np.linalg.svd(base, full_matrices=False)
    # Build power-law singular value spectrum (length = r)
    k = np.arange(1, r + 1)
    spectrum = 1.0 / (k ** exponent)
    spectrum = spectrum / spectrum.max()
    # Scale to the mean scale of the random SVD
    scale = float(np.mean(s_mean)) / max(float(np.max(spectrum)), 1e-12)
    powerlaw_diag = (spectrum * scale).astype(np.float32)  # shape (r,)
    # W = U @ diag(s_powerlaw) @ Vh  — diag is (r,) applied via np.diag inside the matmul chain
    return (U_actual.astype(np.float32) @ np.diag(powerlaw_diag) @ Vh_actual.astype(np.float32)).astype(np.float32)


def random_gaussian_weight(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-architecture sweep
# ---------------------------------------------------------------------------

def sweep_architecture(
    arch_name: str,
    arch: ModelArch,
    tensor_types: List[str] = None,
    rank_budgets: List[int] = None,
) -> List[ScalingResult]:
    """
    Sweep a single architecture across multiple rank budgets and tensor types.
    """
    if tensor_types is None:
        tensor_types = ["qkv_square", "mlp_fc", "mlp_down"]
    if rank_budgets is None:
        rank_budgets = [4, 8, 16, 32, 64, 128, 256]

    results = []

    shape_map = {
        "qkv_square": (arch.d_model, arch.d_model),
        "mlp_fc": (arch.d_model, arch.d_intermediate),
        "mlp_down": (arch.d_intermediate, arch.d_model),
    }

    weight_gen_map = {
        "trained": synthetic_trained_weight,
        "random": random_gaussian_weight,
    }

    for tensor_type in tensor_types:
        shape = shape_map[tensor_type]

        for rank in rank_budgets:
            # Trained weights only (power-law synthetic) for speed
            weight = synthetic_trained_weight(shape, seed=42)
            result = run_pgd_on_tensor(weight, rank_budget=rank)
            # Override arch name for cleaner tables
            result.arch_name = f"{arch_name} {tensor_type}"
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_scaling_experiment(
    arch_names: List[str] = None,
    rank_budgets: List[int] = None,
    tensor_types: List[str] = None,
) -> List[ScalingResult]:
    """
    Full scaling experiment across specified architectures.
    """
    if arch_names is None:
        arch_names = ["toy_128", "gpt2_small", "gemma3_270m"]
    if rank_budgets is None:
        rank_budgets = [4, 8, 16, 32, 64, 128, 256]
    if tensor_types is None:
        tensor_types = ["qkv_square", "mlp_fc", "mlp_down"]

    all_results = []

    for arch_name in arch_names:
        arch = ARCHS[arch_name]
        print(f"\n{'='*70}")
        print(f"Architecture: {arch_name}  (d_model={arch.d_model}, d_intermediate={arch.d_intermediate})")
        print(f"{'='*70}")

        results = sweep_architecture(arch_name, arch, tensor_types, rank_budgets)
        all_results.extend(results)

        # Print a quick table
        print(f"\n{'Tensor Type':<18} | {'R':<4} | {'Walltime':>8} | {'Resid.':>8} | {'Expl. Var':>9} | {'Modes':>5} | {'Comp. Ratio':>11}")
        print("-" * 85)
        prev_type = None
        for r in results:
            marker = " *" if r.modes_converged else ""
            print(
                f"{r.arch_name:<18} | {r.rank_budget:<4} | "
                f"{r.walltime_s:>7.3f}s | {r.residual_ratio:>8.4f} | "
                f"{r.explained_variance:>8.1%} | {r.modes_extracted:>5}{marker} | "
                f"{r.compression_ratio:>10.1f}x"
            )

    return all_results


# ---------------------------------------------------------------------------
# Entropy tracking at scale
# ---------------------------------------------------------------------------

def track_entropy_at_scale(
    arch_name: str = "gpt2_small",
    tensor_type: str = "qkv_square",
    rank_budget: int = 256,
) -> Dict:
    """
    Track how spectral entropy evolves with PGD modes at real model dimensions.
    """
    arch = ARCHS[arch_name]
    shape_map = {
        "qkv_square": (arch.d_model, arch.d_model),
        "mlp_fc": (arch.d_model, arch.d_intermediate),
        "mlp_down": (arch.d_intermediate, arch.d_model),
    }
    shape = shape_map[tensor_type]

    print(f"\n{'='*70}")
    print(f"Entropy Tracking: {arch_name} {tensor_type}  shape={shape}")
    print(f"{'='*70}")

    # Trained weight
    W_trained = synthetic_trained_weight(shape, seed=42)
    initial_svd = np.linalg.svd(W_trained, compute_uv=False)
    init_H = compute_spectral_entropy(initial_svd)
    init_eff_rank = effective_rank(initial_svd)

    # Random baseline
    W_random = random_gaussian_weight(shape, seed=42)
    random_svd = np.linalg.svd(W_random, compute_uv=False)
    rand_H = compute_spectral_entropy(random_svd)

    print(f"\n  Initial spectral entropy (trained): {init_H:.4f}  (eff. rank {init_eff_rank:.1f})")
    print(f"  Random baseline entropy:            {rand_H:.4f}")
    print(f"  Entropy gap:                         {rand_H - init_H:.4f}  ({(rand_H-init_H)/rand_H*100:.1f}% reduction)")

    # PGD decomposition
    print(f"\n  Running PGD with rank budget {rank_budget}...")
    t0 = time.perf_counter()
    modes, residual_norms = pgd_decompose(W_trained, num_modes=rank_budget, seed=42)
    elapsed = time.perf_counter() - t0
    print(f"  PGD walltime: {elapsed:.2f}s   modes extracted: {len(modes)}")

    # Track entropy mode-by-mode
    print(f"\n  {'Mode':<6} | {'Cumul. Explained Var':>20} | {'Cumul. Entropy':>14} | {'Residual Norm':>13}")
    print("  " + "-" * 62)

    cumulative = np.zeros(shape)
    cumulative_entropies = []
    cumulative_explained = []

    for i, mode_factors in enumerate(modes):
        # Reconstruct rank-1 from factors
        d = len(mode_factors)
        rank1 = _reconstruct_rank1(mode_factors, d)
        cumulative = cumulative + rank1

        # SVD of cumulative
        cur_svd = np.linalg.svd(cumulative, compute_uv=False)
        cur_H = compute_spectral_entropy(cur_svd)
        cur_explained = 1.0 - (np.linalg.norm(cumulative) ** 2) / (np.linalg.norm(W_trained) ** 2 + 1e-12)
        cumulative_entropies.append(cur_H)
        cumulative_explained.append(cur_explained)

        if i < 20 or i % 32 == 0 or i == len(modes) - 1:
            print(f"  {i:<6} | {cur_explained:>19.2%} | {cur_H:>14.4f} | {np.linalg.norm(W_trained - cumulative):>13.4f}")

    # Summary
    print("\n  --- Key observations ---")
    final_H = cumulative_entropies[-1]
    print(f"  Final entropy:   {final_H:.4f}  (was {init_H:.4f})")
    print(f"  Total reduction: {(init_H - final_H)/init_H*100:.1f}%")

    # Find modes-to-threshold
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        for i, exp in enumerate(cumulative_explained):
            if exp >= threshold:
                print(f"  Modes to {threshold:.0%} explained variance: {i}")
                break

    return {
        "arch_name": arch_name,
        "tensor_type": tensor_type,
        "shape": shape,
        "init_H": init_H,
        "rand_H": rand_H,
        "final_H": final_H,
        "init_eff_rank": init_eff_rank,
        "cumulative_entropies": cumulative_entropies,
        "cumulative_explained": cumulative_explained,
        "walltime_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("="*70)
    print("PGD SCALING EXPERIMENT")
    print("="*70)
    print("\nPhases:")
    print("  1. Scaling sweep: across architectures and rank budgets")
    print("  2. Entropy tracking: deep dive at realistic dimensions")
    print()

    # Phase 1: targeted scaling sweep (reduced set — just trained weights, key points)
    print("\n" + "#"*70)
    print("# PHASE 1: Targeted Scaling Sweep (trained weights only)")
    print("#"*70)

    arch_names_sweep = ["toy_128", "gpt2_small"]
    results = run_scaling_experiment(
        arch_names=arch_names_sweep,
        rank_budgets=[4, 16, 64, 128],
    )

    # Phase 2: entropy deep dive
    print("\n" + "#"*70)
    print("# PHASE 2: Entropy Deep Dive")
    print("#"*70)

    entropy_results = {}
    for arch_name in arch_names_sweep:
        for tensor_type in ["qkv_square", "mlp_fc"]:
            try:
                entropy_results[f"{arch_name}/{tensor_type}"] = track_entropy_at_scale(
                    arch_name=arch_name,
                    tensor_type=tensor_type,
                    rank_budget=128,
                )
            except Exception as e:
                print(f"  ERROR on {arch_name}/{tensor_type}: {e}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    return results, entropy_results


if __name__ == "__main__":
    results, entropy_results = main()
