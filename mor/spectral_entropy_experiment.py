"""
Spectral Entropy Tracking Experiment
=====================================

This experiment tracks the spectral entropy of weight matrices during PGD decomposition
to understand the structure of trained LLM weights vs. random initialization.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pgd_enrichment import _reconstruct_rank1, pgd_decompose


@dataclass
class SpectralEntropyResult:
    """Results from spectral entropy analysis."""
    weight_shape: Tuple[int, ...]
    initial_entropy: float
    final_entropy: float
    entropy_reduction: float
    initial_singular_values: np.ndarray
    final_singular_values: np.ndarray
    mode_entropies: List[float]
    residual_ratios: List[float]
    is_trained: bool


def compute_spectral_entropy(S: np.ndarray, eps: float = 1e-12) -> float:
    S_sq = S ** 2
    p = S_sq / (np.sum(S_sq) + eps)
    return -np.sum(p * np.log(p + eps))


def compute_spectral_flatness(S: np.ndarray, eps: float = 1e-12) -> float:
    S_sq = S ** 2 + eps
    return np.exp(np.mean(np.log(S_sq))) / np.mean(S_sq)


def compute_effective_rank(S: np.ndarray, eps: float = 1e-12) -> float:
    return np.exp(compute_spectral_entropy(S, eps))


def full_singular_values_from_modes(modes: List[List[np.ndarray]], d: int) -> np.ndarray:
    if not modes:
        return np.array([])

    W = np.zeros([len(modes[0][i]) for i in range(d)])
    for mode_factors in modes:
        rank1 = _reconstruct_rank1(mode_factors, d)
        W = W + rank1

    return np.linalg.svd(W, compute_uv=False)


def analyze_weight_spectral(weight: np.ndarray, num_modes: int = 128, is_trained: bool = True) -> SpectralEntropyResult:
    shape = weight.shape
    initial_svd = np.linalg.svd(weight, compute_uv=False)
    initial_entropy = compute_spectral_entropy(initial_svd)
    pgd_modes, residual_norms = pgd_decompose(weight, num_modes=num_modes, seed=42)

    mode_entropies = []
    residual_ratios = []
    cumulative = None

    for i, mode_factors in enumerate(pgd_modes):
        rank1 = _reconstruct_rank1(mode_factors, len(shape))
        cumulative = rank1 if cumulative is None else cumulative + rank1
        current_svd = np.linalg.svd(cumulative, compute_uv=False)
        mode_entropies.append(compute_spectral_entropy(current_svd))
        residual_ratios.append(residual_norms[i + 1] / residual_norms[0])

    final_svd = full_singular_values_from_modes(pgd_modes, len(shape)) if pgd_modes else initial_svd
    final_entropy = compute_spectral_entropy(final_svd)

    return SpectralEntropyResult(
        weight_shape=shape,
        initial_entropy=initial_entropy,
        final_entropy=final_entropy,
        entropy_reduction=initial_entropy - final_entropy,
        initial_singular_values=initial_svd,
        final_singular_values=final_svd,
        mode_entropies=mode_entropies,
        residual_ratios=residual_ratios,
        is_trained=is_trained,
    )


def _demo_summary() -> Dict[str, float]:
    rng = np.random.default_rng(42)
    trained = rng.standard_normal((32, 32))
    random = rng.standard_normal((32, 32))
    trained_result = analyze_weight_spectral(trained, num_modes=8, is_trained=True)
    random_result = analyze_weight_spectral(random, num_modes=8, is_trained=False)
    return {
        "trained_initial_entropy": trained_result.initial_entropy,
        "trained_final_entropy": trained_result.final_entropy,
        "random_initial_entropy": random_result.initial_entropy,
        "random_final_entropy": random_result.final_entropy,
    }


if __name__ == "__main__":
    summary = _demo_summary()
    output_path = os.path.join(os.path.dirname(__file__), "spectral_entropy_results.md")
    with open(output_path, "w") as f:
        f.write(json.dumps(summary, indent=2))
    print(f"Summary written to {output_path}")
