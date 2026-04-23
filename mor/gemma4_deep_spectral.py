#!/usr/bin/env python3
"""Deep spectral analysis of Gemma 4 E2B real weights."""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
from safetensors.torch import safe_open
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).parent))


def compute_spectral_entropy(S: np.ndarray, eps: float = 1e-12) -> float:
    """Spectral entropy of a normalized singular value distribution."""
    S_sq = S ** 2
    p = S_sq / (np.sum(S_sq) + eps)
    return -np.sum(p * np.log(p + eps))


def effective_rank(S: np.ndarray) -> float:
    """Effective rank = exp(spectral entropy)."""
    return np.exp(compute_spectral_entropy(S))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAFETENSOR_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf/model.safetensors"
)

LAYERS = [0, 17, 34]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_power_law(sv: np.ndarray) -> dict:
    """Fit sigma_i ~ i^(-alpha) via log-log regression."""
    sv = np.sort(sv)[::-1]
    i = np.arange(1, len(sv) + 1)
    mask = sv > 0
    log_i = np.log(i[mask])
    log_sv = np.log(sv[mask])
    slope, intercept, r_val, p_val, std_err = linregress(log_i, log_sv)
    return {
        "alpha": -slope,
        "intercept": intercept,
        "r_squared": r_val ** 2,
        "p_value": p_val,
        "std_err": std_err,
    }


def flatness_coefficient(sv: np.ndarray) -> float:
    """1 - std(log(sv_norm)) / |mean(log(sv_norm))|. Higher = more uniform."""
    sv_norm = sv / (np.linalg.norm(sv) + 1e-12)
    log_sv = np.log(sv_norm + 1e-12)
    return 1.0 - np.std(log_sv) / (np.abs(np.mean(log_sv)) + 1e-12)


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient of a sorted array (0 = perfectly uniform, 1 = maximally unequal)."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x)
    n = len(x)
    cumulative = np.cumsum(x)
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * cumulative[-1])) - (n + 1) / n


def spectral_analysis(weight: np.ndarray, name: str) -> dict:
    """Compute full spectral statistics for a weight matrix."""
    m, n = weight.shape
    min_dim = min(m, n)

    t0 = time.perf_counter()
    full_svd = np.linalg.svd(weight, compute_uv=False)
    svd_time = time.perf_counter() - t0

    # Normalize to unit Frobenius
    sv = full_svd / (np.linalg.norm(full_svd) + 1e-12)

    # Entropy & effective rank
    H_s = compute_spectral_entropy(sv)
    er = effective_rank(sv)

    # Power law fit
    pl = fit_power_law(sv)
    flat = flatness_coefficient(sv)

    # Cumulative explained variance at various R
    cumvar = {}
    for R in [8, 16, 32, 64, 128, 256, 512]:
        if R <= len(sv):
            cumvar[str(R)] = float(np.sum(sv[:R] ** 2))

    # Condition number proxy
    svr = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")

    # Spectral spread (std of log-SVs)
    log_sv = np.log(sv + 1e-12)
    spectral_spread = float(np.std(log_sv))

    # Gini of log-SVs
    gini = gini_coefficient(log_sv)

    return {
        "name": name,
        "shape": list(weight.shape),
        "min_dim": min_dim,
        "svd_time_s": svd_time,
        "spectral_entropy": float(H_s),
        "effective_rank": float(er),
        "effective_rank_ratio": float(er / min_dim),
        "power_law_alpha": pl["alpha"],
        "power_law_r2": pl["r_squared"],
        "flatness_coefficient": float(flat),
        "sv_condition_number": svr,
        "spectral_spread": spectral_spread,
        "gini_log_sv": float(gini),
        "cumvar_at_R": cumvar,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []

    with safe_open(SAFETENSOR_PATH, framework="pt", device="cpu") as f:
        for layer in LAYERS:
            for proj in ["q_proj", "o_proj", "down_proj"]:
                if proj in ["q_proj", "o_proj"]:
                    key = f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
                else:
                    key = f"model.language_model.layers.{layer}.mlp.{proj}.weight"

                weight = f.get_tensor(key).float().numpy().astype(np.float32)
                name = f"{proj}_L{layer}"

                print(
                    f"  Analyzing {proj} L{layer} ({weight.shape[0]}x{weight.shape[1]})...",
                    end=" ",
                    flush=True,
                )
                r = spectral_analysis(weight, name)
                results.append(r)
                print(
                    f"H_s={r['spectral_entropy']:.3f}, alpha={r['power_law_alpha']:.3f}, "
                    f"R2={r['power_law_r2']:.3f}"
                )

    # ---- Print summary table ----
    print("\n" + "=" * 110)
    print("DEEP SPECTRAL ANALYSIS: Gemma 4 E2B")
    print("=" * 110)

    hdr = "%-20s | %-12s | %6s | %8s | %6s | %5s | %6s | %6s | %7s" % (
        "Name", "Shape", "H_s", "Eff Rank", "Ratio", "alpha", "R2", "Flat", "Spread"
    )
    print(hdr)
    print("-" * 110)

    for r in results:
        shape_str = "%dx%d" % (r["shape"][0], r["shape"][1])
        print(
            "%-20s | %-12s | %6.3f | %8.1f | %6.1f%% | %5.3f | %6.3f | %6.3f | %7.3f"
            % (
                r["name"],
                shape_str,
                r["spectral_entropy"],
                r["effective_rank"],
                r["effective_rank_ratio"] * 100,
                r["power_law_alpha"],
                r["power_law_r2"],
                r["flatness_coefficient"],
                r["spectral_spread"],
            )
        )

    # ---- Cumulative variance table ----
    print("\n--- Cumulative Explained Variance ---")
    hdr2 = "%-20s | %7s | %7s | %7s | %7s | %7s | %7s | %7s" % (
        "Name", "R=8", "R=16", "R=32", "R=64", "R=128", "R=256", "R=512"
    )
    print(hdr2)
    print("-" * 75)
    for r in results:
        cv = r["cumvar_at_R"]
        print(
            "%-20s | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%%"
            % (
                r["name"],
                cv.get("8", 0) * 100,
                cv.get("16", 0) * 100,
                cv.get("32", 0) * 100,
                cv.get("64", 0) * 100,
                cv.get("128", 0) * 100,
                cv.get("256", 0) * 100,
                cv.get("512", 0) * 100,
            )
        )

    # ---- Save results ----
    out_path = Path(__file__).parent / "gemma4_weights" / "gemma4_deep_spectral.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved to: %s" % out_path)


if __name__ == "__main__":
    main()
