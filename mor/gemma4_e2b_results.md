# Gemma 4 E2B: Real-Weight PGD MOR Results

**Date:** 2026-04-22
**Branch:** `spectral-llm/experiments` (worktree: `../slipbox-spectral-llm`)

## Context

The previous PGD scaling experiment used **synthetic power-law weights** that simulated LLM spectral structure. This run uses **real Gemma 4 E2B weights** loaded directly from the safetensor checkpoint via memory-mapped I/O.

Model: `google/gemma-4-E2B-it` (~2B params)
- Language model hidden dim: **1536** (constant across all layers)
- Attention Q projection: 2048×1536 (layers 0–15) → 4096×1536 (layers 16–34)
- Attention O projection: 1536×2048 (layers 0–15) → 1536×4096 (layers 16–34)
- MLP down projection: 1536×6144 (layers 0–15) → 1536×12288 (layers 16–34)

## Compression Results

### Compression Ratio (× storage reduction)

| Projection | Shape | R=8 | R=32 | R=128 |
|:---|---:|:---:|:---:|:---:|
| `q_proj` | 2048×1536 | 109.7× | 27.4× | 6.9× |
| `o_proj` | 1536×2048 | 109.7× | 27.4× | 6.9× |
| `down_proj` | 1536×6144 | 153.6× | 38.4× | 9.6× |
| `down_proj` | 1536×12288 | **170.7×** | **42.7×** | **10.7×** |
| `q_proj` | 4096×1536 | 139.6× | 34.9× | 8.7× |
| `o_proj` | 1536×4096 | 139.6× | 34.9× | 8.7× |

### Explained Variance (% of Frobenius norm captured)

| Projection | Shape | R=8 | R=32 | R=128 |
|:---|---:|:---:|:---:|:---:|
| `q_proj` | 2048×1536 | 10% | 24% | 54% |
| `o_proj` | 1536×2048 | 4% | 11% | 33% |
| `down_proj` | 1536×6144 | 6% | 12% | 28% |
| `down_proj` | 1536×12288 | 3% | 6% | 19% |
| `q_proj` | 4096×1536 | 9% | 22% | 51% |
| `o_proj` | 1536×4096 | 3% | 8% | 26% |

### Walltime (seconds)

| Projection | Shape | R=8 | R=32 | R=128 |
|:---|---:|:---:|:---:|:---:|
| `q_proj` | 2048×1536 | 0.3s | 0.8s | 3.3s |
| `o_proj` | 1536×2048 | 0.3s | 0.8s | 3.3s |
| `down_proj` | 1536×6144 | 0.8s | 2.7s | 11.5s |
| `down_proj` | 1536×12288 | 1.6s | 5.6s | 23.9s |
| `q_proj` | 4096×1536 | 0.5s | 1.6s | 6.6s |
| `o_proj` | 1536×4096 | 0.6s | 1.7s | 6.9s |

## Spectral Entropy of Real Gemma 4 E2B Weights

| Projection | Shape | H_s | Effective Rank | Eff Rank / min(m,n) |
|:---|---:|:---:|:---:|:---:|
| `q_proj` | 2048×1536 | 6.59 | 730.6 | 47.6% |
| `o_proj` | 1536×2048 | 6.57 | 716.9 | 46.7% |
| `down_proj` | 1536×6144 | 7.00 | 1101.2 | 71.7% |
| `down_proj` | 1536×12288 | 7.14 | 1256.0 | 81.8% |
| `q_proj` | 4096×1536 | 6.31 | 552.6 | 36.0% |
| `o_proj` | 1536×4096 | 6.92 | 1011.9 | 65.9% |

## Key Findings

### 1. Compression is exceptional, but variance explained is low

At R=8, the `down_proj` (1536×12288) achieves **170× compression** but only captures **3% of variance**. This is consistent with the power-law spectral hypothesis: the tail singular values contribute disproportionately to Frobenius norm even when the effective rank is high.

### 2. Effective rank is much higher than for GPT-2

GPT-2 124M had effective rank ≈ 36% of dimension. Gemma 4 E2B has 47–82% effective rank depending on projection type. This suggests Gemma 4 weights are **less compressible** in the PGD sense (higher soft-rank).

### 3. MLP layers are more compressible than attention layers

`down_proj` (MLP output) has 72–82% effective rank ratio, vs. `q_proj`/`o_proj` at 36–67%. This mirrors the GPT-2 finding but is less extreme.

### 4. Q projection outperforms O projection at equal R

At R=128 on 2048×1536, `q_proj` captures 54% of variance vs. 33% for `o_proj`. The Q projection has more concentrated spectral mass — it's more "low-rank" in the PGD sense.

### 5. Later layers are harder to compress

Late-layer `q_proj` (4096×1536) has only 36% effective rank ratio vs. 47.6% for early layers — later layers are more "diffuse" in spectral space.

### 6. Walltime is tractable

Even the largest tensor (1536×12288) at R=128 takes only **24 seconds**. At R=8, all tensors are under 2s. Full model refitting (35 layers × 5 projection types × 35 weights) would take on the order of minutes.

## Interpretation: The MOR Problem at Scale

The central tension: **high compression ratios coexist with low explained variance**. This is not a bug — it's the nature of power-law spectra.

For a d×d matrix with power-law singular values σ_i ∝ i^(-α), the effective rank grows as d^(1-α) while the top-R approximation error scales as R^(1-2α) for appropriate α. At Gemma 4's dimensions (d=1536) with typical α ≈ 0.5–1.0, this explains the 3–10% variance at R=8.

**The key question for MOR is not "does PGD capture variance?" but "does PGD capture the task-relevant subspace?"** This requires downstream evaluation (perplexity on held-out text), not just reconstruction metrics.

## Files

- `gemma4_e2b_mor.py` — Main experiment script (memory-mapped safetensors)
- `gemma4_weights/gemma4_e2b_pgd_results.json` — Raw results
- `gemma4_weights/E2B/E2B_sample_weights.npz` — Sample weight tensors

## Deep Spectral Analysis (gemma4_deep_spectral.py)

### Power-Law Fits

| Projection | Shape | H_s | Eff Rank | Ratio | α (power-law) | R² | Flatness | Spread |
|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `q_proj` L0 | 2048×1536 | 6.59 | 730.6 | 47.6% | **0.624** | 0.707 | 0.821 | 0.733 |
| `o_proj` L0 | 1536×2048 | 6.57 | 716.9 | 46.7% | **0.636** | 0.703 | 0.818 | 0.751 |
| `down_proj` L0 | 1536×6144 | 7.00 | 1101.2 | 71.7% | **0.348** | 0.850 | 0.902 | 0.374 |
| `q_proj` L17 | 2048×1536 | 6.21 | 497.8 | 32.4% | **0.860** | 0.770 | 0.781 | 0.970 |
| `o_proj` L17 | 1536×2048 | 6.74 | 844.6 | 55.0% | **0.648** | 0.675 | 0.810 | 0.780 |
| `down_proj` L17 | 1536×12288 | 7.14 | 1256.0 | 81.8% | **0.294** | 0.683 | 0.907 | 0.351 |
| `q_proj` L34 | 4096×1536 | 6.31 | 552.6 | 36.0% | **0.770** | 0.805 | 0.803 | 0.849 |
| `o_proj` L34 | 1536×4096 | 6.92 | 1011.9 | 65.9% | **0.518** | 0.618 | 0.836 | 0.652 |
| `down_proj` L34 | 1536×12288 | 7.17 | 1296.0 | 84.4% | **0.268** | 0.686 | 0.915 | 0.319 |

**Flatness coefficient:** 1 - std(log σ)) / |mean(log σ)|. Higher = more uniform (flatter) spectrum. Values >0.9 are extremely flat.

**α (power-law exponent):** σ_i ∝ i^(-α). Lower α = flatter decay = harder to compress via low-rank. MLP down_proj consistently has α≈0.27–0.35; attention projections are α≈0.52–0.86.

### Cumulative Explained Variance

| Projection | R=8 | R=16 | R=32 | R=64 | R=128 | R=256 | R=512 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `q_proj` L0 | 10% | 14% | 18% | 26% | 38% | 56% | 78% |
| `o_proj` L0 | 10% | 13% | 17% | 25% | 38% | 57% | 79% |
| `down_proj` L0 | 6% | 8% | 12% | 18% | 28% | 43% | 63% |
| `q_proj` L17 | 10% | 16% | 24% | 37% | 54% | 73% | 89% |
| `o_proj` L17 | 4% | 6% | 11% | 19% | 33% | 53% | 78% |
| `down_proj` L17 | 4% | 6% | 8% | 13% | 21% | 34% | 57% |
| `q_proj` L34 | 9% | 14% | 22% | 34% | 51% | 71% | 88% |
| `o_proj` L34 | 3% | 5% | 8% | 15% | 26% | 45% | 70% |
| `down_proj` L34 | 3% | 4% | 7% | 11% | 19% | 33% | 55% |

**Key observation:** Even R=512 (33% of min-dim for 1536×12288) captures only 55–57% of down_proj variance. MLP layers are genuinely hard to compress via rank truncation.

## Key Findings

### 1. Gemma 4 spectra are flatter than GPT-2

α ranges from 0.27 (MLP down_proj late layers) to 0.86 (q_proj mid layers). The MLP layers are remarkably flat — α≈0.27–0.35 means the spectrum barely decays, consistent with highly diffuse/reparameterized MLP representations.

### 2. Power-law R² varies (0.62–0.85)

The fit quality is moderate: R²=0.85 for down_proj L0 is good, but R²=0.62 for o_proj L34. The real spectra may follow a mixture distribution or have a break point rather than a pure power law.

### 3. Attention projections vary more across layers

q_proj L0 has α=0.624 but q_proj L17 has α=0.860 — meaning mid-layers concentrate spectral mass much more than early layers. This suggests mid-layers may be more amenable to MOR.

### 4. Cumulative variance confirms the MOR tension

At R=8, even the best-case (q_proj L0) captures only 10.5% of variance. At R=128, q_proj L17 captures 54.4% — this is the practical MOR sweet spot where we get both meaningful compression (27.4× on 2048×1536) and decent reconstruction.

1. **Downstream evaluation**: Measure perplexity after PGD-reconstructed weights are substituted
2. **Refitting harness**: Build `refit_harness.py` to refit Gemma 4 E2B end-to-end
3. **Larger models**: Download E4B, 26B-A4B, 31B and compare spectral properties
4. **Stopping criterion**: Implement entropy-minimum early stopping for PGD
