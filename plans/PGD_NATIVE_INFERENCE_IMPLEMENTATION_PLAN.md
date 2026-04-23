# PGD Native Inference — Implementation Plan

This document turns the research goals in `PGD_NATIVE_INFERENCE_RESEARCH_PLAN.md` into a concrete build plan aligned with the existing `mor/` codebase (`pgd_enrichment.py`, `refit_harness.py`, `gemma3_harness.py`, `gemma4_e2b_mor.py`, `test_pgd_reconstruction.py`).

## 1. Scope and anchor model

**Anchor:** `google/gemma-4-E2B-it` (shapes, mmap loading, and PGD analysis already exist in `gemma4_e2b_mor.py`). New code should be **model-key driven** (safetensors key patterns) so the same harness can later target E4B or another snapshot via environment variables or CLI config.

**Three execution modes** (primary deliverable):

| Mode | Meaning |
|------|--------|
| `dense_baseline` | Stock Hugging Face weights |
| `dense_pgd` | Replace selected `nn.Linear` weights with **dense** matrices rebuilt from PGD modes |
| `native_pgd` | Same layers logically, but matmul uses **on-the-fly** sum of rank-1 terms from stored factors (no full dense `W`) |

## 2. Foundation: PGD as a reusable compressed-weight API

**Gap:** `pgd_decompose` returns Python lists of NumPy vectors; there is no shared contract for reconstruction, matvec, serialization, or dtype/device.

**Add** a small module (e.g. `mor/pgd_weight.py`, or extend `pgd_enrichment.py` if you prefer fewer files):

1. **`PGDLinearSpec`** (dataclass): `out_features`, `in_features`, `rank`, stacked factors (e.g. `[R, out]` and `[R, in]` after stacking modes), `dtype`, optional `layer_key`.
2. **`reconstruct_dense(spec)`** — must match `test_pgd_reconstruction.py` semantics (sum of outer products; respect how alpha is folded into `modes[0][0]` in `pgd_enrichment.py`).
3. **`matvec_native(spec, x)`** — for weight layout `(out, in)` and column vector `x`: \(y = \sum_r u_r (v_r^\top x)\) with correct broadcasting for batched `x`.
4. **Numerical parity tests** (e.g. `mor/test_pgd_linear_ops.py` or extend `test_pgd_reconstruction.py`):
   - `allclose(reconstruct_dense @ x, matvec_native(x))` for random `x` at float32/float64.
   - Random 2D `W`, decompose, compare Frobenius error to existing checks.

**Torch path:** add `matvec_native_torch` and `reconstruct_dense_torch` so inference avoids slow Python loops over rank (vectorize over the rank dimension).

**Acceptance:** parity tests pass; documented complexity \(O(R \cdot (\mathrm{out} + \mathrm{in}))\) per token position versus \(O(\mathrm{out} \cdot \mathrm{in})\) for dense matmul.

## 3. Phase A — Fidelity harness on real weights

**Research mapping:** Phase A (reconstruction fidelity).

**Add** `mor/pgd_fidelity_harness.py` (or `mor/eval/` subpackage):

1. **Inputs:** safetensors path (reuse mmap pattern from `gemma4_e2b_mor.py`), list of `(layer_idx, proj_suffix)` — e.g. `down_proj`, `q_proj`, and optionally `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`.
2. **Per tensor:** run `pgd_decompose` for a **sweep** of \(R \in \{8, 32, 128, \ldots\}\); record Frobenius relative error and explained variance (partially mirrored by `PGDResult` in `gemma4_e2b_mor.py`).
3. **Optional SVD baseline** (research Baseline 4): truncated SVD at the same rank; store \(\|W - W_{\mathrm{svd}}(R)\|_F\) for comparison.
4. **Activation proxy:** without a full transformer forward, load one real `W`, sample \(x \sim \mathcal{N}(0,1)\) (or cached activations once a small capture hook exists); report \(\|\hat W x - W x\| / \|W x\|\) and cosine similarity of outputs.
5. **Outputs:** CSV plus a short summary (printed markdown table or template), one row per `(layer, proj, R)`.

**Acceptance:** one command reproduces rank–error curves for the tensors in the research plan; spot-check matches manual decomposition on one layer.

## 4. Phase B — Dense PGD-reconstructed inference

**Research mapping:** Phase B (H1: reconstructed weights usable).

**Add** `mor/pgd_hf_substitution.py` and a driver such as `mor/run_pgd_perplexity.py`:

1. Load the model with `transformers`; control `device_map` and `torch_dtype` via CLI.
2. **Module selection:** map Gemma module names to safetensors keys (reuse conventions from analysis scripts); support `--layers 0,17,34` and `--projections down_proj` first (per research: start with MLP / `down_proj`).
3. **Per target `nn.Linear`:** read CPU float32 weight → `pgd_decompose` with rank `R` → `reconstruct_dense` → write back into the parameter tensor (matching shape and model dtype).
4. **Metrics:** perplexity / negative log-likelihood on a pinned text snippet or `datasets` sample (fixed seed, max tokens); `torch.cuda.max_memory_allocated` (reset per run); wall time for a forward pass at prefill length \(N\).

**Acceptance:** a “no substitution” or \(R=0\) run matches baseline perplexity within floating noise; each config emits one JSON or CSV row.

## 5. Phase C — Native factorized inference

**Research mapping:** Phase C (H2: execution without dense `W`).

**Implementation pattern (choose one primary path):**

- **Preferred:** replace `nn.Linear` with `PGDLinear` whose `forward` calls `matvec_native_torch`; store factors as buffers (non-trainable).
- **Alternative:** forward hooks that bypass dense matmul — fewer classes but easier to break on refactors.

**Integration:**

1. Same layer/projection selection as Phase B.
2. Load factors into `PGDLinear`; do not materialize a full dense `weight` for substituted layers (or use a custom module without a standard `weight` for fair memory accounting).
3. **Comparison:** same prompts as Phase B for arms: baseline → `dense_pgd` → `native_pgd`.

**Acceptance:** logits or perplexity for `native_pgd` close to `dense_pgd` at the same \(R\) (differences only from op order / float noise); peak memory for `native_pgd` ≤ `dense_pgd` when a large fraction of MLP parameters is substituted.

## 6. Phase D — Systems benchmark

**Research mapping:** Phase D (deployment-style evaluation).

**Add** `mor/pgd_inference_benchmark.py` (or fold into the unified harness):

- Prefill latency, decode latency (if `generate` is used), tokens/sec, peak memory.
- Optional: checkpoint **disk** size once an on-disk factor format exists.
- **Stability:** \(K\) prompts; variance of perplexity; optional max hidden-state norm drift with a debug hook.

**Acceptance:** one table comparing all modes for identical `(model, R, layers, seq_len, device)`.

## 7. Unified compressed inference evaluation harness

Single CLI entry point, e.g. `mor/compressed_inference_harness.py`:

```text
--mode {baseline,dense_pgd,native_pgd,all}
--model google/gemma-4-E2B-it
--safetensors-path ...   # optional override of HF cache path
--rank 128
--layers 0,17,34
--projections down_proj,q_proj
--max-tokens ...
--output-json results.json
```

Orchestrates: argument parsing → selected mode(s) → perplexity, memory, timing, optional Phase A stats (or subprocess / subcommand to the fidelity harness).

## 8. Experiment matrix (initial)

| Run ID | Layers | Projections | R | Mode |
|--------|--------|-------------|---|------|
| A1 | 17 | `down_proj` | 8, 32, 128 | fidelity only |
| B1 | 17 | `down_proj` | 128 | `dense_pgd` |
| B2 | 0, 17, 34 | `down_proj` | 128 | `dense_pgd` |
| C1 | 17 | `down_proj` | 128 | `native_pgd` |
| D1 | compare B2 vs C1 | — | — | benchmark |

**Optional:** SVD rank-\(R\) row for the same tensors as A1.

**Pass/fail thresholds** (set after first baseline variance):

- **Tier 1:** `dense_pgd` perplexity within X% of baseline for single-layer `down_proj` at layer 17.
- **Tier 2:** `native_pgd` within Y% of `dense_pgd` at same \(R\); memory reduction ≥ Z% when substituting ≥ N% of MLP parameters.
- **Tier 3:** tokens/sec not below baseline by more than an agreed factor (native path may need custom CUDA if PyTorch overhead dominates).

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Low Frobenius error but bad logits | Always report perplexity and activation relative error (Phase A); start with one layer. |
| Native path too slow | Torch-vectorized rank; profile vs dense; document regimes where native wins. |
| Errors stack with depth | CLI for layer sets; curve perplexity vs number of substituted layers. |
| Tensor classes behave differently | Per-projection dimension tables keyed by layer (E2B early/late split already documented in `gemma4_e2b_mor.py`). |

## 10. Implementation order (sprints)

1. **Sprint 0:** PGD spec + reconstruct + native matvec + parity tests (Section 2).
2. **Sprint 1:** Phase A fidelity CLI on mmap weights (Section 3).
3. **Sprint 2:** Phase B HF dense substitution + perplexity driver (Section 4).
4. **Sprint 3:** `PGDLinear` + Phase C parity vs `dense_pgd` (Section 5).
5. **Sprint 4:** Unified harness + Phase D benchmarks + frozen experiment matrix (Sections 6–8).

## 11. Dependencies and reproducibility

- **Packages:** `torch`, `transformers`, `safetensors`, `numpy`; pin versions in `requirements.txt` when the harness lands.
- **Secrets:** Hugging Face token for model download where required.
- **Hardware:** record GPU model and driver in every `results.json` for comparability.

## 12. Relation to the research plan

| Research doc | This implementation doc |
|--------------|-------------------------|
| Phases A–D | Sections 3–6 |
| Baselines 1–4 | `dense_baseline`, `dense_pgd`, `native_pgd`, optional SVD in Phase A |
| “Compressed inference evaluation harness” | Section 7 |
| Immediate next steps (single tensor → MLP → harness) | Sprints 1–4 and matrix Section 8 |
