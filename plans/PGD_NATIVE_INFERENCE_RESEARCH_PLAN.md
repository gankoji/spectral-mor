# PGD-Compressed Inference Research Plan

## Research Goal

We have already demonstrated that Proper Generalized Decomposition (PGD) can compress LLM tensors by very large factors. The next research question is whether those compressed representations are *actually usable at inference time*.

The key distinction is between:

1. **Compression as storage** — reconstructing a dense weight tensor after factorization.
2. **Compression as execution** — using the factorized PGD representation directly during inference.

The second case is the real research objective, because it could reduce both memory and compute.

## Core Research Question

Can a PGD-compressed LLM weight tensor be used for inference with acceptable quality, while reducing memory and runtime enough to justify the compression pipeline?

### Sub-questions

- Does the approximation preserve perplexity and output quality?
- Can inference run directly from PGD factors without dense reconstruction?
- Is the factorized execution numerically stable across layers and prompts?
- Does the method scale across tensor types and model sizes?

## Hypothesis

A useful working hypothesis is:

> PGD factors can be evaluated on-the-fly during inference, replacing dense weight tensors with separable factorized operators, and the resulting model will preserve most of the original language modeling behavior at substantially lower memory cost.

This should be tested in two stages:

- **H1: Reconstructed PGD weights are usable.**
  Build dense tensors from PGD factors and run standard inference.

- **H2: Native compressed inference is usable.**
  Never materialize the full tensor; compute layer outputs directly from the factors.

H1 is the bridge; H2 is the actual research target.

## Experimental Ladder

### Phase A — Reconstruction Fidelity

Goal: verify that PGD factors can faithfully approximate the original tensor.

#### What to measure

- Tensor reconstruction error
- Layer output error on real activations
- Rank vs. error curves
- Sensitivity by tensor type:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `up_proj`
  - `down_proj`

#### Why this matters

Before asking whether compressed inference works, we need to know whether the factorization preserves the layer transformation well enough.

### Phase B — Dense Reconstructed Inference

Goal: replace original weights with PGD-reconstructed dense weights and run standard inference.

This is the easiest downstream test because it isolates approximation quality from the complexity of factorized execution.

#### Experiment

- Load a baseline model
- Replace a chosen subset of weights with PGD-reconstructed versions
- Evaluate on held-out text

#### Metrics

- Perplexity delta
- Next-token accuracy / log-likelihood
- Layerwise activation drift
- GPU memory footprint
- End-to-end latency

#### Interpretation

- If this fails badly, the factorization is not preserving semantics.
- If this succeeds, the model is at least robust to the approximation.

### Phase C — Native Compressed Inference

Goal: compute linear layer outputs directly from PGD factors without reconstructing the dense matrix.

This is the crucial proof of usability.

For a linear layer \( y = Wx \), if PGD provides a separated representation, the inference path should exploit that separability directly rather than building \(W\) explicitly.

#### Comparison arms

1. Dense baseline inference
2. Dense inference with reconstructed PGD weights
3. Direct factorized inference

#### Key question

How much overhead is introduced by evaluating factors, and how much is saved by avoiding dense tensors?

### Phase D — System-Level Evaluation

Goal: determine whether compressed inference is useful in a practical deployment setting.

#### Metrics

- Peak memory
- Total parameter storage
- Token throughput
- Prefill latency
- Decode latency
- Accuracy / perplexity
- Numerical drift over long generations

#### Target outcome

A Pareto improvement:

- lower memory
- comparable quality
- acceptable or improved runtime

## Recommended Technical Comparison Arms

To make the study convincing, include these baselines:

### Baseline 1: Original dense model

Reference point for quality and runtime.

### Baseline 2: Dense PGD reconstruction

Same model structure, but with reconstructed weights.

### Baseline 3: Direct PGD factorized inference

The candidate for reduced compute and memory.

### Baseline 4: Optional low-rank baseline

Compare against SVD/LoRA-style low-rank approximation to characterize where PGD is stronger or weaker.

## Which Layers to Start With

Do not start with the whole model. Start with the layers most likely to be compressible and easiest to evaluate.

Recommended order:

1. Single linear projection
2. MLP block
3. Attention block
4. Several consecutive layers
5. Whole-model substitution

The MLP projections, especially `down_proj`, appear to be strong initial targets based on the current compression results in `spectral-llm-research-summary.md:67-82` and `spectral-llm-research-summary.md:83-103`.

## Metrics That Matter

The next research stage should not be judged only by reconstruction error.

### Quality metrics

- Perplexity on a held-out corpus
- Average token log-likelihood
- Task-specific accuracy if available

### Approximation metrics

- Frobenius reconstruction error
- Relative activation error
- Layerwise cosine similarity
- Output KL divergence

### Systems metrics

- Peak memory
- Model checkpoint size
- Token/sec
- Prefill latency
- Decode latency
- Decomposition time

### Stability metrics

- Variance across prompts
- Error accumulation over depth
- Sensitivity to rank `R`
- Sensitivity to layer selection

## Success Criteria

### Tier 1: Scientifically positive

- Dense PGD-reconstructed model preserves perplexity reasonably well
- Error stays bounded across layers
- Result is better than naive low-rank baselines in at least some regimes

### Tier 2: Methodologically strong

- Direct factorized inference works without materializing dense weights
- Memory savings are substantial
- Runtime overhead is not prohibitive

### Tier 3: Deployment-relevant

- End-to-end inference cost improves meaningfully
- Compressed model is stable enough to be used as a drop-in option for some workloads

## Key Risks

### Risk 1: Low reconstruction error may not imply low output error

A tensor can be close in norm but still alter logits enough to hurt generation.

### Risk 2: Direct factorized inference may be computationally awkward

The math may compress storage but still be too slow if factor evaluation is expensive.

### Risk 3: Layer interactions may amplify small errors

A few good layers may still produce a bad whole-model outcome.

### Risk 4: Different tensor classes may behave very differently

Attention and MLP tensors may not compress or behave identically.

## Best Next Deliverable

The cleanest next deliverable is a **compressed inference evaluation harness** with three modes:

1. Original
2. PGD-reconstructed
3. PGD-native factorized

That harness should report:

- perplexity
- memory
- runtime
- error statistics
- per-layer substitution results

This naturally extends the refit-harness direction already identified in `spectral-llm-research-summary.md:122-125`.

## Suggested Research Statement

> We have shown that PGD can compress LLM tensors dramatically. The next step is to determine whether this compressed representation can be used as an execution format, not merely as a storage format. We will test whether factorized PGD tensors can support inference directly, with acceptable perplexity, stability, and runtime cost relative to the original dense model.

## Immediate Next Step

Start with this sequence:

1. Single-tensor inference test
   - one projection
   - one prompt set
   - compare dense vs reconstructed vs factorized

2. MLP block test
   - likely the best candidate for compression

3. Whole-layer replacement
   - test end-to-end effect on perplexity

4. Build the compressed inference harness
   - make the experiment repeatable

## Recommended Follow-Up

The next useful artifact would be a concrete experiment matrix for a target model such as Gemma 4 E2B, with exact run conditions, metrics, and pass/fail thresholds.
