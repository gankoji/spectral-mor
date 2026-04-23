# Gemma 4 E2B: Deep Spectral Analysis

**Date:** 2026-04-22

Analyzes the full singular value distributions of real Gemma 4 E2B weights to understand:
1. Spectral decay profile (power-law vs. other distributions)
2. Flatness coefficient comparison across projection types
3. Spectral entropy vs. effective rank relationship

## Methodology

Load all 9 key projection tensors (q_proj, o_proj, down_proj × 3 layers) and compute full SVD.
Then fit power-law to singular value spectrum and compute:
- Power-law exponent α via log-log regression
- Flatness coefficient (variance of log-SVs normalized)
- Cumulative explained variance at R=8, 32, 128
- Spectral entropy H_s and effective rank

## Results

```python
# Load tensors
python inbox/agents/spectral-llm/mor/gemma4_deep_spectral.py
```

## Key Questions

1. Is Gemma 4 spectral decay consistent with the power-law hypothesis?
2. Do MLP layers have different spectral profiles than attention layers?
3. Does effective rank vary significantly across layers?
4. What fraction of variance is captured at R=8/32/128?