# Gemma 3 270M Refitting Harness Plan

- [ ] Audit: Define exact tensor shapes for Gemma 3 270M (Hidden: 1024, MLP: 4096, Layers: 16).
- [ ] RED: Implement a test that ensures the harness can load/generate tensors of the correct shapes and that PGD reduces their residual.
- [ ] GREEN: Implement the refitting harness in `refit_harness.py`.
    - Function to generate/load realistic weight distributions.
    - Loop over layers and projection types (QKV, MLP Gate/Up/Down).
    - Apply `pgd_decompose` from `pgd_enrichment.py`.
    - Collect stats (Residual vs. Modes).
- [ ] Spec Compliance Review: Verify the harness correctly applies PGD to the intended layers.
- [ ] Verification: Run the harness and output a summary report (e.g., CSV/Markdown) of the compression results.
