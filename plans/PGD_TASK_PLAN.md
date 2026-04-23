# PGD Enrichment Task Plan

- [x] Audit: Analyze weight tensor structures and identify refitting objective.
- [x] RED: Implement failing test for PGD greedy enrichment in \`pgd_enrichment_test.py\`.
- [x] GREEN: Implement PGD algorithm in \`pgd_enrichment.py\`.
- [x] Refactor: Fix JAX compatibility bug (jnp.random), add fixed-point convergence early exit, and add @jax.jit.
- [ ] Final Polish: Fix static_argnums index in _reconstruct_rank1_jit and add safety epsilon to factor normalization.
- [x] Spec Compliance Review: Verify implementation matches the greedy enrichment spec.
- [x] Architectural Audit: Audit for numerical stability and convergence.
- [ ] Verification: Measure residual error and enrichment steps on a sample tensor.

