# Task 2: Continuous Flow Fitting Plan

- [x] Audit: Map the "trajectory" of a specific projection type (e.g., Attention Key) across all layers of Gemma 3.
- [x] RED: Implement a test that verifies a Neural ODE can reconstruct the weight trajectory across layers with lower error than a simple average.
- [x] GREEN: Implement the ODE Fitting utility in `ode_flow.py`.
    - Function to treat layer index $ as time $.
    - Parametrize the weight (t)$ as a polynomial or small MLP.
    - Minimize $\sum_l ||W(t_l) - W_l||^2$.
- [x] Verification: Measure the reconstruction error of the continuous weight flow vs. the original discrete weights.
- [x] Update Research Note: Document findings on depth-wise "spectral" smoothness.
