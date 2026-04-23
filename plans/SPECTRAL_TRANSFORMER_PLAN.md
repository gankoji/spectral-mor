# Task 3: Spectral Transformer PoC Plan

- [ ] Audit: Define the 1D FNO block (FFT -> Linear in Spectral Domain -> Inverse FFT).
- [ ] RED: Implement a test that ensures the FNO output shape matches input shape regardless of sequence length.
- [ ] GREEN: Implement the Spectral Transformer PoC in `spectral_transformer.py`.
    - 1D FNO layer with learnable complex weights.
    - Global pooling or linear readout.
- [ ] Verification: Resolution Invariance Test.
    - Train on sequence length =64$.
    - Evaluate on =128$ and =256$.
    - Compare with a standard Attention-based Transformer (which typically fails on out-of-distribution sequence lengths).
- [ ] Update Research Note: Document findings on "grid-free" learning.
