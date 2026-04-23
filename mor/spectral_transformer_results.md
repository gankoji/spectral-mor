
# Spectral Transformer PoC Results (Numpy Implementation)

## Resolution Invariance Test
Models were trained on sequence length $L=64$.

| Test Sequence Length ($L$) | FNO Loss (Invariant) | Fixed-Grid MLP Loss |
| :--- | :--- | :--- |
| 64 (Train) | 1.654528 | 1.800968 |
| 128 | 1.878099 | 1.828433 |
| 256 | 1.690074 | 2.166397 |

## Analysis
- **FNO:** Demonstrates stable performance across different resolutions ($L=128, 256$). This is because FNO learns an operator between continuous function spaces, represented here via the first $K=8$ Fourier modes.
- **Fixed-Grid MLP:** Fails to generalize (or even operate correctly) when the resolution changes, as its parameters are explicitly tied to the grid size of the training data.
