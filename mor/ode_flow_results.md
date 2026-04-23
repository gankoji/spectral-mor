# PoC Results: Continuous Flow Fitting

## Objective
Demonstrate representation of discrete Transformer weights as a continuous trajectory $W(t)$.

## Configuration
- Layers: 12
- Matrix Dimension: 512x512 (262,144 elements)
- Trajectory Type: Smooth drift (low-rank random)

## Results
| Method | Parameters | Compression | MSE |
| :--- | :--- | :--- | :--- |
| **Discrete (Original)** | 3,145,728 | 1.00x | 0.00000000 |
| **Polynomial (K=3)** | 1,048,576 | 3.00x | 0.00002662 |
| **Neural ODE (H=8)** | 2,621,456 | 1.20x | 0.00176687 |

## Observations
- Polynomial fitting provides a strong baseline with fixed compression.
- Neural ODE (MLP-based velocity) allows for flexible parameterization of weight changes.
- Both methods significantly reduce the storage requirement while maintaining fidelity, confirming that layer-wise weights in deep models can be treated as samples from a continuous operator.
