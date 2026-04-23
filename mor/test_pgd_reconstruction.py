import sys; sys.path.insert(0, '.')
from pgd_enrichment import pgd_decompose
import numpy as np

# 2D matrix test
np.random.seed(42)
W = np.random.randn(10, 20).astype(np.float32)

modes, residuals = pgd_decompose(W, num_modes=5, max_fixed_point_iters=20)
print(f'Number of modes: {len(modes)}')
print(f'First mode: {len(modes[0])} factors, shapes: {[f.shape for f in modes[0]]}')
print(f'First factor shape: {modes[0][0].shape}, second: {modes[0][1].shape}')

# Reconstruct
reconstructed = np.zeros((10,20), dtype=np.float32)
for mode in modes:
    reconstructed += np.outer(mode[0], mode[1])
rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)
print(f'Relative error: {rel_error}')
print(f'Residuals: {residuals}')
print('Test passed' if rel_error < 0.1 else 'High error')