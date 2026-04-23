import sys
sys.path.insert(0, '.')
from gemma4_pgd_analysis import load_weight, analyze_tensor
import time

model = 'E2B'
layer = 0
proj = 'down_proj'

print(f'Loading {model} layer {layer} {proj}...')
W = load_weight(model, layer, proj)
print(f'Shape: {W.shape}')
print('Running PGD...')
t0 = time.perf_counter()
results = analyze_tensor(W, ranks=[8, 32], max_iters=20)
t1 = time.perf_counter()
print(f'Analysis completed in {t1 - t0:.1f}s')
print(f'H_full: {results["H_full"]:.3f}')
print(f'eff_rank_full: {results["eff_rank_full"]:.1f}')
for R in ['8', '32']:
    r = results['ranks'][R]
    print(f'R={R}: compression={r["compression"]:.1f}x, error={r["rel_error"]:.4f}, walltime={r["walltime_s"]:.1f}s')