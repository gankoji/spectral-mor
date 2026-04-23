#!/usr/bin/env python3
"""Quick PGD summary across Gemma 4 models."""

import sys
import json
import time
import numpy as np
sys.path.insert(0, '.')
from pgd_enrichment import pgd_decompose

# Reuse helper functions
def compute_spectral_entropy(s):
    s = s / np.sum(s)
    mask = s > 1e-12
    return -np.sum(s[mask] * np.log(s[mask]))

def spectral_entropy_of_matrix(W):
    s = np.linalg.svd(W, compute_uv=False)
    return compute_spectral_entropy(s)

def effective_rank(W):
    s = np.linalg.svd(W, compute_uv=False)
    return np.exp(compute_spectral_entropy(s))

def power_law_fit(s):
    """Fit exponent alpha to singular values s[i] ∝ (i+1)^(-alpha)."""
    from scipy.stats import linregress
    x = np.log(np.arange(1, len(s) + 1))
    y = np.log(s)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return -slope, r_value**2

# Load weight function (simplified)
from safetensors.torch import safe_open
import os

def load_weight(model_name, layer, projection):
    cache = os.path.expanduser('~/.cache/huggingface/hub')
    if model_name == 'E2B':
        repo = 'models--google--gemma-4-E2B-it'
        snap = 'b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf'
        shard = 'model.safetensors'
    elif model_name == 'E4B':
        repo = 'models--google--gemma-4-E4B-it'
        snap = '83df0a889143b1dbfc61b591bbc639540fd9ce4c'
        shard = 'model.safetensors'
    elif model_name == '26B-A4B':
        repo = 'models--google--gemma-4-26B-A4B-it'
        snap = '7d4c97e54145f8ffd1a4dd1b4986a5015a517842'
        shard = 'model-00001-of-00002.safetensors'
    else:
        raise ValueError(f'Unknown model {model_name}')
    
    path = os.path.join(cache, repo, 'snapshots', snap, shard)
    key = f'model.language_model.layers.{layer}.{projection}.weight'
    with safe_open(path, framework='pt', device='cpu') as f:
        return f.get_tensor(key).float().numpy().astype(np.float32)

def analyze(W, R=8):
    shape = W.shape
    H = spectral_entropy_of_matrix(W)
    r_eff = effective_rank(W)
    
    # Power-law fit
    s = np.linalg.svd(W, compute_uv=False)
    alpha, r2 = power_law_fit(s)
    
    # PGD
    t0 = time.perf_counter()
    modes, residuals = pgd_decompose(W, num_modes=R, max_fixed_point_iters=20, seed=42)
    elapsed = time.perf_counter() - t0
    
    # Reconstruct
    recon = np.zeros(shape, dtype=np.float32)
    for mode in modes:
        recon += np.outer(mode[0], mode[1])
    
    rel_error = np.linalg.norm(W - recon) / np.linalg.norm(W)
    explained = 1 - rel_error**2
    
    # Compression
    n, m = shape
    full = n * m
    mor = sum(sum(f.shape[0] for f in mode) for mode in modes)
    compression = full / mor
    
    return {
        'shape': shape,
        'H': float(H),
        'r_eff': float(r_eff),
        'alpha': float(alpha),
        'r2': float(r2),
        'compression': float(compression),
        'explained': float(explained),
        'walltime': float(elapsed),
    }

def main():
    models = ['E2B', 'E4B', '26B-A4B']
    layer = 0
    projections = ['q_proj', 'down_proj']
    R = 8
    
    results = {}
    
    for model in models:
        print(f'=== {model} ===')
        results[model] = {}
        for proj in projections:
            print(f'  {proj}...', end='', flush=True)
            try:
                W = load_weight(model, layer, proj)
                print(f' shape {W.shape}', end='')
                res = analyze(W, R)
                print(f' done ({res["walltime"]:.1f}s)')
                results[model][proj] = res
            except Exception as e:
                print(f' ERROR: {e}')
                continue
    
    # Output table
    print('\n--- Results (R=8) ---')
    print('Model | Projection | Shape | H | r_eff | α | R² | Compression | Explained | Walltime')
    for model in models:
        for proj in projections:
            if proj not in results[model]:
                continue
            r = results[model][proj]
            print(f'{model} | {proj} | {r["shape"][0]}x{r["shape"][1]} | {r["H"]:.2f} | {r["r_eff"]:.0f} | {r["alpha"]:.3f} | {r["r2"]:.3f} | {r["compression"]:.1f}x | {r["explained"]:.3f} | {r["walltime"]:.1f}s')
    
    # Save JSON
    with open('pgd_quick_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to pgd_quick_summary.json')

if __name__ == '__main__':
    main()