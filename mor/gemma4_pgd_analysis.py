#!/usr/bin/env python3
"""PGD analysis for Gemma 4 models."""

import os
import json
import time
import numpy as np
from safetensors.torch import safe_open
from typing import Dict, List, Tuple, Optional

# Add local path
import sys
sys.path.append(os.path.dirname(__file__))
from pgd_enrichment import pgd_decompose

# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------

MODEL_SPECS = {
    'E2B': {
        'model_id': 'google/gemma-4-E2B-it',
        'hidden_size': 1536,
        'intermediate_size': 6144,
        'num_layers': 35,
    },
    'E4B': {
        'model_id': 'google/gemma-4-E4B-it',
        'hidden_size': 2560,
        'intermediate_size': 10240,
        'num_layers': 42,
    },
    '26B-A4B': {
        'model_id': 'google/gemma-4-26B-A4B-it',
        'hidden_size': 2816,
        'intermediate_size': 2112,  # note: dense MLP intermediate size
        'num_layers': 30,
    },
}

# Cache directory
HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')

def locate_safetensors(model_name: str) -> List[str]:
    """Return list of safetensor file paths for the given model."""
    spec = MODEL_SPECS[model_name]
    # Map model ID to cache directory
    repo_dir = spec['model_id'].replace('/', '--')
    snap_dir = os.path.join(HF_CACHE, f'models--{repo_dir}', 'snapshots')
    if not os.path.exists(snap_dir):
        raise FileNotFoundError(f'Cache not found: {snap_dir}')
    # Get snapshot hash (first directory)
    snapshots = os.listdir(snap_dir)
    if not snapshots:
        raise FileNotFoundError(f'No snapshots in {snap_dir}')
    snapshot = snapshots[0]
    snapshot_path = os.path.join(snap_dir, snapshot)
    # Find safetensors files
    safetensors = []
    for f in os.listdir(snapshot_path):
        if f.endswith('.safetensors'):
            safetensors.append(os.path.join(snapshot_path, f))
    if not safetensors:
        raise FileNotFoundError(f'No safetensors in {snapshot_path}')
    return safetensors

# ----------------------------------------------------------------------
# Tensor loading
# ----------------------------------------------------------------------

def get_tensor_path(layer: int, projection: str) -> str:
    """Return tensor key path for given layer and projection.
    
    projection: one of 'q_proj', 'o_proj', 'gate_proj', 'down_proj'
    """
    if projection == 'q_proj':
        return f'model.language_model.layers.{layer}.self_attn.q_proj.weight'
    elif projection == 'o_proj':
        return f'model.language_model.layers.{layer}.self_attn.o_proj.weight'
    elif projection == 'gate_proj':
        return f'model.language_model.layers.{layer}.mlp.gate_proj.weight'
    elif projection == 'down_proj':
        return f'model.language_model.layers.{layer}.mlp.down_proj.weight'
    else:
        raise ValueError(f'Unknown projection: {projection}')

def load_weight(model_name: str, layer: int, projection: str) -> np.ndarray:
    """Load a weight tensor as numpy array (float32)."""
    safetensors = locate_safetensors(model_name)
    # For simplicity, try each shard until found
    path = get_tensor_path(layer, projection)
    for st in safetensors:
        with safe_open(st, framework='pt', device='cpu') as f:
            if path in f.keys():
                tensor = f.get_tensor(path).float().numpy().astype(np.float32)
                return tensor
    raise KeyError(f'Tensor {path} not found in any shard for {model_name}')

# ----------------------------------------------------------------------
# Spectral entropy utilities (copy from existing code)
# ----------------------------------------------------------------------

def compute_spectral_entropy(s: np.ndarray) -> float:
    """Compute spectral entropy from singular values."""
    s = s / np.sum(s)
    # Remove zeros for log
    mask = s > 1e-12
    return -np.sum(s[mask] * np.log(s[mask]))

def effective_rank(W: np.ndarray) -> float:
    """Effective rank (exponential of spectral entropy)."""
    s = np.linalg.svd(W, compute_uv=False)
    return np.exp(compute_spectral_entropy(s))

def spectral_entropy_of_matrix(W: np.ndarray) -> float:
    """Compute spectral entropy of matrix W."""
    s = np.linalg.svd(W, compute_uv=False)
    return compute_spectral_entropy(s)

# ----------------------------------------------------------------------
# PGD analysis
# ----------------------------------------------------------------------

def analyze_tensor(W: np.ndarray, ranks: List[int] = [8, 32, 128],
                   max_iters: int = 20, seed: int = 42) -> Dict:
    """Run PGD decomposition on W for each rank.
    
    Returns dict with keys: shape, H_full, eff_rank_full,
    and for each rank a dict with compression, rel_error, H_recon, H_gap_pct, walltime_s.
    """
    shape = W.shape
    H_full = spectral_entropy_of_matrix(W)
    r_full = effective_rank(W)
    
    results = {
        'shape': list(shape),
        'H_full': float(H_full),
        'eff_rank_full': float(r_full),
        'ranks': {}
    }
    
    for R in ranks:
        np.random.seed(seed + R)
        t_start = time.perf_counter()
        modes, residual = pgd_decompose(
            W, num_modes=R, max_fixed_point_iters=max_iters, seed=seed+R
        )
        elapsed = time.perf_counter() - t_start
        
        # Reconstruct
        reconstructed = np.zeros(shape, dtype=np.float32)
        for mode in modes:
            reconstructed += np.outer(mode[0], mode[1])
        
        rel_error = np.linalg.norm(W - reconstructed) / np.linalg.norm(W)
        
        # Compression ratio
        n, m = shape
        full_params = n * m
        mor_params = sum(sum(f.shape[0] for f in mode) for mode in modes)
        compression = full_params / mor_params
        
        # Spectral entropy of reconstruction
        H_recon = spectral_entropy_of_matrix(reconstructed)
        H_gap_pct = (H_full - H_recon) / H_full * 100
        
        results['ranks'][str(R)] = {
            'compression': float(compression),
            'rel_error': float(rel_error),
            'H_recon': float(H_recon),
            'H_gap_pct': float(H_gap_pct),
            'walltime_s': float(elapsed),
        }
    
    return results

# ----------------------------------------------------------------------
# Main experiment definition
# ----------------------------------------------------------------------

def run_experiment(config: Dict) -> Dict:
    """Run PGD analysis across models, layers, projections."""
    results = {}
    
    for model_name in config['models']:
        print(f'=== {model_name} ===')
        results[model_name] = {}
        
        spec = MODEL_SPECS[model_name]
        num_layers = spec['num_layers']
        layer_samples = config['layer_samples']
        if isinstance(layer_samples, str) and layer_samples == 'first_mid_last':
            layers = [0, num_layers // 2, num_layers - 1]
        else:
            layers = layer_samples
            
        for layer in layers:
            print(f'  Layer {layer}')
            layer_key = f'layer_{layer}'
            results[model_name][layer_key] = {}
            
            for proj in config['projections']:
                print(f'    {proj}', end=' ')
                try:
                    W = load_weight(model_name, layer, proj)
                except Exception as e:
                    print(f'ERROR: {e}')
                    continue
                    
                print(f'shape={W.shape}')
                analysis = analyze_tensor(W, ranks=config['ranks'])
                results[model_name][layer_key][proj] = analysis
                
        print()
    
    return results

# ----------------------------------------------------------------------
# Default configuration
# ----------------------------------------------------------------------

DEFAULT_CONFIG = {
    'models': ['E2B', 'E4B', '26B-A4B'],
    'layer_samples': 'first_mid_last',
    'projections': ['q_proj', 'o_proj', 'down_proj'],
    'ranks': [8, 32, 128],
    'max_iters': 20,
    'seed': 42,
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--output', type=str, default='pgd_results.json')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
    
    print('Starting PGD analysis across Gemma 4 models')
    results = run_experiment(config)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {args.output}')