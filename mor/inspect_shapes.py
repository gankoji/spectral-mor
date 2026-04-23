#!/usr/bin/env python3
"""Inspect tensor shapes across Gemma 4 models."""

from safetensors.torch import safe_open
import os
import json

MODEL_PATHS = {
    'E2B': '~/slipbox-spectral-llm/inbox/agents/spectral-llm/mor/gemma4_weights/E2B/model.safetensors',
    'E4B': '~/slipbox-spectral-llm/inbox/agents/spectral-llm/mor/gemma4_weights/E4B/model.safetensors',
    '26B-A4B': '~/slipbox-spectral-llm/inbox/agents/spectral-llm/mor/gemma4_weights/26B-A4B/model-00001-of-00002.safetensors',
}

def expand(path):
    return os.path.expanduser(path)

def inspect_model(name, path):
    """Return dict of tensor shapes for key layers."""
    if not os.path.exists(expand(path)):
        print(f'  {name}: missing {path}')
        return None
    
    with safe_open(expand(path), framework='pt', device='cpu') as f:
        keys = sorted(f.keys())
        # Select tensors: q_proj, o_proj, gate_proj, down_proj for layers 0, N/2, N-1
        selected = {}
        for k in keys:
            if 'language_model' not in k:
                continue
            if 'layers.' not in k:
                continue
            layer = int(k.split('layers.')[1].split('.')[0])
            if layer not in (0, 15, 34) and layer not in (0, 21, 41) and layer not in (0, 15, 29):
                continue
            if any(proj in k for proj in ['q_proj', 'o_proj', 'gate_proj', 'down_proj']):
                selected[k] = tuple(f.get_tensor(k).shape)
                if len(selected) >= 20:
                    break
        return selected

if __name__ == '__main__':
    for name, path in MODEL_PATHS.items():
        print(f'=== {name} ===')
        shapes = inspect_model(name, path)
        if shapes:
            for k, shape in sorted(shapes.items()):
                print(f'  {k}: {shape}')
        else:
            print('  Not found')
        print()