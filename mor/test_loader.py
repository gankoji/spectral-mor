#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from gemma4_pgd_analysis import locate_safetensors, load_weight

for model in ['E2B', 'E4B', '26B-A4B']:
    print(f'=== {model} ===')
    try:
        paths = locate_safetensors(model)
        print(f'  Found {len(paths)} safetensors:')
        for p in paths:
            print(f'    {p}')
        # Try loading q_proj layer 0
        W = load_weight(model, 0, 'q_proj')
        print(f'  q_proj shape: {W.shape}')
    except Exception as e:
        print(f'  ERROR: {e}')