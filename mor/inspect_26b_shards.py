#!/usr/bin/env python3
"""Inspect 26B-A4B safetensors keys."""

from safetensors.torch import safe_open
import os

# Locate shards
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
snap = os.path.join(cache_dir, 'models--google--gemma-4-26B-A4B-it', 'snapshots', '7d4c97e54145f8ffd1a4dd1b4986a5015a517842')
shard1 = os.path.join(snap, 'model-00001-of-00002.safetensors')
shard2 = os.path.join(snap, 'model-00002-of-00002.safetensors')

print('Shard 1:', shard1, os.path.getsize(shard1) if os.path.exists(shard1) else 'missing')
print('Shard 2:', shard2, os.path.getsize(shard2) if os.path.exists(shard2) else 'missing')

# Open shard1
with safe_open(shard1, framework='pt', device='cpu') as f:
    keys = sorted(f.keys())
    print(f'Shard1 keys: {len(keys)}')
    for k in keys[:10]:
        print(f'  {k}')
    print('...')
    
    # Find language_model keys
    lm_keys = [k for k in keys if 'language_model' in k]
    for k in lm_keys[:5]:
        t = f.get_tensor(k)
        print(f'  {k}: {tuple(t.shape)}')
        
    # Count layers
    import re
    layers = set()
    for k in lm_keys:
        m = re.search(r'layers\.(\d+)\.', k)
        if m:
            layers.add(int(m.group(1)))
    print(f'Found layers: {sorted(layers)[:5]}...{sorted(layers)[-3:]} (total {len(layers)})')