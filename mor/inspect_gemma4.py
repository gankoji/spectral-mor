"""
Extract specific layers from Gemma 4 E2B safetensor file directly,
without loading the full model.
"""

import numpy as np
from safetensors.torch import load_file
import torch

safefile = "/Users/jacbaile/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf/model.safetensors"

print("Loading safetensor file...")
tensors = load_file(safefile, device="cpu")
print(f"Loaded {len(tensors)} tensors")
print(f"Total size: {sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9:.2f} GB")

# Categorize
layers = {}
for name, param in tensors.items():
    if "layer" not in name or param.ndim != 2:
        continue
    parts = name.split(".")
    layer_num = None
    for p in parts:
        if p.isdigit():
            layer_num = int(p)
            break
    if layer_num is None:
        continue
    if layer_num not in layers:
        layers[layer_num] = {}
    layers[layer_num][name] = param

# Sort layer numbers
layer_nums = sorted(layers.keys())
print(f"\nLayers found: {len(layer_nums)} ({layer_nums[0]}-{layer_nums[-1]})")

# Show all tensor names and shapes for layer 0
print("\nLayer 0 tensor names:")
for name, param in sorted(layers[0].items()):
    print(f"  {name}: {param.shape} ({param.numel() * 2 / 1e6:.1f} MB FP16)")

# Sample: layer 0, 9, 17, 26, 34
sample_layers = [0, 9, 17, 26, 34]

print("\n--- Sampling layers ---")
for ln in sample_layers:
    if ln not in layers:
        print(f"Layer {ln}: NOT FOUND")
        continue
    layer_tensors = layers[ln]
    print(f"\nLayer {ln} ({len(layer_tensors)} tensors):")
    for name, param in sorted(layer_tensors.items()):
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"]):
            mbps = param.numel() * 2 / 1e6
            print(f"  {name}: {param.shape} ({mbps:.1f} MB)")
