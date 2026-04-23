from safetensors.torch import safe_open
import sys

path = "/Users/jacbaile/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/7d4c97e54145f8ffd1a4dd1b4986a5015a517842/model-00001-of-00002.safetensors"
print(f"Opening {path}")
with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    # find a q_proj weight
    for k in keys[:20]:
        print(k)
    # look for language_model
    lm_keys = [k for k in keys if 'language_model' in k]
    print(f"Language model keys: {len(lm_keys)}")
    if lm_keys:
        sample = lm_keys[0]
        t = f.get_tensor(sample)
        print(f"Sample tensor {sample}: shape {t.shape}, dtype {t.dtype}")
        # pick q_proj layer 0
        for k in lm_keys:
            if 'q_proj' in k and 'layers.0.' in k:
                W = f.get_tensor(k).float().numpy().astype('float32')
                print(f"Found q_proj layer 0: shape {W.shape}")
                import numpy as np
                print(f"Norm: {np.linalg.norm(W):.2f}")
                break