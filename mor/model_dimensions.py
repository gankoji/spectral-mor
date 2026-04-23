#!/usr/bin/env python3
"""Extract dimensions of Gemma 4 models."""

from transformers import AutoConfig
import json

models = {
    'E2B': 'google/gemma-4-E2B-it',
    'E4B': 'google/gemma-4-E4B-it',
    '26B-A4B': 'google/gemma-4-26B-A4B-it',
}

def get_dimensions(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tc = cfg.text_config
    return {
        'hidden_size': tc.hidden_size,
        'intermediate_size': tc.intermediate_size,
        'num_hidden_layers': tc.num_hidden_layers,
        'num_attention_heads': tc.num_attention_heads,
        'num_key_value_heads': tc.num_key_value_heads,
        'head_dim': tc.head_dim,
        'vocab_size': tc.vocab_size,
    }

if __name__ == '__main__':
    dims = {}
    for name, model_id in models.items():
        print(f'Fetching {name} ({model_id})...')
        dims[name] = get_dimensions(model_id)
        print(f'  hidden={dims[name]["hidden_size"]}, int={dims[name]["intermediate_size"]}, layers={dims[name]["num_hidden_layers"]}')
        print(f'  heads={dims[name]["num_attention_heads"]}, kv_heads={dims[name]["num_key_value_heads"]}')
        print()
    
    with open('model_dimensions.json', 'w') as f:
        json.dump(dims, f, indent=2)
    print('Saved to model_dimensions.json')