from transformers import AutoConfig
import sys

model_id = 'google/gemma-4-26B-A4B-it'
print(f"Loading config for {model_id}...")
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tc = cfg.text_config
print(f"hidden_size: {tc.hidden_size}")
print(f"intermediate_size: {tc.intermediate_size}")
print(f"num_hidden_layers: {tc.num_hidden_layers}")
print(f"num_attention_heads: {tc.num_attention_heads}")
print(f"num_key_value_heads: {tc.num_key_value_heads}")
print(f"head_dim: {tc.head_dim}")
print(f"vocab_size: {tc.vocab_size}")

# Estimate shapes
h = tc.hidden_size
i = tc.intermediate_size
l = tc.num_hidden_layers
print(f"\nEstimated projection shapes:")
print(f"  q_proj: ({h}, {h})")
print(f"  k_proj: ({tc.head_dim * tc.num_key_value_heads}, {h})")
print(f"  v_proj: same as k_proj")
print(f"  o_proj: ({h}, {h})")
print(f"  gate/up_proj: ({i}, {h})")
print(f"  down_proj: ({h}, {i})")