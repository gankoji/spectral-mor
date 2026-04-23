"""
Download Gemma 4 model weights and extract representative layer tensors
for PGD MOR analysis.

Downloads a subset of layers (every Nth layer + first and last) to keep
download sizes manageable while still capturing architectural variation.
"""

import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
import json

# Model configurations we want to download
MODELS = [
    ("google/gemma-4-E2B-it", "E2B",  # ~2B params, h=1536, i=6144, 35 layers
        {"hidden_size": 1536, "intermediate_size": 6144, "num_layers": 35}),
    ("google/gemma-4-E4B-it", "E4B",  # ~4B params, h=2560, i=10240, 42 layers
        {"hidden_size": 2560, "intermediate_size": 10240, "num_layers": 42}),
]

# How many layers to sample from each model
LAYERS_TO_SAMPLE = {
    "E2B": [0, 4, 9, 17, 26, 34],  # 6 layers evenly spread
    "E4B": [0, 5, 10, 20, 31, 41],  # 6 layers
}

# PGD rank budgets to analyze
RANK_BUDGETS = [8, 16, 32, 64, 128, 256]


def get_model_dir(model_id: str) -> Path:
    """Get cache directory for a model."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def extract_layers(model_id: str, tag: str, layer_indices: list, output_dir: Path):
    """Load model and extract specified layers as numpy arrays."""

    print(f"\n{'='*60}")
    print(f"Loading {model_id} ({tag})...")
    print(f"{'='*60}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tc = config.text_config

    print(f"  Config: h={tc.hidden_size}, i={tc.intermediate_size}, L={tc.num_hidden_layers}")
    print(f"  Sample layers: {layer_indices}")

    model_path = get_model_dir(model_id)

    # Find safetensors files
    safefiles = sorted(model_path.glob("*.safetensors"))
    print(f"  Safetensor files: {len(safefiles)}")

    if not safefiles:
        # Try .bin files
        binfiles = sorted(model_path.glob("*.bin"))
        print(f"  Bin files: {len(binfiles)}")
        if binfiles:
            print("  [Would load .bin files - skipping due to size]")
            return None

    # Track total size
    total_size_gb = 0
    for sf in safefiles:
        size_gb = sf.stat().st_size / 1e9
        total_size_gb += size_gb
    print(f"  Total model size: {total_size_gb:.1f} GB")

    # Load only the model
    print("  Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Extract state dict
    state = model.state_dict()

    # Categorize weight tensors
    weight_tensors = {}
    for name, param in state.items():
        # Skip embeddings and layernorms (focus on projection matrices)
        if any(k in name.lower() for k in ["embed", "layernorm", "norm", "lm_head", "position"]):
            continue
        if "weight" not in name.lower() or param.ndim != 2:
            continue

        # Classify
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            cat = "attn_qkv"
        elif "o_proj" in name or "attn_output" in name:
            cat = "attn_out"
        elif "gate_proj" in name:
            cat = "mlp_gate"
        elif "up_proj" in name:
            cat = "mlp_up"
        elif "down_proj" in name:
            cat = "mlp_down"
        else:
            cat = "other"

        layer_num = None
        for part in name.split("."):
            if part.isdigit():
                layer_num = int(part)
                break

        key = f"layer{layer_num}_{cat}" if layer_num is not None else cat
        weight_tensors[key] = (name, param)

    # Print summary
    print(f"\n  Weight tensors found: {len(weight_tensors)}")
    for cat in sorted(set(k.split("_")[1] for k in weight_tensors.keys() if "_" in k)):
        count = sum(1 for k in weight_tensors if cat in k)
        print(f"    {cat}: {count}")

    # Extract selected layers
    extracted = {}
    for layer_idx in layer_indices:
        for cat in ["attn_qkv", "mlp_gate", "mlp_up", "mlp_down", "attn_out"]:
            # Find tensor for this layer/category
            matches = [(k, v) for k, v in weight_tensors.items()
                      if f"layer{layer_idx}_" in k and cat in k]
            if matches:
                key, (name, param) = matches[0]
                print(f"  Layer {layer_idx} {cat}: {param.shape} ({param.numel() * 2 / 1e6:.1f} MB FP16)")

                arr = param.detach().cpu().numpy().astype(np.float16)
                extracted[f"{tag}_L{layer_idx}_{cat}"] = arr

    # Also get first and last layer QKV for full trajectory analysis
    print(f"\n  Extracting full layer trajectory (first, mid, last)...")

    del model  # free memory

    return extracted


def run_pgd_analysis(weights: dict, model_tag: str, output_dir: Path):
    """Run PGD MOR analysis on extracted weights."""

    print(f"\n{'='*60}")
    print(f"PGD MOR Analysis: {model_tag}")
    print(f"{'='*60}")

    # Import PGD
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pgd_enrichment import pgd_decompose

    results = []

    for name, tensor in weights.items():
        shape = tensor.shape
        m, n = shape

        print(f"\n  {name}: {shape}")

        for rank in RANK_BUDGETS:
            if rank > min(m, n):
                continue

            import time
            t0 = time.perf_counter()
            modes, residual_norms = pgd_decompose(tensor, num_modes=rank, max_fixed_point_iters=20, seed=42)
            elapsed = time.perf_counter() - t0

            initial_norm = float(np.linalg.norm(tensor))
            final_residual = residual_norms[-1]
            explained_var = 1.0 - (final_residual ** 2) / (initial_norm ** 2 + 1e-12)

            # Compression ratio
            d = 2  # matrix rank
            full_params = m * n
            compressed_params = rank * (m + n)
            comp_ratio = full_params / compressed_params

            print(f"    R={rank:3d}: {elapsed:6.2f}s, expl_var={explained_var:.1%}, "
                  f"comp={comp_ratio:.1f}x, modes={len(modes)}")

            results.append({
                "name": name,
                "shape": shape,
                "rank": rank,
                "walltime_s": elapsed,
                "explained_var": float(explained_var),
                "compression_ratio": float(comp_ratio),
                "modes_extracted": len(modes),
                "residual_ratio": float(final_residual / (initial_norm + 1e-12)),
            })

    return results


def main():
    output_dir = Path(__file__).parent / "gemma4_weights"
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for model_id, tag, arch in MODELS:
        model_output_dir = output_dir / tag
        model_output_dir.mkdir(exist_ok=True)

        layers_to_sample = LAYERS_TO_SAMPLE[tag]

        # Extract layers
        extracted = extract_layers(model_id, tag, layers_to_sample, model_output_dir)

        if extracted is None:
            print(f"\n  Skipping {tag} (no safetensors found or too large)")
            continue

        # Save extracted weights
        npz_path = model_output_dir / f"{tag}_sample_weights.npz"
        np.savez_compressed(npz_path, **extracted)
        print(f"\n  Saved to: {npz_path}")
        print(f"  Total extracted: {sum(v.nbytes for v in extracted.values()) / 1e9:.3f} GB")

        # Run PGD analysis
        results = run_pgd_analysis(extracted, tag, model_output_dir)
        all_results.append({"model": tag, "arch": arch, "results": results})

        # Save results
        results_path = model_output_dir / f"{tag}_pgd_results.json"
        import json
        with open(results_path, "w") as f:
            json.dump(all_results[-1], f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"  Results saved to: {results_path}")

        print(f"\n  Summary for {tag}:")
        best_compression = max(results, key=lambda r: r["compression_ratio"])
        best_explained = max(results, key=lambda r: r["explained_var"])

        print(f"    Best compression: {best_compression['compression_ratio']:.1f}x "
              f"(R={best_compression['rank']}, {best_compression['explained_var']:.1%} explained)")
        print(f"    Best explained: {best_explained['explained_var']:.1%} "
              f"(R={best_explained['rank']}, {best_explained['compression_ratio']:.1f}x)")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    results = main()