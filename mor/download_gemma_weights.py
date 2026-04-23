"""
Download Gemma 3 270M weights from HuggingFace and save as numpy arrays.
"""
import numpy as np
from pathlib import Path


def download_and_save_gemma_weights():
    """Download Gemma 3 270M weights and save as npz."""
    try:
        from transformers import AutoModelForMaskedLM, AutoConfig
        import torch
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please run: pip install transformers torch")
        return None

    model_name = "google/gemma-3-270m-it"

    print(f"Downloading {model_name}...")
    print("(This may take a few minutes on first run due to model download)")

    config = AutoConfig.from_pretrained(model_name)
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    model.eval()

    weights_dict = {}

    print("\nExtracting weights...")
    for name, param in model.named_parameters():
        if 'embed' in name.lower():
            continue

        print(f"  {name}: {param.shape}")
        weights_dict[name] = param.detach().cpu().numpy().astype(np.float32)

    output_dir = Path(__file__).parent
    output_path = output_dir / "gemma3_270m_weights.npz"

    print(f"\nSaving to {output_path}...")
    np.savez_compressed(output_path, **weights_dict)

    print(f"\nSaved {len(weights_dict)} tensors")
    print(f"Total parameters: {sum(w.size for w in weights_dict.values()):,}")

    print("\n--- Weight Summary ---")
    layer_weights = {}
    for name, weight in weights_dict.items():
        if 'layer' in name:
            layer_num = name.split('.layer')[1].split('.')[0] if '.layer' in name else name
            layer_weights[layer_num] = layer_weights.get(layer_num, 0) + weight.size

    for layer, params in sorted(layer_weights.items())[:5]:
        print(f"  Layer {layer}: {params:,} params")
    if len(layer_weights) > 5:
        print(f"  ... and {len(layer_weights) - 5} more layers")

    return str(output_path)


if __name__ == "__main__":
    path = download_and_save_gemma_weights()
    if path:
        print(f"\nWeights saved to: {path}")
