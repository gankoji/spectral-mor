"""
Download small open LLM weights from HuggingFace and save as numpy arrays.
Uses smaller models that don't require gated access.
"""
import numpy as np
from pathlib import Path


def download_small_llm_weights():
    """Download a small open LLM and save as npz."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        import torch
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please run: pip install transformers torch")
        return None

    models_to_try = [
        ("openai-community/gpt2", "GPT-2 (124M params)"),
        ("distilbert/distilgpt2", "DistilGPT-2 (82M params)"),
        ("EleutherAI/gpt-neo-125m", "GPT-Neo 125M"),
        ("facebook/opt-125m", "OPT 125M"),
    ]

    for model_name, model_desc in models_to_try:
        print(f"\n{'=' * 60}")
        print(f"Trying: {model_desc} ({model_name})")
        print('=' * 60)

        try:
            print("Downloading...")
            config = AutoConfig.from_pretrained(model_name)
            print(
                f"Config: hidden_size={getattr(config, 'hidden_size', 'N/A')}, "
                f"num_layers={getattr(config, 'num_hidden_layers', 'N/A')}"
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            model.eval()

            weights_dict = {}
            total_params = 0

            print("\nExtracting weights...")
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in ["embed", "wte", "wpe"]):
                    continue

                print(f"  {name}: {param.shape}")
                weights_dict[name] = param.detach().cpu().numpy().astype(np.float32)
                total_params += param.numel()

            print(f"\nExtracted {len(weights_dict)} weight tensors")
            print(f"Total parameters: {total_params:,}")

            output_dir = Path(__file__).parent
            safe_name = model_name.replace('/', '_')
            output_path = output_dir / f"{safe_name}_weights.npz"

            print(f"\nSaving to {output_path}...")
            np.savez_compressed(output_path, **weights_dict)

            print(f"Saved {len(weights_dict)} tensors")
            return str(output_path)

        except Exception as e:
            print(f"Failed: {e}")
            continue

    print("\nCould not download any models.")
    return None


def download_gemma3_1b():
    """Try to download Gemma 3 1B if available."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        import torch
    except ImportError:
        print("Missing dependencies")
        return None

    model_name = "google/gemma-3-1b-it"

    print(f"\n{'=' * 60}")
    print(f"Trying: Gemma 3 1B ({model_name})")
    print('=' * 60)

    try:
        print("Downloading...")
        config = AutoConfig.from_pretrained(model_name)
        print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.eval()

        weights_dict = {}
        total_params = 0

        print("\nExtracting weights...")
        for name, param in model.named_parameters():
            if any(x in name.lower() for x in ["embed", "wte", "wpe"]):
                continue

            print(f"  {name}: {param.shape}")
            weights_dict[name] = param.detach().cpu().numpy().astype(np.float32)
            total_params += param.numel()

        output_dir = Path(__file__).parent
        output_path = output_dir / "gemma3_1b_weights.npz"

        print(f"\nSaving {len(weights_dict)} tensors ({total_params:,} params)...")
        np.savez_compressed(output_path, **weights_dict)

        return str(output_path)

    except Exception as e:
        print(f"Failed: {e}")
        return None


if __name__ == "__main__":
    path = download_small_llm_weights()
    if path:
        print(f"\nWeights saved to: {path}")
    else:
        print("\nTrying Gemma 3 1B...")
        path = download_gemma3_1b()
        if path:
            print(f"\nWeights saved to: {path}")
