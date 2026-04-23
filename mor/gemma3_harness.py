import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pgd_enrichment import pgd_decompose


def run_experiment():
    weights_path = os.environ.get("SPECTRAL_LLM_GEMMA_WEIGHTS", "")
    if not weights_path:
        print("Set SPECTRAL_LLM_GEMMA_WEIGHTS to a .npz file to run this harness.")
        return

    data = np.load(weights_path)
    target_keys = [
        "vision_encoder/siglip_encoder/Transformer/encoderblock_0/MlpBlock_0/Dense_0/kernel",
        "vision_encoder/siglip_encoder/Transformer/encoderblock_0/MlpBlock_0/Dense_1/kernel",
        "vision_encoder/siglip_encoder/Transformer/encoderblock_0/MultiHeadDotProductAttention_0/query/kernel",
        "vision_encoder/siglip_encoder/Transformer/encoderblock_0/MultiHeadDotProductAttention_0/key/kernel",
        "vision_encoder/siglip_encoder/Transformer/encoderblock_0/MultiHeadDotProductAttention_0/value/kernel",
        "vision_encoder/siglip_encoder/Transformer/encoderblock_0/MultiHeadDotProductAttention_0/out/kernel",
    ]

    results = []
    target_residual_ratio = 0.1
    max_modes = 128

    print(f"{'Layer Key':<80} | {'Shape':<15} | {'Modes to 90%':<12} | {'Res @ 128':<10} | {'Rand Res @ 128':<15}")
    print("-" * 145)

    for key in target_keys:
        if key not in data:
            print(f"Key {key} not found in weights file.")
            continue

        weight = data[key]
        if len(weight.shape) < 2:
            continue

        modes, res_norms = pgd_decompose(weight, num_modes=max_modes)
        initial_norm = res_norms[0]

        modes_to_target = "N/A"
        for i, norm in enumerate(res_norms):
            if norm / initial_norm <= target_residual_ratio:
                modes_to_target = i
                break

        final_res_ratio = res_norms[-1] / initial_norm

        random_weight = np.random.normal(size=weight.shape).astype(weight.dtype)
        _, rand_res_norms = pgd_decompose(random_weight, num_modes=max_modes)
        rand_initial_norm = rand_res_norms[0]
        rand_final_res_ratio = rand_res_norms[-1] / rand_initial_norm

        results.append(
            {
                "key": key,
                "shape": weight.shape,
                "modes_to_target": modes_to_target,
                "final_res_ratio": final_res_ratio,
                "rand_final_res_ratio": rand_final_res_ratio,
            }
        )

        print(
            f"{key[:78]:<80} | {str(weight.shape):<15} | {str(modes_to_target):<12} | {final_res_ratio:.4f}     | {rand_final_res_ratio:.4f}"
        )

    print("\n### Markdown Table Result\n")
    print("| Layer Key | Shape | Modes to 90% Reduc. | Real Res @ 128 | Random Res @ 128 |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['key']} | {r['shape']} | {r['modes_to_target']} | {r['final_res_ratio']:.4f} | {r['rand_final_res_ratio']:.4f} |")


if __name__ == "__main__":
    run_experiment()
