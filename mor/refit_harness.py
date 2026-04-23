import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pgd_enrichment import pgd_decompose

D_MODEL = 1024
NUM_LAYERS = 16
D_INTERMEDIATE = 4096
MAX_MODES = 128
TARGET_RESIDUAL_RATIO = 0.10


def generate_weight(shape):
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(scale=std, size=shape)


def run_harness():
    np.random.seed(42)

    projection_types = ["Q", "K", "V", "Gate", "Up", "Down"]
    results = {pt: {"modes": [], "residuals": []} for pt in projection_types}

    print("### Gemma 3 270M PGD Refitting Harness")
    print(f"- Target Residual: < {TARGET_RESIDUAL_RATIO * 100}%")
    print(f"- Max Modes: {MAX_MODES}")
    print(f"- Layers: {NUM_LAYERS}")
    print("-" * 40)

    for layer in range(NUM_LAYERS):
        layer_start = time.time()
        shapes = {
            "Q": (D_MODEL, D_MODEL),
            "K": (D_MODEL, D_MODEL),
            "V": (D_MODEL, D_MODEL),
            "Gate": (D_MODEL, D_INTERMEDIATE),
            "Up": (D_MODEL, D_INTERMEDIATE),
            "Down": (D_INTERMEDIATE, D_MODEL),
        }

        for pt, shape in shapes.items():
            weight = generate_weight(shape)
            _, residual_norms = pgd_decompose(weight, num_modes=MAX_MODES)

            initial_norm = residual_norms[0]
            modes_needed = MAX_MODES
            final_res_ratio = residual_norms[-1] / initial_norm

            for i, norm in enumerate(residual_norms):
                ratio = norm / initial_norm
                if ratio <= TARGET_RESIDUAL_RATIO:
                    modes_needed = i
                    final_res_ratio = ratio
                    break

            results[pt]["modes"].append(modes_needed)
            results[pt]["residuals"].append(final_res_ratio)

        layer_time = time.time() - layer_start
        print(f"Layer {layer + 1}/{NUM_LAYERS} completed in {layer_time:.2f}s")

    print("\n### Summary Results: Average Modes per Projection Type")
    print("| Projection Type | Avg Modes for 90% Compression | Avg Final Residual (%) |")
    print("| :--- | :---: | :---: |")

    for pt in projection_types:
        avg_modes = sum(results[pt]["modes"]) / NUM_LAYERS
        avg_res = (sum(results[pt]["residuals"]) / NUM_LAYERS) * 100
        print(f"| {pt} | {avg_modes:.1f} | {avg_res:.2f}% |")


if __name__ == "__main__":
    start_time = time.time()
    run_harness()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
