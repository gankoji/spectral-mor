#!/usr/bin/env python3
"""Print copy-paste shell examples from ``experiment_matrix.json`` (no jobs executed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> int:
    path = _ROOT / "experiment_matrix.json"
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    model = "google/gemma-4-E2B-it"
    print("# Examples only — adjust paths, device, and HF cache as needed.\n")
    for run in data.get("runs", []):
        rid = run.get("run_id", "?")
        print(f"## {rid}")
        mode = run.get("mode", "")
        if mode == "fidelity_only":
            layers = run.get("layers", [])
            ranks = run.get("ranks", [])
            print(
                f"export SPECTRAL_MOR_SAFETENSORS=/path/to/model.safetensors\n"
                f"python pgd_fidelity_harness.py \\\n"
                f"  --layers {','.join(map(str, layers))} \\\n"
                f"  --projections {','.join(run.get('projections', []))} \\\n"
                f"  --ranks {','.join(map(str, ranks))} \\\n"
                f"  --svd --output-csv results_{rid}.csv\n"
            )
            continue
        if mode in ("dense_pgd", "native_pgd"):
            layers = ",".join(map(str, run.get("layers", [])))
            projs = ",".join(run.get("projections", []))
            r = run.get("rank", 128)
            print(
                f"python compressed_inference_harness.py \\\n"
                f"  --model {model} --mode {mode} \\\n"
                f"  --layers {layers} --projections {projs} --rank {r} \\\n"
                f"  --use-default-prompt-set \\\n"
                f"  --drift-layers {layers} \\\n"
                f"  --prefill-repeats 5 --max-new-tokens 32 \\\n"
                f"  --output-json results_{rid}.json\n"
            )
            continue
        if mode == "benchmark":
            print(
                f"python pgd_inference_benchmark.py \\\n"
                f"  --model {model} --mode all \\\n"
                f"  --layers 0,17,34 --projections down_proj --rank 128 \\\n"
                f"  --use-default-prompt-set --drift-layers 17,34 \\\n"
                f"  --prefill-repeats 10 --max-new-tokens 64 \\\n"
                f"  --output-json benchmark_{rid}.json\n"
            )
            continue
        print(f"# (no template for mode={mode})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
