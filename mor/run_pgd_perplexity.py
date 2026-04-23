#!/usr/bin/env python3
"""
Load a causal LM, optionally substitute selected linears (dense or native PGD), then
report NLL / perplexity. Prefer ``compressed_inference_harness.py`` for multi-arm runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference_eval import (
    load_model_and_tokenizer,
    nll_metrics,
    peak_cuda_memory_mb,
    reset_cuda_peak_memory,
    resolve_device,
)
from pgd_hf_substitution import (
    parse_layers_spec,
    parse_projections_spec,
    substitute_selected_linears,
    substitute_selected_linears_native,
    validate_projections,
)

DEFAULT_SNIPPET = (
    "The goal is to measure whether low-rank PGD reconstructions preserve "
    "language-model behavior on a fixed prompt."
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PGD substitution + perplexity on a snippet.")
    p.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    p.add_argument("--rank", type=int, default=128)
    p.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma-separated layer indices (empty = baseline).",
    )
    p.add_argument("--projections", type=str, default="down_proj")
    p.add_argument(
        "--substitution",
        type=str,
        default="dense",
        choices=["none", "dense", "native"],
        help="none = baseline weights; dense = PGD reconstruction in nn.Linear; native = PGDLinear.",
    )
    p.add_argument("--pgd-iters", type=int, default=20, dest="pgd_iters")
    p.add_argument("--pgd-seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--text", type=str, default=DEFAULT_SNIPPET)
    p.add_argument("--text-file", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    text = args.text
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")

    layers = parse_layers_spec(args.layers)
    projections = parse_projections_spec(args.projections)
    if args.substitution != "none":
        if not layers:
            print("ERROR: --layers required when substitution is not none.", file=sys.stderr)
            return 1
        validate_projections(projections)

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    load_s = time.perf_counter() - t0
    dev = resolve_device(model, "cpu")

    substituted: list = []
    sub_s = 0.0
    mode = "baseline"
    if args.substitution == "dense":
        mode = "dense_pgd"
        print(
            f"Dense PGD substitution rank {args.rank} layers={layers} projections={projections} ...",
            flush=True,
        )
        t1 = time.perf_counter()
        substituted = substitute_selected_linears(
            model,
            layers,
            projections,
            args.rank,
            max_fixed_point_iters=args.pgd_iters,
            seed=args.pgd_seed,
        )
        sub_s = time.perf_counter() - t1
    elif args.substitution == "native":
        mode = "native_pgd"
        print(
            f"Native PGD substitution rank {args.rank} layers={layers} projections={projections} ...",
            flush=True,
        )
        t1 = time.perf_counter()
        substituted = substitute_selected_linears_native(
            model,
            layers,
            projections,
            args.rank,
            max_fixed_point_iters=args.pgd_iters,
            seed=args.pgd_seed,
        )
        sub_s = time.perf_counter() - t1

    reset_cuda_peak_memory()

    t2 = time.perf_counter()
    metrics = nll_metrics(
        model,
        tokenizer,
        text,
        dev,
        max_length=args.max_length,
    )
    fwd_s = time.perf_counter() - t2

    peak_mb = peak_cuda_memory_mb()

    result: Dict[str, Any] = {
        "mode": mode,
        "substitution": args.substitution,
        "model": args.model,
        "substituted": [{"layer": a, "proj": b, "shape": list(c)} for a, b, c in substituted],
        "rank": args.rank if args.substitution != "none" else 0,
        "layers": layers,
        "projections": projections,
        "load_time_s": load_s,
        "substitution_time_s": sub_s,
        "forward_time_s": fwd_s,
        "peak_gpu_memory_mb": peak_mb,
        **metrics,
    }

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
