#!/usr/bin/env python3
"""
Load a causal LM, optionally substitute selected linears (dense or native PGD), then
report NLL / perplexity (single or multi-prompt). Prefer ``compressed_inference_harness.py``
for full multi-arm + timing benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hidden_state_hooks import capture_post_layer_hidden_auto, hidden_drift_stats
from inference_eval import (
    collect_run_environment,
    load_model_and_tokenizer,
    load_prompts_for_eval,
    nll_metrics_many,
    peak_cuda_memory_mb,
    reset_cuda_peak_memory,
    resolve_device,
)
from pgd_hf_substitution import (
    get_decoder_with_layers,
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
DEFAULT_PROMPTS_PATH = _ROOT / "default_eval_prompts.txt"


def _parse_drift_layers(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PGD substitution + perplexity on snippet(s).")
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
    p.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Single prompt: entire file contents (use --prompts-file for one line per prompt).",
    )
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="One evaluation prompt per non-empty line.",
    )
    p.add_argument(
        "--use-default-prompt-set",
        action="store_true",
        help="Use bundled default_eval_prompts.txt when no --prompts-file.",
    )
    p.add_argument(
        "--drift-layers",
        type=str,
        default="",
        help="Comma-separated layer indices for hidden-state drift (substitution modes only).",
    )
    p.add_argument("--drift-prompt-index", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    text = args.text
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")

    prompts = load_prompts_for_eval(
        text=text,
        prompts_file=args.prompts_file,
        use_default_prompt_set=args.use_default_prompt_set,
        default_prompts_path=DEFAULT_PROMPTS_PATH,
    )

    layers = parse_layers_spec(args.layers)
    projections = parse_projections_spec(args.projections)
    drift_layers = _parse_drift_layers(args.drift_layers)

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

    n_dec = len(get_decoder_with_layers(model).layers)
    for li in drift_layers:
        if li < 0 or li >= n_dec:
            print(
                f"ERROR: drift layer {li} out of range [0, {n_dec}).",
                file=sys.stderr,
            )
            return 1

    dpi = max(0, min(args.drift_prompt_index, len(prompts) - 1))
    drift_prompt = prompts[dpi].strip() if prompts[dpi].strip() else prompts[dpi]
    enc_d = tokenizer(
        drift_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids_d = enc_d["input_ids"].to(dev)
    attn_d = enc_d.get("attention_mask")
    if attn_d is not None:
        attn_d = attn_d.to(dev)

    substituted: list = []
    sub_s = 0.0
    mode = "baseline"
    drift_report: Optional[Dict[str, Dict[str, float]]] = None

    if args.substitution == "dense":
        mode = "dense_pgd"
        print(
            f"Dense PGD substitution rank {args.rank} layers={layers} projections={projections} ...",
            flush=True,
        )
        h_ref = None
        if drift_layers:
            decoder = get_decoder_with_layers(model)
            h_ref = capture_post_layer_hidden_auto(
                decoder, drift_layers, input_ids=input_ids_d, attention_mask=attn_d
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
        if drift_layers and h_ref is not None:
            decoder = get_decoder_with_layers(model)
            h_mod = capture_post_layer_hidden_auto(
                decoder, drift_layers, input_ids=input_ids_d, attention_mask=attn_d
            )
            drift_report = {
                str(li): hidden_drift_stats(h_ref[li], h_mod[li]).as_dict()
                for li in drift_layers
            }
    elif args.substitution == "native":
        mode = "native_pgd"
        print(
            f"Native PGD substitution rank {args.rank} layers={layers} projections={projections} ...",
            flush=True,
        )
        h_ref = None
        if drift_layers:
            decoder = get_decoder_with_layers(model)
            h_ref = capture_post_layer_hidden_auto(
                decoder, drift_layers, input_ids=input_ids_d, attention_mask=attn_d
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
        if drift_layers and h_ref is not None:
            decoder = get_decoder_with_layers(model)
            h_mod = capture_post_layer_hidden_auto(
                decoder, drift_layers, input_ids=input_ids_d, attention_mask=attn_d
            )
            drift_report = {
                str(li): hidden_drift_stats(h_ref[li], h_mod[li]).as_dict()
                for li in drift_layers
            }

    reset_cuda_peak_memory()

    t2 = time.perf_counter()
    nll_block = nll_metrics_many(
        model, tokenizer, prompts, dev, max_length=args.max_length
    )
    fwd_s = time.perf_counter() - t2

    peak_mb = peak_cuda_memory_mb()

    fp0 = nll_block["per_prompt"][0] if nll_block["per_prompt"] else {}

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
        "prompt_stats": nll_block,
        "hidden_state_drift": drift_report,
        "drift_layers": drift_layers,
        "drift_prompt_index": dpi,
        "run_environment": collect_run_environment(),
        "num_prompts": len(prompts),
    }
    if fp0:
        result["loss_nats_per_token"] = fp0["loss_nats_per_token"]
        result["perplexity"] = fp0["perplexity"]
        result["num_tokens"] = fp0["num_tokens"]

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
