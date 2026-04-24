#!/usr/bin/env python3
"""
Unified evaluation harness: baseline, dense PGD reconstruction, or native factorized
``PGDLinear`` (Phase C), plus optional prefill/decode timing (Phase D), multi-prompt
statistics, and hidden-state drift vs pre-substitution activations.
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
    time_forward_pass,
    time_generate,
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


def run_single_arm(
    mode: str,
    *,
    model_id: str,
    device: str,
    torch_dtype: str,
    layers: List[int],
    projections: List[str],
    rank: int,
    pgd_iters: int,
    pgd_seed: int,
    prompts: List[str],
    max_length: int,
    prefill_warmup: int,
    prefill_repeats: int,
    max_new_tokens: int,
    decode_repeats: int,
    drift_layers: List[int],
    drift_prompt_index: int,
) -> Dict[str, Any]:
    t_load_0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    load_time = time.perf_counter() - t_load_0
    dev = resolve_device(model, "cpu")

    n_dec = len(get_decoder_with_layers(model).layers)
    for li in drift_layers:
        if li < 0 or li >= n_dec:
            raise ValueError(
                f"drift layer {li} out of range for model with {n_dec} decoder layers"
            )

    first_prompt = prompts[0].strip() if prompts[0].strip() else prompts[0]
    dpi = max(0, min(drift_prompt_index, len(prompts) - 1))
    drift_prompt = prompts[dpi].strip() if prompts[dpi].strip() else prompts[dpi]
    enc_d = tokenizer(
        drift_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids_d = enc_d["input_ids"].to(dev)
    attn_d = enc_d.get("attention_mask")
    if attn_d is not None:
        attn_d = attn_d.to(dev)

    sub_time = 0.0
    substituted: List[Any] = []
    drift_report: Optional[Dict[str, Dict[str, float]]] = None

    if mode == "baseline":
        pass
    elif mode in ("dense_pgd", "native_pgd"):
        h_ref: Optional[Dict[int, torch.Tensor]] = None
        if drift_layers:
            decoder = get_decoder_with_layers(model)
            h_ref = capture_post_layer_hidden_auto(
                decoder,
                drift_layers,
                input_ids=input_ids_d,
                attention_mask=attn_d,
            )
        t_sub = time.perf_counter()
        if mode == "dense_pgd":
            substituted = substitute_selected_linears(
                model,
                layers,
                projections,
                rank,
                max_fixed_point_iters=pgd_iters,
                seed=pgd_seed,
            )
        else:
            substituted = substitute_selected_linears_native(
                model,
                layers,
                projections,
                rank,
                max_fixed_point_iters=pgd_iters,
                seed=pgd_seed,
            )
        sub_time = time.perf_counter() - t_sub
        if drift_layers and h_ref is not None:
            decoder = get_decoder_with_layers(model)
            h_mod = capture_post_layer_hidden_auto(
                decoder,
                drift_layers,
                input_ids=input_ids_d,
                attention_mask=attn_d,
            )
            drift_report = {
                str(li): hidden_drift_stats(h_ref[li], h_mod[li]).as_dict()
                for li in drift_layers
            }
    else:
        raise ValueError(f"unknown mode {mode!r}")

    reset_cuda_peak_memory()

    nll_block = nll_metrics_many(
        model, tokenizer, prompts, dev, max_length=max_length
    )

    enc = tokenizer(
        first_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(dev)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(dev)

    prefill_mean, prefill_std = time_forward_pass(
        model,
        input_ids,
        attn,
        warmup=prefill_warmup,
        repeats=prefill_repeats,
    )

    decode_s: Optional[float] = None
    decode_new_tokens: Optional[int] = None
    decode_toks_per_s: Optional[float] = None
    if max_new_tokens > 0:
        dg_s, n_new = time_generate(
            model,
            tokenizer,
            first_prompt,
            dev,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_length,
            warmup=0,
            repeats=decode_repeats,
        )
        decode_s = dg_s
        decode_new_tokens = n_new
        if decode_s and decode_s > 0 and decode_new_tokens:
            decode_toks_per_s = float(decode_new_tokens) / decode_s

    peak = peak_cuda_memory_mb()

    fp0 = nll_block["per_prompt"][0] if nll_block["per_prompt"] else {}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out: Dict[str, Any] = {
        "mode": mode,
        "model": model_id,
        "substituted": [{"layer": a, "proj": b, "shape": list(c)} for a, b, c in substituted],
        "rank": rank if mode != "baseline" else 0,
        "layers": layers,
        "projections": projections,
        "load_time_s": load_time,
        "substitution_time_s": sub_time,
        "prefill_mean_s": prefill_mean,
        "prefill_std_s": prefill_std,
        "decode_wall_s_per_run": decode_s,
        "decode_new_tokens_per_run": decode_new_tokens,
        "decode_tokens_per_s": decode_toks_per_s,
        "peak_gpu_memory_mb": peak,
        "prompt_stats": nll_block,
        "hidden_state_drift": drift_report,
        "drift_layers": drift_layers,
        "drift_prompt_index": dpi,
    }
    if fp0:
        out["loss_nats_per_token"] = fp0["loss_nats_per_token"]
        out["perplexity"] = fp0["perplexity"]
        out["num_tokens"] = fp0["num_tokens"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compressed inference harness (baseline / dense_pgd / native_pgd / all)."
    )
    p.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "dense_pgd", "native_pgd", "all"],
    )
    p.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument(
        "--safetensors-path",
        type=Path,
        default=None,
        help="Optional path for bookkeeping (fidelity uses pgd_fidelity_harness).",
    )
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
        default="17",
        help="Comma-separated layer indices (ignored for baseline).",
    )
    p.add_argument("--projections", type=str, default="down_proj")
    p.add_argument("--pgd-iters", type=int, default=20, dest="pgd_iters")
    p.add_argument("--pgd-seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--text", type=str, default=DEFAULT_SNIPPET)
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="One prompt per line (non-empty). Overrides --text when provided with lines.",
    )
    p.add_argument(
        "--use-default-prompt-set",
        action="store_true",
        help=f"Use bundled default_eval_prompts.txt if --prompts-file is not set.",
    )
    p.add_argument(
        "--drift-layers",
        type=str,
        default="",
        help="Comma-separated decoder layer indices for hidden-state drift (PGD modes only).",
    )
    p.add_argument(
        "--drift-prompt-index",
        type=int,
        default=0,
        help="Which prompt (0-based) to use for drift forward passes.",
    )
    p.add_argument("--prefill-warmup", type=int, default=1)
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="If > 0, run greedy generate and report decode timing.",
    )
    p.add_argument("--decode-repeats", type=int, default=1)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--experiment-matrix",
        type=Path,
        default=None,
        help="Optional path to experiment_matrix.json (included in payload metadata).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    prompts = load_prompts_for_eval(
        text=args.text,
        prompts_file=args.prompts_file,
        use_default_prompt_set=args.use_default_prompt_set,
        default_prompts_path=DEFAULT_PROMPTS_PATH,
    )

    layers = parse_layers_spec(args.layers)
    projections = parse_projections_spec(args.projections)
    drift_layers = _parse_drift_layers(args.drift_layers)

    modes = ["baseline", "dense_pgd", "native_pgd"] if args.mode == "all" else [args.mode]
    if any(m != "baseline" for m in modes):
        if not layers:
            print("ERROR: --layers required when mode is not baseline.", file=sys.stderr)
            return 1
        validate_projections(projections)

    matrix_path = args.experiment_matrix
    if matrix_path is None:
        default_mx = _ROOT / "experiment_matrix.json"
        if default_mx.is_file():
            matrix_path = default_mx

    arms: List[Dict[str, Any]] = []
    for m in modes:
        arm_layers = [] if m == "baseline" else layers
        arm_projs = [] if m == "baseline" else projections
        arms.append(
            run_single_arm(
                m,
                model_id=args.model,
                device=args.device,
                torch_dtype=args.torch_dtype,
                layers=arm_layers,
                projections=arm_projs,
                rank=args.rank,
                pgd_iters=args.pgd_iters,
                pgd_seed=args.pgd_seed,
                prompts=prompts,
                max_length=args.max_length,
                prefill_warmup=args.prefill_warmup,
                prefill_repeats=args.prefill_repeats,
                max_new_tokens=args.max_new_tokens,
                decode_repeats=args.decode_repeats,
                drift_layers=[] if m == "baseline" else drift_layers,
                drift_prompt_index=args.drift_prompt_index,
            )
        )

    payload: Dict[str, Any] = {
        "arms": arms,
        "prompts_resolved_count": len(prompts),
        "prompts_preview": prompts[:3],
        "run_environment": collect_run_environment(),
        "safetensors_path_note": str(args.safetensors_path) if args.safetensors_path else None,
        "experiment_matrix_path": str(matrix_path) if matrix_path else None,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
