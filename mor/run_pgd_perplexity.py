#!/usr/bin/env python3
"""
Load a causal LM, optionally substitute selected linears with dense PGD weights, then
report cross-entropy loss (NLL) / perplexity on a short text (Phase B driver).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_hf_substitution import (
    parse_layers_spec,
    parse_projections_spec,
    substitute_selected_linears,
    validate_projections,
)

DEFAULT_SNIPPET = (
    "The goal is to measure whether low-rank PGD reconstructions preserve "
    "language-model behavior on a fixed prompt."
)


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    td = dtype_map[torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=td,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=td,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(device)
    model.eval()
    return model, tokenizer


def nll_metrics(
    model: nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_length: int,
) -> Dict[str, Any]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    if attn is not None:
        n_tokens = int(attn.sum().item())
    else:
        n_tokens = int(input_ids.shape[1])
    input_ids = input_ids.to(device)
    if attn is not None:
        attn = attn.to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            labels=input_ids,
        )
    loss = float(out.loss.item())
    ppl = float(torch.exp(torch.tensor(loss)).item())
    return {
        "loss_nats_per_token": loss,
        "perplexity": ppl,
        "num_tokens": n_tokens,
        "max_length": max_length,
    }


def resolve_device(model: nn.Module, fallback: str) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return torch.device(fallback)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PGD dense substitution + perplexity on a snippet.")
    p.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
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
        help="Comma-separated layer indices (empty = no substitution, baseline).",
    )
    p.add_argument(
        "--projections",
        type=str,
        default="down_proj",
        help="Comma-separated: q_proj, k_proj, ... gate_proj, up_proj, down_proj",
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
    if layers:
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
    if layers:
        print(
            f"Substituting PGD rank {args.rank} for layers={layers} projections={projections} ...",
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

    peak_mb: Optional[float] = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t2 = time.perf_counter()
    metrics = nll_metrics(
        model,
        tokenizer,
        text,
        dev,
        max_length=args.max_length,
    )
    fwd_s = time.perf_counter() - t2

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    result: Dict[str, Any] = {
        "model": args.model,
        "substituted": [{"layer": a, "proj": b, "shape": list(c)} for a, b, c in substituted],
        "rank": args.rank if layers else 0,
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
