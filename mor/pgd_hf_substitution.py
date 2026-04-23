"""
Phase B: replace selected Hugging Face ``nn.Linear`` weights with dense PGD reconstructions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_enrichment import pgd_decompose
from pgd_fidelity_harness import ATTN_PROJS, MLP_PROJS
from pgd_weight import reconstruct_dense, spec_from_pgd_modes


def get_decoder_with_layers(model: nn.Module) -> nn.Module:
    """Return the submodule that owns ``layers`` (decoder stack)."""
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model
        if hasattr(inner, "layers"):
            return inner
    raise RuntimeError(
        "Cannot locate decoder layers on this model. "
        "Expected language_model.layers or model.layers (Gemma-style)."
    )


def linear_submodule(
    decoder: nn.Module,
    layer_idx: int,
    proj: str,
) -> nn.Linear:
    """Resolve ``nn.Linear`` for a Gemma-style layer index and projection name."""
    if proj in ATTN_PROJS:
        block = decoder.layers[layer_idx].self_attn
    elif proj in MLP_PROJS:
        block = decoder.layers[layer_idx].mlp
    else:
        raise ValueError(
            f"unknown proj {proj!r}; expected attention or MLP projection name"
        )
    mod = getattr(block, proj, None)
    if mod is None or not isinstance(mod, nn.Linear):
        raise TypeError(f"{proj} at layer {layer_idx} is not an nn.Linear (got {type(mod)})")
    return mod


def substitute_linear_weight_with_pgd(
    linear: nn.Linear,
    rank: int,
    *,
    max_fixed_point_iters: int = 20,
    seed: int = 42,
) -> None:
    """In-place: set ``linear.weight`` to rank-``rank`` PGD reconstruction (float32 PGD, cast back)."""
    device = linear.weight.device
    param_dtype = linear.weight.dtype
    w = linear.weight.detach().float().cpu().numpy()
    if rank <= 0:
        raise ValueError("rank must be positive")
    r_max = min(rank, w.shape[0], w.shape[1])
    modes, _ = pgd_decompose(
        w,
        num_modes=r_max,
        max_fixed_point_iters=max_fixed_point_iters,
        seed=seed,
    )
    spec = spec_from_pgd_modes(modes, dtype=np.float32)
    w_hat = reconstruct_dense(spec)
    w_t = torch.from_numpy(w_hat).to(device=device, dtype=param_dtype)
    linear.weight.data.copy_(w_t)


def substitute_selected_linears(
    model: nn.Module,
    layer_indices: Sequence[int],
    projections: Sequence[str],
    rank: int,
    *,
    max_fixed_point_iters: int = 20,
    seed: int = 42,
) -> List[Tuple[int, str, Tuple[int, int]]]:
    """
    Apply PGD reconstruction to each selected linear. Returns list of
    ``(layer, proj, (out_features, in_features))`` for substituted modules.
    """
    decoder = get_decoder_with_layers(model)
    n_layers = len(decoder.layers)
    done: List[Tuple[int, str, Tuple[int, int]]] = []
    for li in layer_indices:
        if li < 0 or li >= n_layers:
            raise IndexError(f"layer index {li} out of range [0, {n_layers})")
        for proj in projections:
            lin = linear_submodule(decoder, li, proj)
            substitute_linear_weight_with_pgd(
                lin,
                rank,
                max_fixed_point_iters=max_fixed_point_iters,
                seed=seed,
            )
            done.append((li, proj, (lin.out_features, lin.in_features)))
    return done


def parse_layers_spec(spec: str) -> List[int]:
    s = spec.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_projections_spec(spec: str) -> List[str]:
    s = spec.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def validate_projections(projections: Iterable[str]) -> None:
    allowed: Set[str] = set(ATTN_PROJS) | set(MLP_PROJS)
    for p in projections:
        if p not in allowed:
            raise ValueError(f"unknown projection {p!r}; allowed: {sorted(allowed)}")
