"""Capture decoder hidden states for drift vs baseline (post-substitution analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class HiddenDriftLayerStats:
    relative_l2: float
    cosine_mean: float
    ref_rms: float
    mod_rms: float
    ref_hidden_norm_max: float
    mod_hidden_norm_max: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "relative_l2": self.relative_l2,
            "cosine_mean": self.cosine_mean,
            "ref_rms": self.ref_rms,
            "mod_rms": self.mod_rms,
            "ref_hidden_norm_max": self.ref_hidden_norm_max,
            "mod_hidden_norm_max": self.mod_hidden_norm_max,
        }


def hidden_drift_stats(h_ref: torch.Tensor, h_mod: torch.Tensor, eps: float = 1e-8) -> HiddenDriftLayerStats:
    """
    Compare two hidden tensors ``(..., hidden)`` (e.g. batch, seq, dim) from the same positions.
    """
    ref = h_ref.detach().float()
    mod = h_mod.detach().float()
    if ref.shape != mod.shape:
        raise ValueError(f"shape mismatch: ref {ref.shape} vs mod {mod.shape}")
    diff = mod - ref
    ref_norm = ref.norm()
    rel_l2 = float((diff.norm() / (ref_norm + eps)).item())
    ref_f = ref.reshape(-1, ref.shape[-1])
    mod_f = mod.reshape(-1, mod.shape[-1])
    cos = float(F.cosine_similarity(ref_f, mod_f, dim=-1, eps=eps).mean().item())
    ref_rms = float((ref_f.pow(2).mean().sqrt()).item())
    mod_rms = float((mod_f.pow(2).mean().sqrt()).item())
    ref_max = float(ref_f.norm(dim=-1).max().item())
    mod_max = float(mod_f.norm(dim=-1).max().item())
    return HiddenDriftLayerStats(
        relative_l2=rel_l2,
        cosine_mean=cos,
        ref_rms=ref_rms,
        mod_rms=mod_rms,
        ref_hidden_norm_max=ref_max,
        mod_hidden_norm_max=mod_max,
    )


def _hidden_after_layer_from_tuple(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
) -> torch.Tensor:
    """
    HF convention: ``hidden_states[0]`` = embeddings, ``hidden_states[i+1]`` = output after layer ``i``.
    """
    j = layer_idx + 1
    if j < 0 or j >= len(hidden_states):
        raise IndexError(
            f"layer_idx {layer_idx} -> hidden_states[{j}] out of range (len={len(hidden_states)})"
        )
    return hidden_states[j]


def forward_decoder_with_hidden_states(
    decoder: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    """Run decoder forward and return the ``hidden_states`` tuple."""
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
    }
    out = decoder(**kwargs)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError(
            "Decoder did not return hidden_states; try a Gemma/Llama-style model or use hooks."
        )
    return hs


def capture_post_layer_hidden(
    decoder: nn.Module,
    layer_indices: List[int],
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Map layer index -> hidden tensor after that layer (CPU float copy)."""
    hs = forward_decoder_with_hidden_states(
        decoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return {
        li: _hidden_after_layer_from_tuple(hs, li).detach().float().cpu()
        for li in layer_indices
    }


class LayerForwardHooks:
    """Fallback: forward hooks on ``decoder.layers[i]`` if hidden_states API differs."""

    def __init__(self, decoder: nn.Module, layer_indices: List[int]) -> None:
        self.stored: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        def make_hook(li: int) -> Callable:
            def _hook(
                module: nn.Module,
                inp: Tuple[torch.Tensor, ...],
                out: torch.Tensor | Tuple[torch.Tensor, ...],
            ) -> None:
                t = out[0] if isinstance(out, tuple) else out
                self.stored[li] = t.detach().float().cpu()

            return _hook

        layers = decoder.layers
        for li in layer_indices:
            self._hooks.append(layers[li].register_forward_hook(make_hook(li)))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def capture_post_layer_hidden_hooks(
    decoder: nn.Module,
    layer_indices: List[int],
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    hooks = LayerForwardHooks(decoder, layer_indices)
    try:
        with torch.no_grad():
            decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        hooks.remove()
    if set(hooks.stored.keys()) != set(layer_indices):
        missing = set(layer_indices) - set(hooks.stored.keys())
        raise RuntimeError(f"hook capture incomplete; missing layers {missing}")
    return hooks.stored


def capture_post_layer_hidden_auto(
    decoder: nn.Module,
    layer_indices: List[int],
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Try tuple ``hidden_states`` first; fall back to per-layer hooks."""
    try:
        return capture_post_layer_hidden(
            decoder,
            layer_indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    except (RuntimeError, TypeError, AttributeError):
        return capture_post_layer_hidden_hooks(
            decoder,
            layer_indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
