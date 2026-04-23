"""Tests for HF PGD substitution (tiny ``nn.Module``, no checkpoint download)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_hf_substitution import (
    get_decoder_with_layers,
    linear_submodule,
    substitute_linear_weight_with_pgd,
    substitute_selected_linears,
)


class TinySelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinySelfAttn()
        self.mlp = TinyMLP()


class TinyCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])


class TestHFSubstitution(unittest.TestCase):
    def test_get_decoder_and_substitute_down_proj(self) -> None:
        m = TinyCausal()
        dec = get_decoder_with_layers(m)
        lin = linear_submodule(dec, 0, "down_proj")
        w0 = lin.weight.data.clone()
        substitute_linear_weight_with_pgd(lin, rank=3, max_fixed_point_iters=15, seed=0)
        self.assertEqual(lin.weight.shape, w0.shape)
        self.assertFalse(torch.allclose(lin.weight, w0))

    def test_substitute_selected_linears_smoke(self) -> None:
        m = TinyCausal()
        done = substitute_selected_linears(
            m, [1], ["q_proj"], rank=2, max_fixed_point_iters=10, seed=1
        )
        self.assertEqual(done, [(1, "q_proj", (8, 8))])


if __name__ == "__main__":
    unittest.main()
