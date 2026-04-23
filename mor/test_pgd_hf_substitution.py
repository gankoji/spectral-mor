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

from pgd_linear import PGDLinear
from pgd_hf_substitution import (
    get_decoder_with_layers,
    linear_submodule,
    substitute_linear_weight_with_pgd,
    substitute_selected_linears,
    substitute_selected_linears_native,
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

    def test_substitute_native_replaces_with_pgd_linear(self) -> None:
        m = TinyCausal()
        substitute_selected_linears_native(
            m, [0], ["down_proj"], rank=3, max_fixed_point_iters=12, seed=2
        )
        dec = get_decoder_with_layers(m)
        down = dec.layers[0].mlp.down_proj
        self.assertIsInstance(down, PGDLinear)
        x = torch.randn(4, 16)
        y = down(x)
        self.assertEqual(y.shape, (4, 8))

    def test_native_matches_dense_pgd_outputs(self) -> None:
        torch.manual_seed(11)
        base = TinyCausal()
        sd = base.state_dict()
        m_dense = TinyCausal()
        m_dense.load_state_dict(sd)
        m_native = TinyCausal()
        m_native.load_state_dict(sd)
        rank = 5
        seed = 123
        substitute_selected_linears(
            m_dense, [0], ["down_proj"], rank=rank, max_fixed_point_iters=15, seed=seed
        )
        substitute_selected_linears_native(
            m_native, [0], ["down_proj"], rank=rank, max_fixed_point_iters=15, seed=seed
        )
        x = torch.randn(6, 16)
        y_d = m_dense.language_model.layers[0].mlp.down_proj(x)
        y_n = m_native.language_model.layers[0].mlp.down_proj(x)
        torch.testing.assert_close(y_d, y_n, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
