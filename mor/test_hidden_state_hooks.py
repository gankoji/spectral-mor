"""Unit tests for hidden-state capture and drift metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hidden_state_hooks import (
    LayerForwardHooks,
    _hidden_after_layer_from_tuple,
    hidden_drift_stats,
)


class TinyDecoder(nn.Module):
    """Minimal stack so hook tests do not need Hugging Face."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache: bool = False):
        x = torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            4,
            device=input_ids.device,
            dtype=torch.float32,
        )
        for layer in self.layers:
            x = layer(x)
        return x


class TestHiddenDriftStats(unittest.TestCase):
    def test_identical_zero_drift(self) -> None:
        h = torch.randn(2, 5, 8)
        s = hidden_drift_stats(h, h)
        self.assertAlmostEqual(s.relative_l2, 0.0, places=6)
        self.assertAlmostEqual(s.cosine_mean, 1.0, places=5)

    def test_hidden_tuple_indexing(self) -> None:
        hs = tuple(torch.randn(1, 2, 3, 8) for _ in range(4))
        h1 = _hidden_after_layer_from_tuple(hs, 0)
        self.assertEqual(h1.shape, (1, 2, 3, 8))


class TestLayerHooks(unittest.TestCase):
    def test_hooks_capture_layer_outputs(self) -> None:
        dec = TinyDecoder()
        hooks = LayerForwardHooks(dec, [0, 1])
        try:
            ids = torch.ones(1, 3, dtype=torch.long)
            dec(ids)
        finally:
            hooks.remove()
        self.assertEqual(set(hooks.stored.keys()), {0, 1})
        self.assertEqual(hooks.stored[0].shape[-1], 4)


if __name__ == "__main__":
    unittest.main()
