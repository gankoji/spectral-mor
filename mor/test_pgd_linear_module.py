"""Tests for ``PGDLinear`` (exact SVD factors + parity with functional linear)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_linear import PGDLinear


class TestPGDLinearModule(unittest.TestCase):
    def test_forward_matches_linear_for_svd_factors(self) -> None:
        torch.manual_seed(0)
        out_f, in_f, r = 8, 12, 4
        w = torch.randn(out_f, in_f, dtype=torch.float64)
        u_full, s, vh = torch.linalg.svd(w, full_matrices=False)
        ur = u_full[:, :r] * s[:r]
        vr = vh[:r, :]
        w_r = ur @ vr
        u_st = ur.T.contiguous()
        v_st = vr.contiguous()
        b = torch.randn(out_f, dtype=torch.float64)
        m = PGDLinear(out_f, in_f, u_st, v_st, b)
        x = torch.randn(2, 5, in_f, dtype=torch.float64)
        y_m = m(x)
        y_ref = F.linear(x, w_r, b)
        torch.testing.assert_close(y_m, y_ref)

    def test_no_bias(self) -> None:
        torch.manual_seed(1)
        out_f, in_f, r = 5, 7, 3
        w = torch.randn(out_f, in_f)
        u_full, s, vh = torch.linalg.svd(w, full_matrices=False)
        ur = u_full[:, :r] * s[:r]
        vr = vh[:r, :]
        w_r = ur @ vr
        m = PGDLinear(out_f, in_f, ur.T.contiguous(), vr.contiguous(), None)
        x = torch.randn(11, in_f)
        torch.testing.assert_close(m(x), F.linear(x, w_r, None))


if __name__ == "__main__":
    unittest.main()
