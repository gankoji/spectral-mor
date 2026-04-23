"""Tests for pgd_fidelity_harness helpers (no safetensors required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_fidelity_harness import (
    fidelity_rows_for_weight,
    frobenius_relative,
    gemma4_e2b_weight_key,
    truncated_svd_reconstruct,
)
from pgd_weight import reconstruct_dense, spec_from_pgd_modes
from pgd_enrichment import pgd_decompose


class TestFidelityHarness(unittest.TestCase):
    def test_weight_key_attn_mlp(self) -> None:
        self.assertIn("self_attn.q_proj", gemma4_e2b_weight_key(3, "q_proj"))
        self.assertIn("mlp.down_proj", gemma4_e2b_weight_key(3, "down_proj"))

    def test_weight_key_unknown(self) -> None:
        with self.assertRaises(ValueError):
            gemma4_e2b_weight_key(0, "not_a_proj")

    def test_truncated_svd_rank(self) -> None:
        np.random.seed(1)
        W = np.random.randn(30, 20).astype(np.float32)
        for r in (1, 5, 15):
            Wr = truncated_svd_reconstruct(W, r)
            u, s, vh = np.linalg.svd(W, full_matrices=False)
            expect = (u[:, :r] * s[:r]) @ vh[:r, :]
            np.testing.assert_allclose(Wr, expect, rtol=1e-5, atol=1e-6)

    def test_fidelity_rows_match_manual_frobenius(self) -> None:
        np.random.seed(2)
        W = np.random.randn(14, 22).astype(np.float32)
        rows = fidelity_rows_for_weight(
            W, layer=0, proj="down_proj", rank_budgets=[4, 8], include_svd=True
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            modes, _ = pgd_decompose(
                W,
                num_modes=row.rank_budget,
                max_fixed_point_iters=20,
                seed=42,
            )
            spec = spec_from_pgd_modes(modes, dtype=np.float32)
            W_pgd = reconstruct_dense(spec)
            manual_fro = frobenius_relative(W, W_pgd)
            self.assertAlmostEqual(row.frobenius_rel_pgd, manual_fro, places=5)
            W_svd = truncated_svd_reconstruct(W, row.rank_budget)
            manual_svd = frobenius_relative(W, W_svd)
            self.assertIsNotNone(row.frobenius_rel_svd)
            self.assertAlmostEqual(row.frobenius_rel_svd, manual_svd, places=5)


if __name__ == "__main__":
    unittest.main()
