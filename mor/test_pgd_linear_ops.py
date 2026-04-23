"""Parity tests for PGD linear reconstruction and native matvec."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_enrichment import pgd_decompose
from pgd_weight import (
    PGDLinearSpec,
    matvec_native,
    matvec_native_torch,
    reconstruct_dense,
    reconstruct_dense_torch,
    spec_from_pgd_modes,
)


class TestPGDLinearOps(unittest.TestCase):
    def test_reconstruct_matches_outer_sum(self) -> None:
        np.random.seed(0)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                W = np.random.randn(12, 18).astype(dtype)
                modes, _ = pgd_decompose(W, num_modes=6, max_fixed_point_iters=20, seed=1)
                spec = spec_from_pgd_modes(modes, dtype=dtype)
                manual = np.zeros_like(W)
                for mode in modes:
                    manual += np.outer(mode[0], mode[1])
                dense = reconstruct_dense(spec)
                np.testing.assert_allclose(dense, manual, rtol=1e-5, atol=1e-6)

    def test_matvec_matches_dense_multiply_vector(self) -> None:
        np.random.seed(2)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                W = np.random.randn(10, 24).astype(dtype)
                modes, _ = pgd_decompose(W, num_modes=5, seed=3)
                spec = spec_from_pgd_modes(modes, dtype=dtype)
                W_hat = reconstruct_dense(spec)
                x = np.random.randn(24).astype(dtype)
                y_dense = W_hat @ x
                y_native = matvec_native(spec, x)
                np.testing.assert_allclose(y_native, y_dense, rtol=1e-5, atol=1e-6)

    def test_matvec_batched(self) -> None:
        np.random.seed(4)
        W = np.random.randn(8, 16).astype(np.float32)
        modes, _ = pgd_decompose(W, num_modes=4, seed=5)
        spec = spec_from_pgd_modes(modes, dtype=np.float32)
        W_hat = reconstruct_dense(spec)
        x = np.random.randn(5, 7, 16).astype(np.float32)
        y_dense = x @ W_hat.T
        y_native = matvec_native(spec, x)
        np.testing.assert_allclose(y_native, y_dense, rtol=1e-4, atol=1e-5)

    def test_torch_parity(self) -> None:
        np.random.seed(6)
        W = np.random.randn(11, 19).astype(np.float32)
        modes, _ = pgd_decompose(W, num_modes=5, seed=7)
        spec = spec_from_pgd_modes(modes, dtype=np.float32)
        W_np = reconstruct_dense(spec)
        W_t = reconstruct_dense_torch(spec)
        torch.testing.assert_close(W_t, torch.from_numpy(W_np))

        x = torch.randn(3, 19, dtype=torch.float32)
        y_np = matvec_native(spec, x.numpy())
        y_t = matvec_native_torch(spec, x)
        torch.testing.assert_close(y_t, torch.from_numpy(y_np), rtol=1e-5, atol=1e-6)

    def test_frobenius_error_consistent_with_manual(self) -> None:
        np.random.seed(8)
        W = np.random.randn(20, 15).astype(np.float64)
        modes, _ = pgd_decompose(W, num_modes=8, seed=9)
        spec = spec_from_pgd_modes(modes, dtype=np.float64)
        W_hat = reconstruct_dense(spec)
        err = np.linalg.norm(W - W_hat) / np.linalg.norm(W)
        manual = np.zeros_like(W)
        for mode in modes:
            manual += np.outer(mode[0], mode[1])
        err_m = np.linalg.norm(W - manual) / np.linalg.norm(W)
        self.assertAlmostEqual(float(err), float(err_m), places=10)


if __name__ == "__main__":
    unittest.main()
