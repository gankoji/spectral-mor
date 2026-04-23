import importlib.util
import os
import sys

import jax
import jax.numpy as jnp

PATH = os.path.join(os.path.dirname(__file__), "pgd_enrichment.py")
spec = importlib.util.spec_from_file_location("pgd_enrichment", PATH)
pgd_enrichment = importlib.util.module_from_spec(spec)
sys.modules["pgd_enrichment"] = pgd_enrichment
spec.loader.exec_module(pgd_enrichment)


def test_pgd_monotonic_residual_decrease():
    key = jax.random.PRNGKey(42)
    shape = (10, 12, 8)
    key, subkey = jax.random.split(key)
    target = jax.random.normal(subkey, shape)

    num_modes = 5
    key, subkey = jax.random.split(key)
    _, residuals = pgd_enrichment.pgd_decompose(target, num_modes=num_modes, seed=42)

    for i in range(1, len(residuals)):
        assert residuals[i] <= residuals[i - 1] + 1e-10
        if i < num_modes:
            assert residuals[i] < residuals[0]


def test_pgd_reconstruction_error():
    f1 = jnp.array([1.0, 2.0])
    f2 = jnp.array([3.0, 4.0, 5.0])
    f3 = jnp.array([6.0, 7.0])

    target = jnp.einsum(f1, [0], f2, [1], f3, [2], [0, 1, 2])

    modes, _ = pgd_enrichment.pgd_decompose(target, num_modes=1, seed=123)
    reconstructed = jnp.einsum(modes[0][0], [0], modes[0][1], [1], modes[0][2], [2], [0, 1, 2])

    error = jnp.linalg.norm(target - reconstructed)
    assert error < 1e-4


if __name__ == "__main__":
    test_pgd_monotonic_residual_decrease()
    test_pgd_reconstruction_error()
    print("All tests passed!")
