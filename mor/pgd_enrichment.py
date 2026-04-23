import numpy as np


def _update_factor(res, factors, j, d):
    """Single factor update using einsum."""
    res_indices = list(range(d))
    inputs = [res, res_indices]
    for i in range(d):
        if i != j:
            inputs.append(factors[i])
            inputs.append([i])
    inputs.append([j])
    new_f = np.einsum(*inputs)
    fnorm = np.linalg.norm(new_f)
    if fnorm > 1e-12:
        return new_f / fnorm
    return new_f


def _compute_alpha(res, factors, d):
    """Computes the optimal scaling factor alpha."""
    inputs = [res, list(range(d))]
    for i in range(d):
        inputs.append(factors[i])
        inputs.append([i])
    inputs.append([])
    return np.einsum(*inputs)


def _reconstruct_rank1(factors, d):
    """Reconstructs the unit rank-1 tensor."""
    res_indices = list(range(d))
    inputs = []
    for i in range(d):
        inputs.append(factors[i])
        inputs.append([i])
    inputs.append(res_indices)
    return np.einsum(*inputs)


def pgd_decompose(tensor, num_modes=10, max_fixed_point_iters=20, seed=None, tol=1e-6):
    r"""
    Decomposes a d-dimensional tensor into a sum of rank-1 separable products using NumPy.

    u(x1, ..., xd) \approx \sum_{m=1}^M \prod_{j=1}^d f_j^m(x_j)
    """
    if seed is not None:
        np.random.seed(seed)

    shape = tensor.shape
    d = len(shape)
    residual = np.copy(tensor)
    modes = []

    initial_norm = np.linalg.norm(tensor)
    residual_norms = [float(initial_norm)]

    for _ in range(num_modes):
        factors = [np.random.normal(size=(shape[i],)) for i in range(d)]
        factors = [f / (np.linalg.norm(f) + 1e-12) for f in factors]

        for _ in range(max_fixed_point_iters):
            factors_prev = [np.copy(f) for f in factors]
            for j in range(d):
                factors[j] = _update_factor(residual, factors, j, d)

            delta = sum(np.linalg.norm(f - f_p) for f, f_p in zip(factors, factors_prev))
            if delta < tol:
                break

        alpha = _compute_alpha(residual, factors, d)

        stored_factors = [np.copy(f) for f in factors]
        stored_factors[0] = stored_factors[0] * alpha
        modes.append(stored_factors)

        rank1_unit = _reconstruct_rank1(factors, d)
        residual = residual - alpha * rank1_unit

        res_norm = np.linalg.norm(residual)
        residual_norms.append(float(res_norm))

        if res_norm < 1e-10 * initial_norm:
            break

    return modes, residual_norms
