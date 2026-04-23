"""Compressed linear layer representation from PGD modes (2D weights)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, Sequence[Any]]


@dataclass(frozen=True)
class PGDLinearSpec:
    """Rank-R approximation W ≈ sum_r u_r v_r^T with rows u_r, v_r stacked as U, V."""

    out_features: int
    in_features: int
    rank: int
    u: np.ndarray  # (rank, out_features)
    v: np.ndarray  # (rank, in_features)
    dtype: np.dtype
    layer_key: Optional[str] = None

    def __post_init__(self) -> None:
        u = np.asarray(self.u)
        v = np.asarray(self.v)
        if u.shape != (self.rank, self.out_features):
            raise ValueError(
                f"u must have shape ({self.rank}, {self.out_features}), got {u.shape}"
            )
        if v.shape != (self.rank, self.in_features):
            raise ValueError(
                f"v must have shape ({self.rank}, {self.in_features}), got {v.shape}"
            )
        if u.dtype != v.dtype:
            raise ValueError(f"u.dtype {u.dtype} != v.dtype {v.dtype}")
        if np.dtype(self.dtype) != u.dtype:
            raise ValueError(f"dtype field {self.dtype} does not match u/v dtype {u.dtype}")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)


def spec_from_pgd_modes(
    modes: Sequence[Sequence[np.ndarray]],
    *,
    dtype: Optional[np.dtype] = None,
    layer_key: Optional[str] = None,
) -> PGDLinearSpec:
    """Build a spec from ``pgd_decompose`` output for a 2D matrix (alpha folded into mode[0])."""
    if not modes:
        raise ValueError("modes must be non-empty")
    out_features = int(modes[0][0].shape[0])
    in_features = int(modes[0][1].shape[0])
    rank = len(modes)
    u = np.stack([np.asarray(m[0], dtype=np.float64) for m in modes], axis=0)
    v = np.stack([np.asarray(m[1], dtype=np.float64) for m in modes], axis=0)
    if dtype is None:
        dtype = np.result_type(u, v)
    u = u.astype(dtype, copy=False)
    v = v.astype(dtype, copy=False)
    return PGDLinearSpec(
        out_features=out_features,
        in_features=in_features,
        rank=rank,
        u=u,
        v=v,
        dtype=np.dtype(dtype),
        layer_key=layer_key,
    )


def reconstruct_dense(spec: PGDLinearSpec) -> np.ndarray:
    """Dense (out_features, in_features) matrix equal to sum_r outer(u_r, v_r)."""
    return (spec.u.T @ spec.v).astype(spec.dtype, copy=False)


def matvec_native(spec: PGDLinearSpec, x: ArrayLike) -> np.ndarray:
    """Compute y = W x with W from spec; x is (... , in_features)."""
    x_arr = np.asarray(x, dtype=spec.dtype)
    if x_arr.shape[-1] != spec.in_features:
        raise ValueError(
            f"x last dim must be {spec.in_features}, got {x_arr.shape[-1]}"
        )
    # s[..., r] = x · v_r
    s = np.einsum("...i,ri->...r", x_arr, spec.v)
    return np.einsum("...r,ro->...o", s, spec.u)


def _numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    if dtype == np.float16:
        return torch.float16
    raise TypeError(f"unsupported numpy dtype for torch: {dtype}")


def reconstruct_dense_torch(
    spec: PGDLinearSpec,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    td = dtype or _numpy_dtype_to_torch(spec.dtype)
    u = torch.as_tensor(spec.u, device=device, dtype=td)
    v = torch.as_tensor(spec.v, device=device, dtype=td)
    return u.T @ v


def matvec_native_torch(
    spec: PGDLinearSpec,
    x: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """y = W x; x shape (..., in_features) on same device as x."""
    if x.shape[-1] != spec.in_features:
        raise ValueError(
            f"x last dim must be {spec.in_features}, got {x.shape[-1]}"
        )
    td = dtype or _numpy_dtype_to_torch(spec.dtype)
    u = torch.as_tensor(spec.u, device=x.device, dtype=td)
    v = torch.as_tensor(spec.v, device=x.device, dtype=td)
    x_cast = x.to(dtype=td)
    s = torch.einsum("...i,ri->...r", x_cast, v)
    return torch.einsum("...r,ro->...o", s, u)
