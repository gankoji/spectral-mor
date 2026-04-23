"""``nn.Module`` wrapper for PGD factorized weights (native matmul, no dense W)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PGDLinear(nn.Module):
    """
    Linear layer with stored factors ``U`` (rank × out) and ``V`` (rank × in) so that
    ``W ≈ UᵀV`` matches ``nn.Linear`` with weight shape ``(out, in)``.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        u: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if u.shape[1] != out_features or v.shape[1] != in_features:
            raise ValueError(
                f"u/v incompatible with out={out_features}, in={in_features}: "
                f"u.shape={tuple(u.shape)}, v.shape={tuple(v.shape)}"
            )
        if u.shape[0] != v.shape[0]:
            raise ValueError("u and v must have the same rank (leading dimension)")
        self.out_features = out_features
        self.in_features = in_features
        self.rank = int(u.shape[0])
        self.register_buffer("u", u.contiguous())
        self.register_buffer("v", v.contiguous())
        if bias is not None:
            if bias.shape != (out_features,):
                raise ValueError(f"bias shape {bias.shape} != ({out_features},)")
            self.register_buffer("bias_tensor", bias.contiguous())
        else:
            self.bias_tensor = None  # type: ignore[assignment]

    @property
    def weight(self) -> torch.Tensor:
        """Dense reconstruction for compatibility checks (materializes full W)."""
        return self.u.T @ self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u.to(device=x.device, dtype=x.dtype)
        v = self.v.to(device=x.device, dtype=x.dtype)
        s = torch.einsum("...i,ri->...r", x, v)
        y = torch.einsum("...r,ro->...o", s, u)
        b = getattr(self, "bias_tensor", None)
        if b is not None:
            y = y + b.to(device=x.device, dtype=x.dtype)
        return y
