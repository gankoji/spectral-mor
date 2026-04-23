"""
Phase A: rank–fidelity curves on real (mmap) weights — Frobenius error, explained
variance, optional truncated-SVD baseline, and a Gaussian activation proxy.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pgd_enrichment import pgd_decompose
from pgd_weight import reconstruct_dense, spec_from_pgd_modes

ATTN_PROJS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_PROJS = frozenset({"gate_proj", "up_proj", "down_proj"})


def gemma4_e2b_weight_key(layer: int, proj: str) -> str:
    """HF safetensors key for Gemma 4 E2B linear weights (matches gemma4_e2b_mor)."""
    if proj in ATTN_PROJS:
        return f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
    if proj in MLP_PROJS:
        return f"model.language_model.layers.{layer}.mlp.{proj}.weight"
    raise ValueError(
        f"unknown proj {proj!r}; expected one of {sorted(ATTN_PROJS | MLP_PROJS)}"
    )


def load_weights_mmap(safetensors_path: Path, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load only listed tensors from a safetensors file (memory-efficient)."""
    from safetensors.torch import safe_open

    path = Path(safetensors_path)
    result: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keyset = set(f.keys())
        for key in keys:
            if key not in keyset:
                continue
            t = f.get_tensor(key)
            result[key] = t.float().numpy().astype(np.float32, copy=False)
    return result


def frobenius_relative(W: np.ndarray, What: np.ndarray) -> float:
    denom = np.linalg.norm(W) + 1e-12
    return float(np.linalg.norm(W - What) / denom)


def truncated_svd_reconstruct(W: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-`rank` approximation in Frobenius norm."""
    u, s, vh = np.linalg.svd(W, full_matrices=False)
    r = min(rank, len(s))
    u = u[:, :r]
    s = s[:r]
    vh = vh[:r, :]
    return (u * s) @ vh


def activation_proxy_stats(
    W: np.ndarray,
    What: np.ndarray,
    *,
    rng: np.random.Generator,
    n_samples: int,
) -> Tuple[float, float]:
    """Mean relative L2 error and cosine similarity of W@x vs Ŵ@x for Gaussian x."""
    _, inn = W.shape
    rels: List[float] = []
    coss: List[float] = []
    for _ in range(n_samples):
        x = rng.standard_normal(inn).astype(W.dtype, copy=False)
        y = W @ x
        yh = What @ x
        ny = float(np.linalg.norm(y))
        nyh = float(np.linalg.norm(yh))
        if ny < 1e-12:
            rels.append(0.0 if float(np.linalg.norm(y - yh)) < 1e-12 else float("inf"))
            coss.append(1.0 if nyh < 1e-12 else 0.0)
            continue
        rels.append(float(np.linalg.norm(y - yh) / ny))
        den = ny * nyh + 1e-12
        coss.append(float(np.dot(y, yh) / den))
    return float(np.mean(rels)), float(np.mean(coss))


@dataclass
class FidelityRow:
    layer: int
    proj: str
    shape: Tuple[int, int]
    rank_budget: int
    modes_extracted: int
    frobenius_rel_pgd: float
    explained_var_pgd: float
    walltime_s_pgd: float
    frobenius_rel_svd: Optional[float]
    walltime_s_svd: Optional[float]
    activation_rel_err: float
    activation_cosine: float

    def as_csv_dict(self) -> Dict[str, object]:
        m, n = self.shape
        return {
            "layer": self.layer,
            "proj": self.proj,
            "shape_m": m,
            "shape_n": n,
            "rank_budget": self.rank_budget,
            "modes_extracted": self.modes_extracted,
            "frobenius_rel_pgd": self.frobenius_rel_pgd,
            "explained_var_pgd": self.explained_var_pgd,
            "walltime_s_pgd": self.walltime_s_pgd,
            "frobenius_rel_svd": (
                "" if self.frobenius_rel_svd is None else self.frobenius_rel_svd
            ),
            "walltime_s_svd": (
                "" if self.walltime_s_svd is None else self.walltime_s_svd
            ),
            "activation_rel_err": self.activation_rel_err,
            "activation_cosine": self.activation_cosine,
        }


def fidelity_rows_for_weight(
    W: np.ndarray,
    layer: int,
    proj: str,
    rank_budgets: Sequence[int],
    *,
    pgd_max_fixed_point_iters: int = 20,
    pgd_seed: int = 42,
    include_svd: bool = False,
    activation_samples: int = 8,
    activation_seed: int = 0,
) -> List[FidelityRow]:
    """Sweep PGD ranks for one weight matrix; optional SVD baseline per rank."""
    W = np.asarray(W, dtype=np.float32)
    m, n = W.shape
    initial_norm = float(np.linalg.norm(W))
    initial_norm_sq = initial_norm**2 + 1e-12

    rng = np.random.default_rng(activation_seed)

    rows: List[FidelityRow] = []
    for rank_budget in rank_budgets:
        if rank_budget <= 0:
            continue
        if rank_budget > min(m, n):
            continue

        t0 = time.perf_counter()
        modes, residual_norms = pgd_decompose(
            W,
            num_modes=rank_budget,
            max_fixed_point_iters=pgd_max_fixed_point_iters,
            seed=pgd_seed,
        )
        wall_pgd = time.perf_counter() - t0

        spec = spec_from_pgd_modes(modes, dtype=np.float32)
        W_pgd = reconstruct_dense(spec)
        final_res = float(residual_norms[-1])
        explained = 1.0 - (final_res**2) / initial_norm_sq
        fro_pgd = frobenius_relative(W, W_pgd)
        act_rel, act_cos = activation_proxy_stats(
            W, W_pgd, rng=rng, n_samples=activation_samples
        )

        fro_svd: Optional[float] = None
        wall_svd: Optional[float] = None
        if include_svd:
            t1 = time.perf_counter()
            W_svd = truncated_svd_reconstruct(W, rank_budget)
            wall_svd = time.perf_counter() - t1
            fro_svd = frobenius_relative(W, W_svd)

        rows.append(
            FidelityRow(
                layer=layer,
                proj=proj,
                shape=(m, n),
                rank_budget=rank_budget,
                modes_extracted=len(modes),
                frobenius_rel_pgd=fro_pgd,
                explained_var_pgd=float(explained),
                walltime_s_pgd=wall_pgd,
                frobenius_rel_svd=fro_svd,
                walltime_s_svd=wall_svd,
                activation_rel_err=act_rel,
                activation_cosine=act_cos,
            )
        )
    return rows


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


CSV_FIELDS = [
    "layer",
    "proj",
    "shape_m",
    "shape_n",
    "rank_budget",
    "modes_extracted",
    "frobenius_rel_pgd",
    "explained_var_pgd",
    "walltime_s_pgd",
    "frobenius_rel_svd",
    "walltime_s_svd",
    "activation_rel_err",
    "activation_cosine",
]


def write_csv(path: Path, rows: Iterable[FidelityRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row.as_csv_dict())


def print_markdown_summary(rows: Sequence[FidelityRow]) -> None:
    """Compact table: layer, proj, R, fro_pgd, expl_var, act_rel."""
    print("\n| layer | proj | R | ‖W-Ŵ‖_F/‖W‖ | expl.var | ‖Δy‖/‖y‖ | cos |")
    print("|------:|:-----|--:|-----------:|---------:|--------:|----:|")
    for r in rows:
        print(
            f"| {r.layer} | {r.proj} | {r.rank_budget} | "
            f"{r.frobenius_rel_pgd:.4f} | {r.explained_var_pgd:.4f} | "
            f"{r.activation_rel_err:.4f} | {r.activation_cosine:.4f} |"
        )


def default_safetensors_path() -> Optional[Path]:
    env = os.environ.get("SPECTRAL_MOR_SAFETENSORS")
    if env:
        return Path(env)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PGD rank–fidelity harness (mmap safetensors).")
    p.add_argument(
        "--safetensors-path",
        type=Path,
        default=None,
        help="Path to model.safetensors (or set SPECTRAL_MOR_SAFETENSORS).",
    )
    p.add_argument("--layers", type=str, default="0,17,34", help="Comma-separated layer ids.")
    p.add_argument(
        "--projections",
        type=str,
        default="down_proj",
        help="Comma-separated projection suffixes (e.g. down_proj,q_proj).",
    )
    p.add_argument("--ranks", type=str, default="8,32,128", help="Comma-separated rank budgets.")
    p.add_argument("--output-csv", type=Path, default=Path("pgd_fidelity_results.csv"))
    p.add_argument("--svd", action="store_true", help="Include truncated-SVD baseline per rank.")
    p.add_argument("--pgd-iters", type=int, default=20, dest="pgd_iters")
    p.add_argument("--pgd-seed", type=int, default=42)
    p.add_argument("--activation-samples", type=int, default=8)
    p.add_argument("--activation-seed", type=int, default=0)
    args = p.parse_args(list(argv) if argv is not None else None)

    st_path = args.safetensors_path or default_safetensors_path()
    if st_path is None or not st_path.is_file():
        print(
            "ERROR: provide --safetensors-path to model.safetensors "
            "or set SPECTRAL_MOR_SAFETENSORS to an existing file.",
            file=sys.stderr,
        )
        return 1

    layers = _parse_int_list(args.layers)
    projections = _parse_str_list(args.projections)
    ranks = _parse_int_list(args.ranks)

    keys: List[str] = []
    key_meta: List[Tuple[int, str]] = []
    for layer in layers:
        for proj in projections:
            k = gemma4_e2b_weight_key(layer, proj)
            keys.append(k)
            key_meta.append((layer, proj))

    print(f"Loading {len(keys)} tensors from {st_path} ...")
    tensors = load_weights_mmap(st_path, keys)
    if not tensors:
        print("ERROR: no tensors loaded (check keys / path).", file=sys.stderr)
        return 1

    all_rows: List[FidelityRow] = []
    for (layer, proj), key in zip(key_meta, keys):
        if key not in tensors:
            print(f"  skip missing key: {key}", file=sys.stderr)
            continue
        W = tensors[key]
        print(f"  layer {layer} {proj} {W.shape} ...")
        all_rows.extend(
            fidelity_rows_for_weight(
                W,
                layer,
                proj,
                ranks,
                pgd_max_fixed_point_iters=args.pgd_iters,
                pgd_seed=args.pgd_seed,
                include_svd=args.svd,
                activation_samples=args.activation_samples,
                activation_seed=args.activation_seed,
            )
        )

    write_csv(args.output_csv, all_rows)
    print(f"\nWrote {len(all_rows)} rows to {args.output_csv}")
    print_markdown_summary(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
