"""Shared model load, NLL / timing helpers for PGD inference scripts."""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    td = dtype_map[torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=td,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=td,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(device)
    model.eval()
    return model, tokenizer


def resolve_device(model: nn.Module, fallback: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def nll_metrics(
    model: nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_length: int,
) -> Dict[str, Any]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    if attn is not None:
        n_tokens = int(attn.sum().item())
    else:
        n_tokens = int(input_ids.shape[1])
    input_ids = input_ids.to(device)
    if attn is not None:
        attn = attn.to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            labels=input_ids,
            use_cache=False,
        )
    loss = float(out.loss.item())
    ppl = float(torch.exp(torch.tensor(loss)).item())
    return {
        "loss_nats_per_token": loss,
        "perplexity": ppl,
        "num_tokens": n_tokens,
        "max_length": max_length,
    }


def nll_metrics_many(
    model: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    *,
    max_length: int,
) -> Dict[str, Any]:
    """Per-prompt NLL / perplexity plus mean/std/min/max over non-empty prompts."""
    rows: List[Dict[str, Any]] = []
    losses: List[float] = []
    ppls: List[float] = []
    for i, raw in enumerate(prompts):
        p = raw.strip()
        if not p:
            continue
        m = nll_metrics(model, tokenizer, p, device, max_length=max_length)
        row = {"prompt_index": i, **m}
        rows.append(row)
        losses.append(m["loss_nats_per_token"])
        ppls.append(m["perplexity"])

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def _std_sample(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mu = _mean(xs)
        return float((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)

    return {
        "per_prompt": rows,
        "num_prompts_evaluated": len(losses),
        "loss_mean": _mean(losses),
        "loss_std_sample": _std_sample(losses),
        "loss_min": float(min(losses)) if losses else float("nan"),
        "loss_max": float(max(losses)) if losses else float("nan"),
        "perplexity_mean": _mean(ppls),
        "perplexity_std_sample": _std_sample(ppls),
        "perplexity_min": float(min(ppls)) if ppls else float("nan"),
        "perplexity_max": float(max(ppls)) if ppls else float("nan"),
    }


def time_forward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> Tuple[float, float]:
    """Mean and stdev (across repeats) wall seconds for one forward (with labels)."""
    for _ in range(warmup):
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_t = sum(times) / len(times)
    var = sum((t - mean_t) ** 2 for t in times) / max(len(times), 1)
    std_t = var**0.5
    return mean_t, std_t


def time_generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int,
    max_prompt_length: int,
    warmup: int = 0,
    repeats: int = 1,
) -> Tuple[float, int]:
    """Total wall seconds and total new tokens (best-effort) for greedy generate."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_new = 0
    for _ in range(repeats):
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        total_new += int(out.shape[1] - input_ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / max(repeats, 1), total_new // max(repeats, 1)


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def load_prompts_for_eval(
    *,
    text: str,
    prompts_file: Optional[Path],
    use_default_prompt_set: bool,
    default_prompts_path: Path,
) -> List[str]:
    """Resolve evaluation prompts: explicit file > default set > single ``text``."""
    if prompts_file is not None:
        lines = Path(prompts_file).read_text(encoding="utf-8").splitlines()
        found = [ln for ln in lines if ln.strip()]
        if found:
            return found
    if use_default_prompt_set and default_prompts_path.is_file():
        lines = default_prompts_path.read_text(encoding="utf-8").splitlines()
        found = [ln for ln in lines if ln.strip()]
        if found:
            return found
    t = text.strip()
    return [t] if t else ["."]


def collect_run_environment() -> Dict[str, Any]:
    """Best-effort metadata for experiment logs (hardware, library versions)."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["cuda_version_torch"] = torch.version.cuda
        try:
            env["cudnn_version"] = str(torch.backends.cudnn.version())
        except Exception:
            env["cudnn_version"] = None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                env["nvidia_driver"] = out.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            env["nvidia_driver"] = None
    try:
        import transformers as tr

        env["transformers"] = tr.__version__
    except ImportError:
        env["transformers"] = None
    try:
        import safetensors

        env["safetensors"] = getattr(safetensors, "__version__", "unknown")
    except ImportError:
        env["safetensors"] = None
    return env
