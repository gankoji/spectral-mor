# Cloud GPU runbook — PGD / Gemma evaluation (avoid local RAM thrash)

## Why local runs hurt on a 18 GB unified machine

`google/gemma-4-E2B-it` is large enough that **one** full load in `bfloat16` / `float16` already consumes a big fraction of unified memory. The unified harness with `--mode all` loads the model **three times in sequence** (baseline → dense PGD → native PGD). Each load:

- Materializes weights in PyTorch (device CPU or MPS),
- Keeps **additional** copies transiently during `from_pretrained` / tensor moves,
- Runs **PGD in NumPy on CPU** (`float32` weight matrices) for substituted layers — another large allocation for `down_proj` (e.g. 1536×12288),

so **peak resident set can approach 2× a single model footprint**, especially with evaluation, drift captures, and OS cache. That pushes macOS into heavy swap and makes the whole machine unusable.

**Rule:** Do not run full multi-arm Gemma sweeps on a small unified-memory laptop. Use a **discrete-GPU VM with enough GPU VRAM + enough host RAM** (or run **one arm per job** locally if you must stay on the laptop).

---

## Target environment

| Requirement | Practical guidance |
|-------------|-------------------|
| **GPU** | NVIDIA datacenter or consumer GPU with **≥ 24 GB VRAM** for headroom on E2B-class models in bf16/fp16 (16 GB can work but is tight with long context + hooks). **A100 40/80 GB, L4 24 GB, A10 24 GB** are common choices. |
| **Host RAM** | **≥ 64 GB** recommended if you keep PGD on CPU in float32 for large matrices; **32 GB** may work if you run **one mode per process** and avoid parallel jobs. |
| **Disk** | **≥ 50 GB** free for HF cache + outputs (model shards + results). |
| **OS** | Linux + NVIDIA driver + CUDA stack matching your PyTorch wheel. |
| **Network** | Stable egress to Hugging Face (first run downloads weights; use `huggingface-cli login` or `HF_TOKEN`). |

Providers that fit this pattern well: **AWS** (g5, g6, p4/p5), **GCP** (A2, A3, L4), **Azure** (NCads), **Lambda**, **RunPod**, **CoreWeave**, etc. Pick a single-GPU instance with the VRAM row above.

---

## Repo setup on the VM

```bash
# Example: Ubuntu 22.04+, NVIDIA driver already installed
sudo apt update && sudo apt install -y git python3-venv

git clone <your-repo-url> spectral-mor
cd spectral-mor
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match CUDA to VM
pip install transformers safetensors accelerate numpy

export HF_TOKEN=...   # if the model is gated
huggingface-cli login --token "$HF_TOKEN"   # optional; token env is enough for many flows
```

Use `mor/requirements.txt` as a baseline, but **install `torch` with the CUDA build that matches the VM** (PyTorch “Get Started” matrix).

---

## How to run experiments safely

### 1. Prefer CUDA, not CPU

```bash
cd mor
source ../.venv/bin/activate

python compressed_inference_harness.py \
  --model google/gemma-4-E2B-it \
  --device cuda \
  --torch-dtype bfloat16 \
  --apply-chat-template \
  ...
```

This keeps the model on GPU and avoids pinning multiple full copies in host RAM.

### 2. Avoid triple load unless you have the RAM budget

`--mode all` is convenient but loads the checkpoint **once per arm**. On memory-constrained hosts, run **three separate jobs** instead:

```bash
python compressed_inference_harness.py --mode baseline  --output-json baseline.json  ...
python compressed_inference_harness.py --mode dense_pgd --output-json dense.json   ...
python compressed_inference_harness.py --mode native_pgd --output-json native.json ...
```

Between jobs, the process exits and **all** PyTorch memory is released.

### 3. Start without decode / long context

- Omit `--max-new-tokens` (or set `0`) until core NLL + drift look sane.
- Use `--max-length` that matches your VRAM budget (e.g. 128–256 first).

### 4. PGD cost is still on CPU (today)

`pgd_decompose` is NumPy on **CPU float32**. For `down_proj` at layer 17 that is a large matrix; expect **tens of seconds to a few minutes per rank budget** depending on `--pgd-iters` and CPU core count. The cloud VM should have **enough DRAM** so this does not compete with swap.

Optional future improvement (not required for this runbook): run PGD on GPU or chunked / lower precision — out of scope here.

### 5. Apple MPS (Mac) vs cloud CUDA

MPS is useful for **small** models; for Gemma-scale models, **CUDA VMs are the default** — better supported kernels, predictable VRAM, and no unified-memory contention with the OS.

---

## Suggested first cloud experiment (copy-paste)

After cache is warm:

```bash
cd mor
source ../.venv/bin/activate

python compressed_inference_harness.py \
  --model google/gemma-4-E2B-it \
  --device cuda \
  --torch-dtype bfloat16 \
  --mode baseline \
  --use-default-prompt-set \
  --apply-chat-template \
  --max-length 256 \
  --prefill-warmup 1 \
  --prefill-repeats 3 \
  --max-new-tokens 0 \
  --output-json results_baseline_cuda.json
```

Then repeat with `--mode dense_pgd` / `native_pgd`, `--layers 17`, `--projections down_proj`, `--rank 128`, `--drift-layers 17`, `--pgd-iters 12` (adjust as needed).

---

## Artifacts to save

- JSON outputs from the harness (`--output-json`).
- `run_environment` block inside the JSON (library versions, CUDA availability).
- Instance type, region, GPU SKU, and **driver / CUDA** version in a short `README` or experiment log.

---

## Cost hygiene

- **Stop or terminate** the VM when finished; GPU idle hours add up.
- Use **spot / preemptible** only if you checkpoint results frequently (PGD substitution is long enough to lose work).
- Keep **one experiment process per GPU** unless you know VRAM fits.

---

## Relation to repo scripts

| Script | Role on cloud |
|--------|----------------|
| `mor/compressed_inference_harness.py` | Main baseline / dense / native comparison (use `--device cuda`). |
| `mor/pgd_fidelity_harness.py` | Rank sweeps on mmap’d safetensors (CPU-heavy; needs `--safetensors-path` or env). |
| `mor/run_pgd_perplexity.py` | Lighter single-driver alternative. |

This runbook intentionally does **not** prescribe a single cloud vendor; pick one GPU SKU that satisfies the VRAM + RAM rows above, then follow the same commands.

---

## OpenTofu (or Terraform) — provision GPU VMs without clicking

OpenTofu speaks the same provider protocol as Terraform: use `required_providers` and the [OpenTofu Registry](https://search.opentofu.org/providers) (or HashiCorp’s registry) as usual.

### Where IaC is “boring and well supported” (recommended)

These have **official, maintained providers** and extensive examples for GPU instances:

| Provider | Typical approach | Notes |
|----------|------------------|--------|
| **AWS** | `aws_instance` with `g5`, `g6`, `p4`, `p5` (etc.), or Launch Template + ASG | Quotas and capacity can block specific AZs; use capacity reservations for serious runs. |
| **Google Cloud** | `google_compute_instance` with `guest_accelerator` (e.g. L4, A100) | Often need to request GPU quota in the project. |
| **Microsoft Azure** | `azurerm_linux_virtual_machine` (or VMSS) with NC/ND/H-series SKUs | GPU SKUs are region-specific. |
| **DigitalOcean** | `digitalocean_droplet` with a GPU size slug (e.g. `gpu-l40sx1-48gb`) and `gpu-h100x1-base` image | Official provider; this repo includes a starter stack under `infra/digitalocean/` (see its README). |

**None of these automatically “run your tests”** — they provision the machine (and network, disk, IAM). You still attach bootstrap via **cloud-init** (`user_data`), a **startup script** resource, or a follow-on tool (Ansible, `remote-exec`, GitHub Actions SSH, etc.) to `git clone`, `pip install`, and invoke `compressed_inference_harness.py`.

### GPU-focused clouds (possible, often community providers)

Some GPU-only vendors have **third-party Terraform/OpenTofu providers** (e.g. community providers for **Lambda Labs**). Those can work with OpenTofu but vary in freshness and coverage — treat as “evaluate before betting the farm.” Always check the provider’s registry page and last commit date.

### Practical pattern

1. **Tofu**: VPC/subnet (if needed), security group (SSH from your IP), key pair, GPU instance, disk size, optional public IP.  
2. **cloud-init**: install NVIDIA driver stack *or* start from a **GPU-optimized image** the cloud publishes (simpler).  
3. **One script**: clone repo, venv, `pip install torch` (CUDA wheel matching the image), `pip install -r mor/requirements.txt`, `huggingface-cli login`, run harness, write JSON to object storage or `scp` down.

That keeps **infrastructure** declarative while **workload bootstrap** stays a small, versioned shell script next to your `.tf` / `.tofu` files.
