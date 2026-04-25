# DigitalOcean GPU Droplet — Terraform / OpenTofu

Provisions a **GPU Droplet** with DigitalOcean’s **NVIDIA AI/ML-ready** image (`gpu-h100x1-base` — correct for all self-serve **single-GPU NVIDIA** sizes per [recommended GPU setup](https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/)) and a **firewall** that only allows SSH from your CIDRs.

This repo’s experiment commands are documented in `plans/CLOUD_GPU_EXPERIMENT_RUNBOOK.md` and run under `mor/` (e.g. `compressed_inference_harness.py` with `--device cuda`).

## Prerequisites

1. **DigitalOcean account** with **GPU Droplet** access and quota in the chosen **region**.
2. **Personal access token** with write scope: export as `DIGITALOCEAN_TOKEN` (or configure the provider).
3. **SSH key** uploaded to DO: **Account → Security → SSH keys**. Note each key’s **fingerprint** or **ID** (`doctl compute ssh-key list`).
4. **Terraform** ≥ 1.3 or **OpenTofu** (drop-in): same files, use `tofu init` / `tofu apply` instead of `terraform`.

## GPU size slugs (examples)

From DigitalOcean **Droplet features** (NVIDIA self-serve):

| Slug | GPU VRAM | Droplet RAM | Notes |
|------|-----------|--------------|--------|
| `gpu-l40sx1-48gb` | 48 GB | 64 GiB | Good default for Gemma-class bf16/fp16 |
| `gpu-6000adax1-48gb` | 48 GB | 64 GiB | Alternative |
| `gpu-4000adax1-20gb` | 20 GB | 32 GiB | Tighter; may be marginal for larger models |
| `gpu-h100x1-80gb` | 80 GB | 240 GiB | Heavier jobs |

AMD plans use image `gpu-amd-base` and slugs such as `gpu-mi300x1-192gb` (not covered by the default `image_slug`).

## Quick start

```bash
cd infra/digitalocean
export DIGITALOCEAN_TOKEN="dop_v1_..."

cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: ssh_keys, admin_ssh_cidrs, region, droplet_size_slug

terraform init
terraform plan
terraform apply
```

Outputs include `droplet_ipv4` and `ssh_hint`.

## After SSH

The AI/ML image includes CUDA/driver stack; install PyTorch **matching that CUDA** and run from the repo:

```bash
git clone <your-fork-or-repo-url> spectral-mor
cd spectral-mor
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# Install torch from https://pytorch.org/get-started/locally/ for your CUDA version
pip install transformers safetensors accelerate numpy

export HF_TOKEN=...   # if needed
cd mor
python compressed_inference_harness.py --device cuda --torch-dtype bfloat16 ...
```

## Security

- **Do not** set `admin_ssh_cidrs` to `0.0.0.0/0` unless you understand the risk.
- Keep `terraform.tfvars` out of git (see root `.gitignore`).

## Destroy

```bash
terraform destroy
```
