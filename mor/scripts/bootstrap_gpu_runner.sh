#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

bool_is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SPECTRAL_MOR_REPO_URL="${SPECTRAL_MOR_REPO_URL:-https://github.com/gankoji/spectral-mor.git}"
SPECTRAL_MOR_BRANCH="${SPECTRAL_MOR_BRANCH:-main}"
SPECTRAL_MOR_REPO_DIR="${SPECTRAL_MOR_REPO_DIR:-$DEFAULT_REPO_DIR}"
SPECTRAL_MOR_RESULTS_DIR="${SPECTRAL_MOR_RESULTS_DIR:-/workspace/spectral-mor-results}"
SPECTRAL_MOR_LOG_DIR="${SPECTRAL_MOR_LOG_DIR:-/workspace/spectral-mor-logs}"
SPECTRAL_MOR_RUN_JOBS="${SPECTRAL_MOR_RUN_JOBS:-0}"
SPECTRAL_MOR_RUN_BASELINE="${SPECTRAL_MOR_RUN_BASELINE:-1}"
SPECTRAL_MOR_RUN_DENSE_PGD="${SPECTRAL_MOR_RUN_DENSE_PGD:-1}"
SPECTRAL_MOR_RUN_NATIVE_PGD="${SPECTRAL_MOR_RUN_NATIVE_PGD:-1}"
SPECTRAL_MOR_MODEL="${SPECTRAL_MOR_MODEL:-google/gemma-4-E2B-it}"
SPECTRAL_MOR_TORCH_DTYPE="${SPECTRAL_MOR_TORCH_DTYPE:-bfloat16}"
SPECTRAL_MOR_MAX_LENGTH="${SPECTRAL_MOR_MAX_LENGTH:-256}"
SPECTRAL_MOR_PREFILL_WARMUP="${SPECTRAL_MOR_PREFILL_WARMUP:-1}"
SPECTRAL_MOR_PREFILL_REPEATS="${SPECTRAL_MOR_PREFILL_REPEATS:-3}"
SPECTRAL_MOR_MAX_NEW_TOKENS="${SPECTRAL_MOR_MAX_NEW_TOKENS:-0}"
SPECTRAL_MOR_LAYERS="${SPECTRAL_MOR_LAYERS:-17}"
SPECTRAL_MOR_PROJECTIONS="${SPECTRAL_MOR_PROJECTIONS:-down_proj}"
SPECTRAL_MOR_RANK="${SPECTRAL_MOR_RANK:-128}"
SPECTRAL_MOR_DRIFT_LAYERS="${SPECTRAL_MOR_DRIFT_LAYERS:-17}"
SPECTRAL_MOR_PGD_ITERS="${SPECTRAL_MOR_PGD_ITERS:-12}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
SPECTRAL_MOR_INSTALL_SYSTEM_PACKAGES="${SPECTRAL_MOR_INSTALL_SYSTEM_PACKAGES:-auto}"
SPECTRAL_MOR_FORCE_GIT_PULL="${SPECTRAL_MOR_FORCE_GIT_PULL:-1}"
SPECTRAL_MOR_VENV_DIR="${SPECTRAL_MOR_VENV_DIR:-$SPECTRAL_MOR_REPO_DIR/.venv}"

mkdir -p "$SPECTRAL_MOR_LOG_DIR" "$SPECTRAL_MOR_RESULTS_DIR"
exec > >(tee -a "$SPECTRAL_MOR_LOG_DIR/bootstrap-and-runs.log") 2>&1

install_system_packages() {
  local mode="$SPECTRAL_MOR_INSTALL_SYSTEM_PACKAGES"
  if [ "$mode" = "0" ] || [ "$mode" = "false" ]; then
    log "Skipping system package installation."
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -ne 0 ]; then
      if [ "$mode" = "auto" ]; then
        log "Not root; skipping apt package installation."
        return 0
      fi
      log "Cannot install apt packages without root."
      return 1
    fi
    log "Installing system packages with apt."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y git python3-venv python3-pip jq tmux nvtop ca-certificates
  else
    log "No apt-get detected; assuming required system packages are already present."
  fi
}

prepare_repo() {
  if [ -d "$SPECTRAL_MOR_REPO_DIR/.git" ]; then
    log "Using existing repo at $SPECTRAL_MOR_REPO_DIR"
  else
    log "Cloning $SPECTRAL_MOR_REPO_URL branch $SPECTRAL_MOR_BRANCH into $SPECTRAL_MOR_REPO_DIR"
    mkdir -p "$(dirname "$SPECTRAL_MOR_REPO_DIR")"
    git clone --branch "$SPECTRAL_MOR_BRANCH" "$SPECTRAL_MOR_REPO_URL" "$SPECTRAL_MOR_REPO_DIR"
  fi

  cd "$SPECTRAL_MOR_REPO_DIR"
  git status --short || true
  if bool_is_true "$SPECTRAL_MOR_FORCE_GIT_PULL"; then
    git pull --ff-only || log "git pull failed or was unnecessary; continuing with current checkout."
  fi
}

prepare_python() {
  cd "$SPECTRAL_MOR_REPO_DIR"
  log "Creating/updating virtualenv at $SPECTRAL_MOR_VENV_DIR"
  python3 -m venv "$SPECTRAL_MOR_VENV_DIR"
  # shellcheck disable=SC1091
  . "$SPECTRAL_MOR_VENV_DIR/bin/activate"
  python -m pip install -U pip wheel setuptools
  python -m pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX_URL"
  python -m pip install -r mor/requirements.txt accelerate
  python -m pip install -U transformers huggingface-hub

  if [ -n "${HF_TOKEN:-}" ]; then
    python -m pip install -U huggingface_hub
    huggingface-cli login --token "$HF_TOKEN" || log "Hugging Face CLI login failed; continuing."
  fi
}

verify_cuda() {
  log "Checking NVIDIA/CUDA environment."
  nvidia-smi || true
  python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
PY
}

run_job() {
  local name="$1"
  shift
  log "Starting $name"
  "$@" 2>&1 | tee "$SPECTRAL_MOR_RESULTS_DIR/$name.log"
  log "Finished $name"
}

run_experiments() {
  cd "$SPECTRAL_MOR_REPO_DIR/mor"
  # shellcheck disable=SC1091
  . "$SPECTRAL_MOR_VENV_DIR/bin/activate"

  local common=(
    --model "$SPECTRAL_MOR_MODEL"
    --device cuda
    --torch-dtype "$SPECTRAL_MOR_TORCH_DTYPE"
    --use-default-prompt-set
    --apply-chat-template
    --max-length "$SPECTRAL_MOR_MAX_LENGTH"
    --prefill-warmup "$SPECTRAL_MOR_PREFILL_WARMUP"
    --prefill-repeats "$SPECTRAL_MOR_PREFILL_REPEATS"
    --max-new-tokens "$SPECTRAL_MOR_MAX_NEW_TOKENS"
  )

  if bool_is_true "$SPECTRAL_MOR_RUN_BASELINE"; then
    run_job baseline_cuda python compressed_inference_harness.py \
      "${common[@]}" \
      --mode baseline \
      --output-json "$SPECTRAL_MOR_RESULTS_DIR/results_baseline_cuda.json"
  fi

  if bool_is_true "$SPECTRAL_MOR_RUN_DENSE_PGD"; then
    run_job B1_dense_pgd_cuda python compressed_inference_harness.py \
      "${common[@]}" \
      --mode dense_pgd \
      --layers "$SPECTRAL_MOR_LAYERS" \
      --projections "$SPECTRAL_MOR_PROJECTIONS" \
      --rank "$SPECTRAL_MOR_RANK" \
      --drift-layers "$SPECTRAL_MOR_DRIFT_LAYERS" \
      --pgd-iters "$SPECTRAL_MOR_PGD_ITERS" \
      --output-json "$SPECTRAL_MOR_RESULTS_DIR/results_B1_cuda.json"
  fi

  if bool_is_true "$SPECTRAL_MOR_RUN_NATIVE_PGD"; then
    run_job C1_native_pgd_cuda python compressed_inference_harness.py \
      "${common[@]}" \
      --mode native_pgd \
      --layers "$SPECTRAL_MOR_LAYERS" \
      --projections "$SPECTRAL_MOR_PROJECTIONS" \
      --rank "$SPECTRAL_MOR_RANK" \
      --drift-layers "$SPECTRAL_MOR_DRIFT_LAYERS" \
      --pgd-iters "$SPECTRAL_MOR_PGD_ITERS" \
      --output-json "$SPECTRAL_MOR_RESULTS_DIR/results_C1_cuda.json"
  fi
}

main() {
  log "Starting spectral-mor GPU bootstrap."
  install_system_packages
  prepare_repo
  prepare_python
  verify_cuda

  if ! bool_is_true "$SPECTRAL_MOR_RUN_JOBS"; then
    log "SPECTRAL_MOR_RUN_JOBS is not true; setup complete without launching experiments."
    log "Set SPECTRAL_MOR_RUN_JOBS=1 and rerun $0 to launch experiments."
    exit 0
  fi

  run_experiments
  log "All requested spectral-mor jobs complete. Results in $SPECTRAL_MOR_RESULTS_DIR"
}

main "$@"
