locals {
  runpod_api_key_effective = var.runpod_api_key == null ? "" : var.runpod_api_key

  pod_env = {
    SPECTRAL_MOR_REPO_URL       = var.repo_url
    SPECTRAL_MOR_BRANCH         = var.repo_branch
    SPECTRAL_MOR_REPO_DIR       = "${var.volume_mount_path}/spectral-mor"
    SPECTRAL_MOR_RESULTS_DIR    = "${var.volume_mount_path}/spectral-mor-results"
    SPECTRAL_MOR_LOG_DIR        = "${var.volume_mount_path}/spectral-mor-logs"
    SPECTRAL_MOR_RUN_JOBS       = var.run_jobs ? "1" : "0"
    SPECTRAL_MOR_FORCE_GIT_PULL = "1"
    TORCH_CUDA_INDEX_URL        = var.torch_cuda_index_url
    HF_TOKEN                    = var.hf_token
    HF_HOME                     = "${var.volume_mount_path}/.cache/huggingface"
    TRANSFORMERS_CACHE          = "${var.volume_mount_path}/.cache/huggingface/transformers"
  }

  start_command = "apt-get update && apt-get install -y git ca-certificates python3-venv python3-pip jq tmux nvtop openssh-server && service ssh start || true; mkdir -p ${var.volume_mount_path}; if [ ! -d ${var.volume_mount_path}/spectral-mor/.git ]; then git clone --branch ${var.repo_branch} ${var.repo_url} ${var.volume_mount_path}/spectral-mor; fi; cd ${var.volume_mount_path}/spectral-mor && chmod +x mor/scripts/bootstrap_gpu_runner.sh && bash mor/scripts/bootstrap_gpu_runner.sh; sleep infinity"

  data_center_payload = {
    dataCenterIds      = var.data_center_ids
    dataCenterPriority = "availability"
  }

  create_pod_payload = merge(
    {
      name              = var.pod_name
      cloudType         = var.cloud_type
      computeType       = "GPU"
      gpuCount          = var.gpu_count
      gpuTypeIds        = var.gpu_type_ids
      gpuTypePriority   = var.gpu_type_priority
      imageName         = var.image_name
      containerDiskInGb = var.container_disk_gb
      volumeInGb        = var.volume_gb
      volumeMountPath   = var.volume_mount_path
      minRAMPerGPU      = var.min_ram_per_gpu
      minVCPUPerGPU     = var.min_vcpu_per_gpu
      ports             = var.ports
      supportPublicIp   = var.support_public_ip
      interruptible     = var.interruptible
      locked            = var.locked
      env               = local.pod_env
      dockerEntrypoint  = ["/bin/bash", "-lc"]
      dockerStartCmd    = [local.start_command]
    },
    local.data_center_payload
  )
}

check "runpod_api_key_set" {
  assert {
    condition     = length(trimspace(local.runpod_api_key_effective)) > 0
    error_message = "Set TF_VAR_runpod_api_key before applying."
  }
}

resource "terraform_data" "runpod_pod" {
  input = {
    pod_name          = var.pod_name
    gpu_type_ids      = var.gpu_type_ids
    gpu_count         = var.gpu_count
    gpu_type_priority = var.gpu_type_priority
    cloud_type        = var.cloud_type
    image_name        = var.image_name
    container_disk_gb = var.container_disk_gb
    volume_gb         = var.volume_gb
    repo_url          = var.repo_url
    repo_branch       = var.repo_branch
    run_jobs          = var.run_jobs
    pod_id_path       = "${path.module}/runpod-pod-id.txt"
    response_path     = "${path.module}/runpod-create-response.json"
  }

  provisioner "local-exec" {
    interpreter = ["/usr/bin/env", "bash", "-lc"]
    environment = {
      RUNPOD_API_KEY = local.runpod_api_key_effective
      REQUEST_JSON   = jsonencode(local.create_pod_payload)
      RESPONSE_PATH  = self.input.response_path
      POD_ID_PATH    = self.input.pod_id_path
    }
    command = <<-EOT
      set -euo pipefail
      response=$(curl -fsS -X POST https://rest.runpod.io/v1/pods \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        --data-binary "$REQUEST_JSON")
      printf '%s\n' "$response" > "$RESPONSE_PATH"
      RESPONSE_PATH="$RESPONSE_PATH" POD_ID_PATH="$POD_ID_PATH" python3 - <<'PY'
import json
import os
from pathlib import Path

response_path = Path(os.environ["RESPONSE_PATH"])
pod_id_path = Path(os.environ["POD_ID_PATH"])
response = json.loads(response_path.read_text())
if response.get("errors"):
    raise SystemExit(json.dumps(response["errors"], indent=2))
pod_id = response.get("id")
if not pod_id:
    raise SystemExit(f"Runpod response did not contain a pod id: {json.dumps(response, indent=2)}")
pod_id_path.write_text(pod_id + "\n")
print(json.dumps({
    "id": pod_id,
    "name": response.get("name"),
    "desiredStatus": response.get("desiredStatus"),
    "publicIp": response.get("publicIp"),
    "portMappings": response.get("portMappings"),
    "gpu": response.get("gpu"),
}, indent=2))
PY
    EOT
  }

  provisioner "local-exec" {
    when        = destroy
    interpreter = ["/usr/bin/env", "bash", "-lc"]
    environment = {
      POD_ID_PATH = self.input.pod_id_path
    }
    command = <<-EOT
      set -euo pipefail
      if [ ! -f "$POD_ID_PATH" ]; then
        echo 'No runpod-pod-id.txt found; skipping Runpod deletion.'
        exit 0
      fi
      if [ -z "$${RUNPOD_API_KEY:-}" ]; then
        echo 'RUNPOD_API_KEY is not set; cannot delete Runpod Pod automatically.'
        echo "Delete pod $(cat "$POD_ID_PATH") manually in the Runpod console."
        exit 0
      fi
      pod_id=$(cat "$POD_ID_PATH")
      curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$pod_id" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" || true
      rm -f "$POD_ID_PATH"
    EOT
  }
}
