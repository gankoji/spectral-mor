variable "runpod_api_key" {
  type        = string
  description = "Runpod API key. Prefer setting TF_VAR_runpod_api_key."
  default     = null
  sensitive   = true
}

variable "pod_name" {
  type        = string
  description = "Runpod Pod name."
  default     = "spectral-mor-gpu-experiments"
}

variable "gpu_type_ids" {
  type        = list(string)
  description = "Preferred Runpod GPU type IDs, in order. Runpod will place on available capacity."
  default = [
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100 80GB PCIe",
  ]
}

variable "gpu_count" {
  type        = number
  description = "GPU count for the Pod."
  default     = 1
}

variable "gpu_type_priority" {
  type        = string
  description = "Runpod GPU selection priority: availability or custom."
  default     = "availability"

  validation {
    condition     = contains(["availability", "custom"], var.gpu_type_priority)
    error_message = "gpu_type_priority must be availability or custom."
  }
}

variable "cloud_type" {
  type        = string
  description = "Runpod cloud type: SECURE or COMMUNITY."
  default     = "SECURE"

  validation {
    condition     = contains(["SECURE", "COMMUNITY"], var.cloud_type)
    error_message = "cloud_type must be SECURE or COMMUNITY."
  }
}

variable "data_center_ids" {
  type        = list(string)
  description = "Optional Runpod data center IDs to restrict placement. Leave empty to allow any matching data center."
  default     = []
}

variable "image_name" {
  type        = string
  description = "Container image for the Pod. PyTorch CUDA images work well for this project."
  default     = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
}

variable "container_disk_gb" {
  type        = number
  description = "Container disk size in GB."
  default     = 80
}

variable "volume_gb" {
  type        = number
  description = "Persistent Pod volume size in GB."
  default     = 80
}

variable "volume_mount_path" {
  type        = string
  description = "Persistent volume mount path inside the Pod."
  default     = "/workspace"
}

variable "ports" {
  type        = list(string)
  description = "Runpod ports to expose. Keep SSH exposed."
  default     = ["22/tcp", "8888/http"]
}

variable "support_public_ip" {
  type        = bool
  description = "Request public IP support, especially useful for Community Cloud Pods."
  default     = true
}

variable "interruptible" {
  type        = bool
  description = "Use interruptible/spot capacity for lower cost but less reliability."
  default     = false
}

variable "locked" {
  type        = bool
  description = "Lock the Pod to prevent accidental stop/reset through Runpod."
  default     = false
}

variable "min_ram_per_gpu" {
  type        = number
  description = "Minimum host RAM per GPU in GB."
  default     = 48
}

variable "min_vcpu_per_gpu" {
  type        = number
  description = "Minimum vCPU count per GPU."
  default     = 8
}

variable "repo_url" {
  type        = string
  description = "Git repository to clone on the Pod."
  default     = "https://github.com/gankoji/spectral-mor.git"
}

variable "repo_branch" {
  type        = string
  description = "Git branch to use on the Pod."
  default     = "main"
}

variable "run_jobs" {
  type        = bool
  description = "If true, launch baseline, dense PGD, and native PGD after setup. Keep false for first provisioning if you want to inspect the Pod before running."
  default     = false
}

variable "hf_token" {
  type        = string
  description = "Optional Hugging Face token for gated model access. Prefer setting TF_VAR_hf_token."
  default     = ""
  sensitive   = true
}

variable "torch_cuda_index_url" {
  type        = string
  description = "PyTorch CUDA wheel index URL."
  default     = "https://download.pytorch.org/whl/cu124"
}
