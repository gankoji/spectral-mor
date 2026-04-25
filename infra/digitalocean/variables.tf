variable "region" {
  type        = string
  description = "DigitalOcean region slug (GPU capacity varies by region; confirm in the control panel or API)."
  default     = "nyc1"
}

variable "droplet_name" {
  type        = string
  description = "Name of the GPU Droplet."
  default     = "spectral-mor-gpu-experiments"
}

variable "droplet_size_slug" {
  type        = string
  description = <<-EOT
    GPU Droplet plan slug (see DigitalOcean docs: Droplet Features → GPU Droplets).
    Examples: gpu-l40sx1-48gb (48 GB VRAM), gpu-6000adax1-48gb, gpu-4000adax1-20gb (tighter VRAM).
  EOT
  default     = "gpu-l40sx1-48gb"
}

variable "image_slug" {
  type        = string
  description = <<-EOT
    NVIDIA single-GPU AI/ML-ready image. Per DigitalOcean docs, use gpu-h100x1-base for all
    self-serve single-GPU NVIDIA plans (not only H100). For AMD GPU plans, use gpu-amd-base.
  EOT
  default     = "gpu-h100x1-base"
}

variable "ssh_keys" {
  type        = list(string)
  description = "SSH public key IDs or fingerprints already uploaded to your DO account (Account → Security → SSH keys)."
}

variable "admin_ssh_cidrs" {
  type        = list(string)
  description = "CIDR blocks allowed to reach SSH (port 22). Restrict to your public IP /32."
}

variable "tags" {
  type        = list(string)
  description = "Optional Droplet tags."
  default     = ["spectral-mor", "gpu-experiments"]
}

variable "monitoring" {
  type        = bool
  description = "Enable DO metrics agent (recommended for GPU observability where supported)."
  default     = true
}

variable "user_data" {
  type        = string
  description = "Optional cloud-init user_data (plain text). Leave empty to skip."
  default     = ""
  sensitive   = true
}
