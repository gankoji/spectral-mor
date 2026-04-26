output "pod_name" {
  description = "Requested Runpod Pod name."
  value       = var.pod_name
}

output "gpu_type_ids" {
  description = "Preferred GPU type IDs used for placement."
  value       = var.gpu_type_ids
}

output "runpod_create_response_path" {
  description = "Local file containing the latest Runpod create API response."
  value       = "${path.module}/runpod-create-response.json"
}

output "runpod_pod_id_path" {
  description = "Local file containing the created Pod ID after apply."
  value       = "${path.module}/runpod-pod-id.txt"
}

output "runpod_console_hint" {
  description = "Runpod console URL for Pod SSH details and lifecycle management."
  value       = "https://console.runpod.io/pods"
}

output "bootstrap_results_dir" {
  description = "Results directory inside the Pod."
  value       = "${var.volume_mount_path}/spectral-mor-results"
}

output "bootstrap_logs_dir" {
  description = "Logs directory inside the Pod."
  value       = "${var.volume_mount_path}/spectral-mor-logs"
}
