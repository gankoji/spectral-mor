output "droplet_id" {
  description = "DigitalOcean Droplet ID."
  value       = digitalocean_droplet.gpu.id
}

output "droplet_ipv4" {
  description = "Public IPv4 address for SSH and experiments."
  value       = digitalocean_droplet.gpu.ipv4_address
}

output "droplet_urn" {
  description = "Droplet URN (for IAM / automation)."
  value       = digitalocean_droplet.gpu.urn
}

output "ssh_hint" {
  description = "Typical SSH command (NVIDIA AI/ML images use root)."
  value       = "ssh root@${digitalocean_droplet.gpu.ipv4_address}"
}
