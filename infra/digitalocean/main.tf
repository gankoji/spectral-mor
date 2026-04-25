locals {
  user_data_effective = length(trimspace(var.user_data)) > 0 ? var.user_data : null
}

resource "digitalocean_droplet" "gpu" {
  name       = var.droplet_name
  region     = var.region
  size       = var.droplet_size_slug
  image      = var.image_slug
  ssh_keys   = var.ssh_keys
  tags       = var.tags
  monitoring = var.monitoring
  user_data  = local.user_data_effective

  # GPU Droplets can take longer to provision than CPU Droplets.
  timeouts {
    create = "30m"
  }
}

resource "digitalocean_firewall" "gpu" {
  name = "${var.droplet_name}-ssh"

  droplet_ids = [digitalocean_droplet.gpu.id]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.admin_ssh_cidrs
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
