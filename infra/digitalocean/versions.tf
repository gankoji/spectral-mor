terraform {
  required_version = ">= 1.3"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.50"
    }
  }
}

# Authenticate with DIGITALOCEAN_TOKEN in the environment, or set provider token explicitly.
provider "digitalocean" {}
