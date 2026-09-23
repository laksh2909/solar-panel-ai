#!/usr/bin/env bash
# ==============================================================================
# Solar Panel AI Inspection System — EC2 Bootstrap Script
# Run once as EC2 User Data (or manually as ubuntu user on a fresh instance).
#
# Installs:
#   - Docker Engine (latest, from apt.docker.com)
#   - Docker Compose plugin (v2)
#   - Required system utilities (git, curl, rsync, unzip)
#
# Usage as EC2 User Data:
#   Paste this script into the "User data" field when launching an EC2 instance.
#   It runs as root on first boot.
#
# Usage as manual provisioning (SSH into EC2):
#   chmod +x ec2_bootstrap.sh
#   sudo bash ec2_bootstrap.sh
# ==============================================================================

set -euo pipefail

LOG_FILE="/var/log/solar-panel-ai-bootstrap.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "====================================================="
echo " Solar Panel AI EC2 Bootstrap — $(date -u)"
echo "====================================================="

# ------------------------------------------------------------------
# 1. Update apt package index
# ------------------------------------------------------------------
echo "[1/6] Updating apt package index..."
apt-get update -y

# ------------------------------------------------------------------
# 2. Install system utilities
# ------------------------------------------------------------------
echo "[2/6] Installing system utilities..."
apt-get install -y \
    curl \
    git \
    rsync \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release

# ------------------------------------------------------------------
# 3. Install Docker Engine
# ------------------------------------------------------------------
echo "[3/6] Installing Docker Engine..."

# Remove old conflicting packages if any
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
    apt-get remove -y "$pkg" 2>/dev/null || true
done

# Add Docker's official GPG key and apt repository
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# ------------------------------------------------------------------
# 4. Enable and start Docker service
# ------------------------------------------------------------------
echo "[4/6] Enabling Docker service..."
systemctl enable docker
systemctl start docker

# ------------------------------------------------------------------
# 5. Add ubuntu user to docker group (avoids sudo for docker commands)
# ------------------------------------------------------------------
echo "[5/6] Adding ubuntu user to docker group..."
usermod -aG docker ubuntu

# ------------------------------------------------------------------
# 6. Verify installation
# ------------------------------------------------------------------
echo "[6/6] Verifying installation..."
docker --version
docker compose version

echo ""
echo "====================================================="
echo " Bootstrap complete — $(date -u)"
echo " Docker Engine and Compose plugin are ready."
echo " Log: ${LOG_FILE}"
echo "====================================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Transfer application files to /home/ubuntu/solar-panel-ai/"
echo "     (Use rsync, scp, or git clone from your dev machine)"
echo "  2. Create /home/ubuntu/solar-panel-ai/.env with your EC2 settings"
echo "  3. Run: cd /home/ubuntu/solar-panel-ai && bash scripts/deploy.sh"
echo ""
