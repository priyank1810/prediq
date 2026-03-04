#!/bin/bash
# setup.sh — One-time setup for AWS Lightsail (Ubuntu 22.04)
# Usage: ssh into instance, then: bash deploy/setup.sh
set -euo pipefail

APP_USER="ubuntu"
APP_DIR="/home/${APP_USER}/stock-tracker"
VENV_DIR="${APP_DIR}/venv"
REPO_URL="https://github.com/priyank1810/prediq.git"
SWAP_SIZE="2G"

echo "=== Stock Tracker — Lightsail Setup ==="

# ── 1. System packages ──────────────────────────────────────────────
echo "[1/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    nginx \
    git \
    build-essential \
    libffi-dev libssl-dev \
    ufw

# ── 2. Clone repo ───────────────────────────────────────────────────
echo "[2/8] Cloning repository..."
if [ -d "${APP_DIR}" ]; then
    echo "  Directory ${APP_DIR} already exists, pulling latest..."
    cd "${APP_DIR}" && git pull
else
    git clone "${REPO_URL}" "${APP_DIR}"
fi
cd "${APP_DIR}"

# ── 3. Python virtualenv + deps ─────────────────────────────────────
echo "[3/8] Creating virtualenv and installing dependencies..."
python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools -q
pip install -r requirements.txt -q
pip install gunicorn -q

# Install cmdstan binary (required by Prophet for Bayesian inference)
echo "  Installing cmdstan for Prophet..."
python -m cmdstanpy.install_cmdstan --lightweight 2>/dev/null || true

# ── 4. Swap file (2 GB) ─────────────────────────────────────────────
echo "[4/8] Setting up ${SWAP_SIZE} swap file..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l "${SWAP_SIZE}" /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    # Tune swappiness — prefer RAM, only swap when needed
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    sudo sysctl vm.swappiness=10
    echo "  Swap enabled."
else
    echo "  Swap already exists, skipping."
fi

# ── 5. Environment file ─────────────────────────────────────────────
echo "[5/8] Setting up environment file..."
if [ ! -f "${APP_DIR}/.env" ]; then
    cp "${APP_DIR}/deploy/.env.example" "${APP_DIR}/.env"
    # Generate a random SECRET_KEY
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s/^SECRET_KEY=.*/SECRET_KEY=${SECRET}/" "${APP_DIR}/.env"
    echo "  .env created — EDIT IT with your Angel One credentials!"
else
    echo "  .env already exists, skipping."
fi

# ── 6. Data directories ─────────────────────────────────────────────
echo "[6/8] Creating data directories..."
mkdir -p "${APP_DIR}/data"
mkdir -p "${APP_DIR}/saved_models"

# ── 7. systemd services ─────────────────────────────────────────────
echo "[7/8] Installing systemd services..."
sudo cp "${APP_DIR}/deploy/stocktracker.service" /etc/systemd/system/stocktracker.service
sudo cp "${APP_DIR}/deploy/stocktracker-worker.service" /etc/systemd/system/stocktracker-worker.service
sudo systemctl daemon-reload
sudo systemctl enable stocktracker stocktracker-worker
sudo systemctl start stocktracker stocktracker-worker
echo "  Services started."

# ── 8. nginx reverse proxy ──────────────────────────────────────────
echo "[8/8] Configuring nginx..."
sudo rm -f /etc/nginx/sites-enabled/default
sudo cp "${APP_DIR}/deploy/nginx.conf" /etc/nginx/sites-available/stocktracker
sudo ln -sf /etc/nginx/sites-available/stocktracker /etc/nginx/sites-enabled/stocktracker
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl enable nginx

# ── Firewall ─────────────────────────────────────────────────────────
echo "Configuring firewall..."
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS (for later SSL)
sudo ufw --force enable

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Edit ~/.env with your Angel One credentials:"
echo "     nano ${APP_DIR}/.env"
echo "  2. Restart the service:"
echo "     sudo systemctl restart stocktracker"
echo "  3. Check status:"
echo "     sudo systemctl status stocktracker"
echo "  4. View logs:"
echo "     sudo journalctl -u stocktracker -f"
echo "  5. Test:"
echo "     curl http://localhost/health"
echo ""
