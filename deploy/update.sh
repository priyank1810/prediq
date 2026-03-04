#!/bin/bash
# update.sh — Pull latest code and restart the service
# Usage: bash deploy/update.sh
set -euo pipefail

APP_DIR="/home/ubuntu/stock-tracker"
VENV_DIR="${APP_DIR}/venv"

echo "=== Updating Stock Tracker ==="

cd "${APP_DIR}"

echo "[1/3] Pulling latest code..."
git pull

echo "[2/3] Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install -r requirements.txt -q

echo "[3/3] Restarting service..."
sudo systemctl restart stocktracker

echo ""
echo "=== Update complete! ==="
echo "Check status: sudo systemctl status stocktracker"
echo "View logs:    sudo journalctl -u stocktracker -f"
