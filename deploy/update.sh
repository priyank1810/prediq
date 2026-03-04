#!/bin/bash
# update.sh — Pull latest code and restart the service
# Usage: bash deploy/update.sh
set -euo pipefail

APP_DIR="/home/ubuntu/stock-tracker"
VENV_DIR="${APP_DIR}/venv"

echo "=== Updating Stock Tracker ==="

cd "${APP_DIR}"

echo "[1/4] Pulling latest code..."
git pull

echo "[2/4] Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install -r requirements.txt -q

echo "[3/5] Syncing config files..."
sudo cp "${APP_DIR}/deploy/stocktracker.service" /etc/systemd/system/stocktracker.service
sudo cp "${APP_DIR}/deploy/stocktracker-worker.service" /etc/systemd/system/stocktracker-worker.service
sudo cp "${APP_DIR}/deploy/nginx.conf" /etc/nginx/sites-available/stocktracker
sudo systemctl daemon-reload
sudo nginx -t && sudo systemctl reload nginx

echo "[4/5] Restarting server..."
sudo systemctl restart stocktracker

echo "[5/5] Restarting worker..."
sudo systemctl restart stocktracker-worker

echo ""
echo "=== Update complete! ==="
echo "Check status: sudo systemctl status stocktracker stocktracker-worker"
echo "View logs:    sudo journalctl -u stocktracker -f"
echo "Worker logs:  sudo journalctl -u stocktracker-worker -f"
