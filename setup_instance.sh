#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# SLAY Instance Setup
#
# Usage (from your local machine):
#   HF_TOKEN="..." bash setup_instance.sh ubuntu@<ip>
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_URL="https://github.com/azettaai/slay.git"
REPO_DIR="$HOME/slay"
BRANCH="master"

# ── If a remote host is given, copy this script and run it there ─────
if [[ $# -ge 1 && "$1" != "--local" ]]; then
    HOST="$1"
    SSH_OPTS="-o StrictHostKeyChecking=no"
    echo "Setting up $HOST..."
    scp $SSH_OPTS "$0" "${HOST}:/tmp/setup_instance.sh"
    ssh $SSH_OPTS "$HOST" "HF_TOKEN=${HF_TOKEN:-} bash /tmp/setup_instance.sh --local"
    echo "Done: $HOST"
    exit 0
fi

# ── Runs on the remote instance ───────────────────────────────────────
echo "=== SLAY Instance Setup ==="
echo "Host: $(hostname)"
echo ""

echo "[1/4] Installing system dependencies..."
sudo apt-get install -y python3-pip 2>/dev/null || true

echo "[2/4] Cloning repo..."
if [[ -d "$REPO_DIR" ]]; then
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

echo "[3/4] Installing Python dependencies..."
sudo pip3 install -q --upgrade pip
sudo pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
sudo pip3 install -q deepspeed wandb transformers datasets

echo "[4/4] Pre-downloading datasets..."
HF_TOKEN="${HF_TOKEN:-}" python3 -c "
import os
os.environ['HF_TOKEN'] = '${HF_TOKEN:-}'
from datasets import load_dataset
print('  Downloading wikitext-2...')
load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
print('  Done.')
"

echo ""
echo "Setup complete on $(hostname). Ready to run ablations."
