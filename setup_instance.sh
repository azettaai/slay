#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# SLAY Instance Setup
#
# Run this once on each remote GPU instance before launching ablations.
#
# Usage (from your local machine):
#   bash setup_instance.sh user@instance-ip
#
# Or directly on the remote instance:
#   bash setup_instance.sh --local
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_URL="https://github.com/azettaai/slay.git"
REPO_DIR="$HOME/slay"
BRANCH="master"

# ── If a remote host is given, copy this script and run it there ─────
if [[ $# -ge 1 && "$1" != "--local" ]]; then
    HOST="$1"
    SSH_KEY="${2:-}"
    SSH_OPTS="-o StrictHostKeyChecking=no"
    [[ -n "$SSH_KEY" ]] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

    echo "Setting up $HOST..."
    scp $SSH_OPTS "$0" "${HOST}:/tmp/setup_instance.sh"
    ssh $SSH_OPTS "$HOST" "HF_TOKEN=${HF_TOKEN:-} bash /tmp/setup_instance.sh --local"
    echo "Done: $HOST"
    exit 0
fi

# ── Local execution (runs on the remote instance) ────────────────────
echo "=== SLAY Instance Setup ==="
echo "Host: $(hostname)"
echo ""

# Python + pip
echo "[1/4] Checking Python..."
python3 --version
sudo apt-get install -y python3-pip 2>/dev/null || true

# Clone or update repo
echo "[2/4] Cloning repo..."
if [[ -d "$REPO_DIR" ]]; then
    echo "  Repo already exists — pulling latest..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# Dependencies (install globally — ephemeral instance, no venv needed)
echo "[3/4] Installing dependencies..."
sudo pip3 install -q --upgrade pip
sudo pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
sudo pip3 install -q deepspeed wandb transformers datasets

# Pre-download datasets to avoid rate limits during training
echo "[4/5] Pre-downloading datasets..."
python3 - << 'PYEOF'
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
from datasets import load_dataset
print("  Downloading wikitext validation split...")
load_dataset("wikitext", "wikitext-103-v1", split="validation")
print("  Downloading fineweb-edu sample (streaming — no download needed)...")
print("  Done.")
PYEOF

# Verify GPU + DeepSpeed
echo "[5/5] Verifying setup..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"
python3 -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
ds_report 2>/dev/null | grep -E "ops|cuda" | head -10 || true

echo ""
echo "Setup complete on $(hostname). Ready to run ablations."
