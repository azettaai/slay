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
#   bash setup_instance.sh
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
    ssh $SSH_OPTS "$HOST" bash /tmp/setup_instance.sh --local
    echo "Done: $HOST"
    exit 0
fi

# ── Local execution (runs on the remote instance) ────────────────────
echo "=== SLAY Instance Setup ==="
echo "Host: $(hostname)"
echo ""

# Python + pip
echo "[1/5] Checking Python..."
python3 --version

# Clone or update repo
echo "[2/5] Cloning repo..."
if [[ -d "$REPO_DIR" ]]; then
    echo "  Repo already exists — pulling latest..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# Virtual environment
echo "[3/5] Setting up virtualenv..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Dependencies
echo "[4/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q deepspeed wandb transformers datasets

# Verify GPU + DeepSpeed
echo "[5/5] Verifying setup..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"
python3 -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
ds_report 2>/dev/null | grep -E "ops|cuda" | head -10 || true

echo ""
echo "Setup complete on $(hostname). Ready to run ablations."
