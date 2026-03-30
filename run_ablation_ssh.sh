#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# SLAY Attention Ablation — SSH Dispatcher
#
# Launches one attention ablation per remote GPU instance via SSH.
# Each job runs inside a tmux session — safe to close your laptop.
#
# Usage:
#   bash run_ablation_ssh.sh
#
# To check on a running job:
#   ssh ubuntu@<ip>
#   tmux attach -t slay_<attention>
#
# Logs saved to ~/slay_<attention>.log on each instance.
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Instances & attention types (one per host) ────────────────────────
HOSTS=(
    "ubuntu@38.128.233.231"
    "ubuntu@38.128.233.233"
    "ubuntu@38.128.233.249"
    "ubuntu@38.128.233.148"
)

ATTENTION_TYPES=(
    "yat-spherical"
    "standard"
    "performer"
    "yat-performer"
)

# ── Hyperparameters ───────────────────────────────────────────────────
CONTEXT_LEN=2048
EMBED_DIM=1024
N_LAYERS=24
N_HEADS=16
LR=2e-4
BATCH_SIZE=16
GRAD_ACCUM=4
TOTAL_STEPS=20000
NUM_GPUS=8

# ── Load credentials ──────────────────────────────────────────────────
source "$(dirname "$0")/../.env" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"

if [[ ${#HOSTS[@]} -ne ${#ATTENTION_TYPES[@]} ]]; then
    echo "ERROR: HOSTS and ATTENTION_TYPES must have the same number of entries"
    exit 1
fi

echo "Dispatching ${#HOSTS[@]} ablation jobs..."
echo ""

for i in "${!HOSTS[@]}"; do
    HOST="${HOSTS[$i]}"
    ATTN="${ATTENTION_TYPES[$i]}"
    RUN_NAME="ablation_${ATTN}_$(date +%Y%m%d_%H%M%S)"
    SESSION="slay_${ATTN}"

    # Build a self-contained run script with all variables substituted
    cat > /tmp/_slay_run_${ATTN}.sh << EOF
#!/usr/bin/env bash
export PATH="\$HOME/.local/bin:/usr/local/bin:\$PATH"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export HF_TOKEN="${HF_TOKEN:-}"
cd ~/slay
git pull origin master
deepspeed --num_gpus=${NUM_GPUS} main.py \\
    --attention ${ATTN} \\
    --context-len ${CONTEXT_LEN} \\
    --embed-dim ${EMBED_DIM} \\
    --n-layers ${N_LAYERS} \\
    --n-heads ${N_HEADS} \\
    --lr ${LR} \\
    --batch-size ${BATCH_SIZE} \\
    --gradient-accumulation-steps ${GRAD_ACCUM} \\
    --total-steps ${TOTAL_STEPS} \\
    --run-name ${RUN_NAME} \\
    2>&1 | tee ~/slay_${ATTN}.log; exec bash
EOF

    scp $SSH_OPTS /tmp/_slay_run_${ATTN}.sh "${HOST}:/tmp/_slay_run_${ATTN}.sh" > /dev/null 2>&1

    ssh $SSH_OPTS "$HOST" "
        tmux kill-session -t ${SESSION} 2>/dev/null || true
        tmux new-session -d -s ${SESSION} bash /tmp/_slay_run_${ATTN}.sh
    "

    echo "  [$((i+1))/${#HOSTS[@]}] $HOST → --attention $ATTN  (tmux: ${SESSION})"
done

echo ""
echo "All jobs launched. Safe to close your laptop."
echo "Monitor: ssh ubuntu@<ip> && tmux attach -t slay_<attention>"
