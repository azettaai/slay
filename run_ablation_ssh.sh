#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# SLAY Attention Ablation — SSH Dispatcher
#
# Launches one attention ablation per remote GPU instance via SSH.
# Each job runs inside a tmux session — safe to close your laptop.
#
# Usage:
#   1. Fill in HOSTS below with your instance IPs/hostnames.
#   2. Set SSH_KEY to your private key path (or leave empty for default).
#   3. Run: bash run_ablation_ssh.sh
#
# To check on a running job:
#   ssh ubuntu@<ip>
#   tmux attach -t slay_yat          # replace with attention type
#
# Each remote instance must already have the repo + dependencies set up.
# Run setup_instance.sh on each host first if starting fresh.
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configure your instances here ────────────────────────────────────
HOSTS=(
    "ubuntu@185.216.20.59"
    "ubuntu@38.128.233.103"
    "ubuntu@38.128.233.144"
    "ubuntu@38.128.233.111"
    # "user@instance-5-ip"
    # "user@instance-6-ip"
    # "user@instance-7-ip"
)

SSH_KEY=""                   # e.g. ~/.ssh/my_key.pem — leave empty for default
REMOTE_REPO_DIR="~/slay"     # path to repo on remote instances

# ── Attention types — one per host ───────────────────────────────────
ATTENTION_TYPES=(
    "yat"
    "standard"
    "performer"
    "yat-performer"
    # "yat-spherical"
    # "linear"
    # "cosformer"
)

# ── Shared training hyperparameters ──────────────────────────────────
CONTEXT_LEN=2048
EMBED_DIM=1024
N_LAYERS=24
N_HEADS=16
LR=2e-4
BATCH_SIZE=8
GRAD_ACCUM=4
TOTAL_STEPS=15000
NUM_GPUS=8

# ── Load W&B credentials from .env ───────────────────────────────────
source "$(dirname "$0")/../.env" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

if [[ ${#HOSTS[@]} -ne ${#ATTENTION_TYPES[@]} ]]; then
    echo "ERROR: number of HOSTS (${#HOSTS[@]}) must match ATTENTION_TYPES (${#ATTENTION_TYPES[@]})"
    exit 1
fi

# ── Write the remote runner script ───────────────────────────────────
cat << 'REMOTE_SCRIPT' > /tmp/_slay_remote.sh
#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$1"; ATTN="$2"; RUN_NAME="$3"
CONTEXT_LEN="$4"; EMBED_DIM="$5"; N_LAYERS="$6"; N_HEADS="$7"
LR="$8"; BATCH_SIZE="$9"; GRAD_ACCUM="${10}"; TOTAL_STEPS="${11}"; NUM_GPUS="${12}"
export WANDB_API_KEY="${13}"
export WANDB_ENTITY="${14}"
export WANDB_PROJECT="${15}"

SESSION="slay_${ATTN}"

cd "$REPO_DIR"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Launch inside tmux — job survives SSH disconnection
tmux new-session -d -s "$SESSION" bash -c "
    cd $REPO_DIR
        export WANDB_API_KEY='${13}'
    export WANDB_ENTITY='${14}'
    export WANDB_PROJECT='${15}'
    python3 -m deepspeed --num_gpus=$NUM_GPUS main.py \
        --attention $ATTN \
        --context-len $CONTEXT_LEN \
        --embed-dim $EMBED_DIM \
        --n-layers $N_LAYERS \
        --n-heads $N_HEADS \
        --lr $LR \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation-steps $GRAD_ACCUM \
        --total-steps $TOTAL_STEPS \
        --run-name $RUN_NAME \
    2>&1 | tee ~/slay_${ATTN}.log; exec bash
"

echo "Job launched in tmux session '$SESSION'"
echo "To monitor: ssh <host> then: tmux attach -t $SESSION"
REMOTE_SCRIPT

echo "Dispatching ${#HOSTS[@]} ablation jobs..."
echo ""

for i in "${!HOSTS[@]}"; do
    HOST="${HOSTS[$i]}"
    ATTN="${ATTENTION_TYPES[$i]}"
    RUN_NAME="ablation_${ATTN}_$(date +%Y%m%d_%H%M%S)"

    scp $SSH_OPTS /tmp/_slay_remote.sh "${HOST}:/tmp/_slay_remote.sh" > /dev/null 2>&1

    ssh $SSH_OPTS "$HOST" bash /tmp/_slay_remote.sh \
        "$REMOTE_REPO_DIR" "$ATTN" "$RUN_NAME" \
        "$CONTEXT_LEN" "$EMBED_DIM" "$N_LAYERS" "$N_HEADS" \
        "$LR" "$BATCH_SIZE" "$GRAD_ACCUM" "$TOTAL_STEPS" "$NUM_GPUS" \
        "${WANDB_API_KEY:-}" "${WANDB_ENTITY:-}" "${WANDB_PROJECT:-}"

    echo "  [$((i+1))/${#HOSTS[@]}] $HOST → --attention $ATTN  (tmux: slay_${ATTN})"
done

echo ""
echo "All jobs launched in tmux sessions. Safe to close your laptop."
echo ""
echo "To check on a job:"
echo "  ssh ubuntu@<ip>"
echo "  tmux attach -t slay_<attention>    # e.g. tmux attach -t slay_yat"
echo ""
echo "Logs are also saved to ~/slay_<attention>.log on each instance."
