#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# SLAY Attention Ablation — SSH Dispatcher
#
# Launches one attention ablation per remote GPU instance via SSH.
# Each instance runs an independent deepspeed job on 8 GPUs.
#
# Usage:
#   1. Fill in HOSTS below with your instance IPs/hostnames.
#   2. Set SSH_KEY to your private key path (or leave empty for default).
#   3. Run: bash run_ablation_ssh.sh
#
# Each remote instance must already have the repo + dependencies set up.
# Run setup_instance.sh on each host first if starting fresh.
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configure your instances here ────────────────────────────────────
HOSTS=(
    "user@instance-1-ip"
    "user@instance-2-ip"
    "user@instance-3-ip"
    "user@instance-4-ip"
    # "user@instance-5-ip"
    # "user@instance-6-ip"
    # "user@instance-7-ip"
)

SSH_KEY=""           # e.g. ~/.ssh/my_key.pem  — leave empty to use ssh default
REMOTE_REPO_DIR="~/slay"   # path to repo on remote instances
LOG_DIR="./logs"           # local dir to collect logs

# ── Attention types — one per host (4 ablations) ─────────────────────
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

# ── W&B (loaded from .env locally, forwarded as env vars over SSH) ────
source "$(dirname "$0")/../.env" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

if [[ ${#HOSTS[@]} -ne ${#ATTENTION_TYPES[@]} ]]; then
    echo "ERROR: number of HOSTS (${#HOSTS[@]}) must match ATTENTION_TYPES (${#ATTENTION_TYPES[@]})"
    exit 1
fi

echo "Launching ${#HOSTS[@]} ablation jobs..."
echo ""

PIDS=()

for i in "${!HOSTS[@]}"; do
    HOST="${HOSTS[$i]}"
    ATTN="${ATTENTION_TYPES[$i]}"
    RUN_NAME="ablation_${ATTN}_$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/${ATTN}.log"

    echo "  [$((i+1))/${#HOSTS[@]}] $HOST  →  --attention $ATTN  (log: $LOG_FILE)"

    ssh $SSH_OPTS "$HOST" bash -s -- \
        "$REMOTE_REPO_DIR" "$ATTN" "$RUN_NAME" \
        "$CONTEXT_LEN" "$EMBED_DIM" "$N_LAYERS" "$N_HEADS" \
        "$LR" "$BATCH_SIZE" "$GRAD_ACCUM" "$TOTAL_STEPS" "$NUM_GPUS" \
        "${WANDB_API_KEY:-}" "${WANDB_ENTITY:-}" "${WANDB_PROJECT:-}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

cat << 'REMOTE_SCRIPT' > /tmp/_slay_remote.sh
#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$1"; ATTN="$2"; RUN_NAME="$3"
CONTEXT_LEN="$4"; EMBED_DIM="$5"; N_LAYERS="$6"; N_HEADS="$7"
LR="$8"; BATCH_SIZE="$9"; GRAD_ACCUM="${10}"; TOTAL_STEPS="${11}"; NUM_GPUS="${12}"
export WANDB_API_KEY="${13}"
export WANDB_ENTITY="${14}"
export WANDB_PROJECT="${15}"

cd "$REPO_DIR"
source .venv/bin/activate 2>/dev/null || true

deepspeed --num_gpus="$NUM_GPUS" main.py \
    --attention "$ATTN" \
    --context-len "$CONTEXT_LEN" \
    --embed-dim "$EMBED_DIM" \
    --n-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --total-steps "$TOTAL_STEPS" \
    --run-name "$RUN_NAME"
REMOTE_SCRIPT

# Re-launch with the here-doc script piped properly
# (relaunch below using the temp script approach)
# Kill the background jobs started above (they were placeholders)
for PID in "${PIDS[@]}"; do
    kill "$PID" 2>/dev/null || true
done
PIDS=()

echo ""
echo "Dispatching remote scripts..."
echo ""

for i in "${!HOSTS[@]}"; do
    HOST="${HOSTS[$i]}"
    ATTN="${ATTENTION_TYPES[$i]}"
    RUN_NAME="ablation_${ATTN}_$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/${ATTN}.log"

    # Copy the remote script, then execute it with args
    scp $SSH_OPTS /tmp/_slay_remote.sh "${HOST}:/tmp/_slay_remote.sh" > /dev/null 2>&1

    ssh $SSH_OPTS "$HOST" bash /tmp/_slay_remote.sh \
        "$REMOTE_REPO_DIR" "$ATTN" "$RUN_NAME" \
        "$CONTEXT_LEN" "$EMBED_DIM" "$N_LAYERS" "$N_HEADS" \
        "$LR" "$BATCH_SIZE" "$GRAD_ACCUM" "$TOTAL_STEPS" "$NUM_GPUS" \
        "${WANDB_API_KEY:-}" "${WANDB_ENTITY:-}" "${WANDB_PROJECT:-}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "  [$((i+1))/${#HOSTS[@]}] Dispatched: $HOST → --attention $ATTN  (pid=${PIDS[-1]}, log=$LOG_FILE)"
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Follow logs with: tail -f $LOG_DIR/*.log"
echo ""

# Wait for all and report status
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    ATTN="${ATTENTION_TYPES[$i]}"
    if wait "$PID"; then
        echo "  ✓ $ATTN — done"
    else
        echo "  ✗ $ATTN — FAILED (check $LOG_DIR/${ATTN}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All ${#HOSTS[@]} ablations completed successfully."
else
    echo "$FAILED/${#HOSTS[@]} ablations failed."
    exit 1
fi
