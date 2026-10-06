#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/teacher_forcing_dmd_chunkwise.yaml}"
LOGDIR="${2:-training_outputs/teacher_forcing_dmd_chunkwise}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29531}"

[[ -f "$CONFIG" ]] || { echo "ERROR: missing config: $CONFIG" >&2; exit 1; }
[[ -d wan_models/Wan2.1-T2V-1.3B ]] || {
  echo "ERROR: missing raw Wan base at wan_models/Wan2.1-T2V-1.3B" >&2
  exit 1
}

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "$LOGDIR"

echo "===================================================="
echo "Teacher Forcing DMD"
echo "Config:     $CONFIG"
echo "Logdir:     $LOGDIR"
echo "GPUs:       $NUM_GPUS"
echo "Init:       RAW Wan2.1-T2V-1.3B"
echo "===================================================="

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --no_visualize  \
  2>&1 | tee "$LOGDIR/train_shell.log"

  # --disable-wandb