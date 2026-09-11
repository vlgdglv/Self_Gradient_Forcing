#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/ode_rollout_chunkwise.yaml}"
LOGDIR="${2:-training_outputs/rollout_ode_chunkwise}"
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

python - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
from utils.dataset import ODERegressionLMDBDataset

cfg = OmegaConf.merge(
    OmegaConf.load("configs/default_config.yaml"),
    OmegaConf.load(sys.argv[1]),
)

assert cfg.trainer == "ode"
assert bool(getattr(cfg, "rollout_ode", False))
assert not getattr(cfg, "generator_ckpt", None), "Remove generator_ckpt: smoke should start from raw Wan."
assert cfg.model_kwargs.model_name == "Wan2.1-T2V-1.3B"
assert cfg.num_frame_per_block == 3
assert cfg.model_kwargs.local_attn_size == 9
assert cfg.model_kwargs.sink_size == 3

d = ODERegressionLMDBDataset(cfg.data_path, max_pair=1)
sample = d[0]
x = sample["ode_latent"]

print("[preflight] data_path:", cfg.data_path)
print("[preflight] prompt:", str(sample["prompts"])[:100])
print("[preflight] ode_latent shape:", tuple(x.shape))
print("[preflight] ode_latent dtype:", x.dtype)
print("[preflight] PASS")
PY

echo "===================================================="
echo "Rollout-Matched ODE smoke"
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
  --no_visualize \
  2>&1 | tee "$LOGDIR/train_shell.log"

  # --disable-wandb