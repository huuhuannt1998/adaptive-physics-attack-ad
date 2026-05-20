#!/usr/bin/env bash
# Train TopoGDN on WADI for 5 seeds with BETA hyperparams.
# Sequential to GDN/WADI training to avoid CPU contention.
# Expected wall-clock per seed: TBD (likely 1.5-3x GDN due to persistent-homology cost).
#
# BETA hyperparams: window=100, stride=10, batch=32, epoch=50, val_ratio=0.1, dim=128 (WADI), topk=30 (WADI)

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/repos/TopoGDN"

for SEED in 0 1 2 3 4; do
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG="$RUNS_DIR/topogdn_wadi_seed${SEED}_${STAMP}.log"
  echo "=== seed $SEED start at $(date) -> $LOG ==="
  python -u main.py \
    -dataset wadi \
    -device cpu \
    -batch 32 \
    -epoch 50 \
    -slide_win 100 \
    -slide_stride 10 \
    -dim 128 \
    -val_ratio 0.1 \
    -topk 30 \
    -random_seed "$SEED" \
    -comment "wadi_seed${SEED}" \
    -save_path_pattern "wadi_seed${SEED}" \
    -report best \
    > "$LOG" 2>&1
  RC=$?
  echo "=== seed $SEED end at $(date) (rc=$RC) ==="
done
echo "=== ALL TOPOGDN SEEDS DONE at $(date) ==="
