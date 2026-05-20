#!/usr/bin/env bash
# H_VAR run: TopoGDN/WADI 5 seeds with deterministic last-10% val split.
# Will be launched after GDN/WADI H_VAR run lands clean per Brain authorization.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/repos/TopoGDN"

for SEED in 0 1 2 3 4; do
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG="$RUNS_DIR/topogdn_wadi_hvar_seed${SEED}_${STAMP}.log"
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
echo "=== ALL TOPOGDN H_VAR SEEDS DONE at $(date) ==="
