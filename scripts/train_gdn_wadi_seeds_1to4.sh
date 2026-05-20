#!/usr/bin/env bash
# Resume GDN/WADI training for seeds 1..4 only (seed 0 already complete).
# Per Brain's H5 lock-in (jrn_01KQH27KYYY6HDGB1X0JYRSM22): PA-F1 @ topk=10 is the gating metric;
# both raw F1 and PA-F1 will be aggregated post-training via scripts/evaluate_with_pa_f1.py.
# Hyperparams identical to the previous run: window=100, stride=10, batch=32, dim=128, K=30, val_ratio=0.1.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/repos/GDN"

for SEED in 1 2 3 4; do
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG="$RUNS_DIR/gdn_wadi_seed${SEED}_${STAMP}.log"
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
echo "=== ALL SEEDS 1-4 DONE at $(date) ==="
