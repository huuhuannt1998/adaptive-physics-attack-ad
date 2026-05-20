#!/usr/bin/env bash
# Train GDN on WADI for 5 seeds with BETA hyperparams; capture each seed's stdout to runs/.
# Total expected wall-clock on M-series CPU: ~3.75 hours (~45 min/seed × 5).
#
# BETA hyperparams (per lit_01KQG4D8J9H9PMKP5W13SXMC46 + lit_01KQG4DY67KE3Z87FEFQZTJCWK):
#   window=100, stride=10, batch=32, epoch=50, val_ratio=0.1, dim=128 (WADI), topk=30 (WADI)
# Validation split: GDN's default random-position 10% (NOT last-10%) — fidelity to GDN.

set -u  # do not -e: continue running remaining seeds even if one crashes
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/repos/GDN"

for SEED in 0 1 2 3 4; do
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
echo "=== ALL SEEDS DONE at $(date) ==="
