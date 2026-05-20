#!/usr/bin/env bash
# H_PI_4 SWaT prep: GDN-SWaT 10 seeds Option 0 with 2-at-a-time parallel batches.
# BLOCKED until iTrust DUA closes (chk_01KQGQDN5J2C9ESNT9BBPTE088); script ready
# for immediate launch on data arrival.
#
# Hyperparams: BETA stated config + GDN paper SWaT defaults
#   - window=100, stride=10
#   - batch=32, dim=64 (GDN paper SWaT-specific; not 128 as for WADI)
#   - val_ratio=0.1, topk=15 (BETA's TopoGDN-inheritance for SWaT, NOT 30 as for WADI)
#   - epoch=50, patience=10
# Locked: PA-F1@K=10, H_VAR deterministic last-10% val split, K=8-of-10 seed selection.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

# Pre-flight: confirm SWaT data is present.
if [ ! -f "$PROJECT_ROOT/repos/GDN/data/swat/train.csv" ]; then
  echo "ERROR: SWaT preprocessed train.csv not found at repos/GDN/data/swat/train.csv"
  echo "       Run loaders/swat_loader.py first; data load is BLOCKED on iTrust DUA"
  echo "       (chk_01KQGQDN5J2C9ESNT9BBPTE088)."
  exit 2
fi

source "$PROJECT_ROOT/.venv/bin/activate"

run_seed() {
  local SEED=$1
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local LOG="$RUNS_DIR/gdn_swat_seed${SEED}_${STAMP}.log"
  echo "=== seed $SEED start at $(date) -> $LOG ==="
  (
    cd "$PROJECT_ROOT/repos/GDN" && \
    python -u main.py \
      -dataset swat \
      -device cpu \
      -batch 32 \
      -epoch 50 \
      -slide_win 100 \
      -slide_stride 10 \
      -dim 64 \
      -val_ratio 0.1 \
      -topk 15 \
      -random_seed "$SEED" \
      -comment "swat_seed${SEED}" \
      -save_path_pattern "swat_seed${SEED}" \
      -report best \
      > "$LOG" 2>&1
  )
  local RC=$?
  echo "=== seed $SEED end at $(date) (rc=$RC) ==="
}

# 5 batches of 2-at-a-time parallel.
run_seed 0 & P0=$!; run_seed 1 & P1=$!; wait $P0 $P1
run_seed 2 & P2=$!; run_seed 3 & P3=$!; wait $P2 $P3
run_seed 4 & P4=$!; run_seed 5 & P5=$!; wait $P4 $P5
run_seed 6 & P6=$!; run_seed 7 & P7=$!; wait $P6 $P7
run_seed 8 & P8=$!; run_seed 9 & P9=$!; wait $P8 $P9

echo "=== ALL GDN-SWAT SEEDS DONE at $(date) ==="
