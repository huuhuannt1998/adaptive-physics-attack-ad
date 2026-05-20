#!/usr/bin/env bash
# H_PI_4 SWaT prep: TopoGDN-SWaT 10 seeds Option 0 with 2-at-a-time parallel batches.
# BLOCKED until iTrust DUA closes; script ready for immediate launch on data arrival.
# All TopoGDN bug fixes applied to repos/TopoGDN/main.py (use_topo+use_tcn=True,
# _slice_dict patches, H_VAR deterministic split). Same as WADI runs.
#
# Hyperparams: BETA stated config + TopoGDN paper SWaT defaults
#   - window=100, stride=10
#   - batch=32, dim=64 (matches GDN-SWaT)
#   - val_ratio=0.1, topk=15 (SWaT-specific)
#   - epoch=50, patience=10

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

if [ ! -f "$PROJECT_ROOT/repos/TopoGDN/data/swat/train.csv" ]; then
  echo "ERROR: SWaT preprocessed train.csv not found at repos/TopoGDN/data/swat/train.csv"
  echo "       Run loaders/swat_loader.py first; data load is BLOCKED on iTrust DUA."
  exit 2
fi

source "$PROJECT_ROOT/.venv/bin/activate"

run_seed() {
  local SEED=$1
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local LOG="$RUNS_DIR/topogdn_swat_seed${SEED}_${STAMP}.log"
  echo "=== seed $SEED start at $(date) -> $LOG ==="
  (
    cd "$PROJECT_ROOT/repos/TopoGDN" && \
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

run_seed 0 & P0=$!; run_seed 1 & P1=$!; wait $P0 $P1
run_seed 2 & P2=$!; run_seed 3 & P3=$!; wait $P2 $P3
run_seed 4 & P4=$!; run_seed 5 & P5=$!; wait $P4 $P5
run_seed 6 & P6=$!; run_seed 7 & P7=$!; wait $P6 $P7
run_seed 8 & P8=$!; run_seed 9 & P9=$!; wait $P8 $P9

echo "=== ALL TOPOGDN-SWAT SEEDS DONE at $(date) ==="
