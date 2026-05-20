#!/usr/bin/env bash
# Path C: TopoGDN/WADI 10 seeds with out_layer_inter_dim=64 (TopoGDN's argparse default).
# Authorized via jrn_01KQM8GXE9NT4J2H7P2BZSANQJ — Path E exonerated modernization;
# now testing whether the BETA TopoGDN target (PA-F1 0.90) is reachable with TopoGDN's
# own paper hyperparameter (out_layer_inter_dim=64, vs the 256 we used previously).
# Same locked stack: PA-F1@K=10, deterministic last-10% val split (H_VAR), use_topo+use_tcn=True,
# patience=10, K=8-of-10 selection rule.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"

run_seed() {
  local SEED=$1
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local LOG="$RUNS_DIR/topogdn_wadi_pathc_seed${SEED}_${STAMP}.log"
  echo "=== seed $SEED start at $(date) -> $LOG ==="
  (
    cd "$PROJECT_ROOT/repos/TopoGDN" && \
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
      -out_layer_inter_dim 64 \
      -random_seed "$SEED" \
      -comment "wadi_seed${SEED}" \
      -save_path_pattern "wadi_seed${SEED}" \
      -report best \
      > "$LOG" 2>&1
  )
  local RC=$?
  echo "=== seed $SEED end at $(date) (rc=$RC) ==="
}

# 5 batches of 2-at-a-time parallel.
run_seed 0 &
P0=$!; run_seed 1 &
P1=$!; wait $P0 $P1

run_seed 2 &
P2=$!; run_seed 3 &
P3=$!; wait $P2 $P3

run_seed 4 &
P4=$!; run_seed 5 &
P5=$!; wait $P4 $P5

run_seed 6 &
P6=$!; run_seed 7 &
P7=$!; wait $P6 $P7

run_seed 8 &
P8=$!; run_seed 9 &
P9=$!; wait $P8 $P9

echo "=== ALL TOPOGDN PATH-C SEEDS DONE at $(date) ==="
