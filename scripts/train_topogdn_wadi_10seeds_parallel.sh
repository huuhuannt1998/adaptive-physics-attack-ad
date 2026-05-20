#!/usr/bin/env bash
# TopoGDN/WADI 10-seed Option 0 with 2-at-a-time parallel batches.
# Authorized via dec_01KQHZ6ZZHAZR3TSYPTRBYGGV1 (Path A) + jrn_01KQHZ8WX7XVDTY39JHJR8TKR1.
# Same locked config as GDN/WADI:
#   PA-F1@K=10, deterministic last-10% val split (H_VAR), patience=10.
# H_VAR patch already applied to repos/TopoGDN/main.py:get_loaders.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"

run_seed() {
  local SEED=$1
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local LOG="$RUNS_DIR/topogdn_wadi_hvar_seed${SEED}_${STAMP}.log"
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
      -random_seed "$SEED" \
      -comment "wadi_seed${SEED}" \
      -save_path_pattern "wadi_seed${SEED}" \
      -report best \
      > "$LOG" 2>&1
  )
  local RC=$?
  echo "=== seed $SEED end at $(date) (rc=$RC) ==="
}

# 5 batches of 2-at-a-time parallel, except last batch which is alone.
# Batch 1: seeds 0 + 1
run_seed 0 &
P0=$!
run_seed 1 &
P1=$!
wait $P0 $P1

# Batch 2: seeds 2 + 3
run_seed 2 &
P2=$!
run_seed 3 &
P3=$!
wait $P2 $P3

# Batch 3: seeds 4 + 5
run_seed 4 &
P4=$!
run_seed 5 &
P5=$!
wait $P4 $P5

# Batch 4: seeds 6 + 7
run_seed 6 &
P6=$!
run_seed 7 &
P7=$!
wait $P6 $P7

# Batch 5: seeds 8 + 9
run_seed 8 &
P8=$!
run_seed 9 &
P9=$!
wait $P8 $P9

echo "=== ALL TOPOGDN SEEDS 0-9 DONE at $(date) ==="
