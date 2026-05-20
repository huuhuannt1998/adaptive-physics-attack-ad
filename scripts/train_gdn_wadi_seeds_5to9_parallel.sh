#!/usr/bin/env bash
# Option 0 expansion: train seeds 5..9 with 2-at-a-time parallel batches.
# Reuses existing H_VAR seeds 0..4 checkpoints. PI-authorized parallelism per
# dec_01KQHNFER6H61C07F76FB97E47 (jrn_01KQHNGCS3NF4917PVNQT2JNXX).
#
# Same locked config: PA-F1@K=10, deterministic last-10% val split (H_VAR), patience=10.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"

run_seed() {
  local SEED=$1
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local LOG="$RUNS_DIR/gdn_wadi_hvar_seed${SEED}_${STAMP}.log"
  echo "=== seed $SEED start at $(date) -> $LOG ==="
  (
    cd "$PROJECT_ROOT/repos/GDN" && \
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

# Batch 1: seeds 5 + 6 in parallel
run_seed 5 &
PID5=$!
run_seed 6 &
PID6=$!
wait $PID5 $PID6

# Batch 2: seeds 7 + 8 in parallel
run_seed 7 &
PID7=$!
run_seed 8 &
PID8=$!
wait $PID7 $PID8

# Batch 3: seed 9 alone
run_seed 9

echo "=== ALL SEEDS 5-9 DONE at $(date) ==="
