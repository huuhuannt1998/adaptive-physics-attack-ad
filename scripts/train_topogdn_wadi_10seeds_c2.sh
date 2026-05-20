#!/usr/bin/env bash
# Path C-2: TopoGDN/WADI 10 seeds with TopoGDN paper's argparse defaults.
# Authorized via jrn_01KQMRXVZH4JG0AG0XPJ1YQ4Z3 in response to chk_01KQMREB3YM6SF2R4GGZ3DRGK1.
#
# Changes from previous Path A run (dim=256 archived under _oldim256):
#   batch_size: 32  → 16   (TopoGDN paper default)
#   val_ratio:  0.10 → 0.02 (PRESERVE H_VAR deterministic last-N% method)
#   topk:       30   → 15   (TopoGDN paper default for graph attention)
#   epoch:      50   → 150  (TopoGDN paper default)
#
# Preserved:
#   - patience=10 (early-stop)
#   - PA-F1@K=10 evaluation, K=8-of-10 selection rule
#   - deterministic last-N% val split (H_VAR)
#   - use_topo + use_tcn = True
#   - dim=128, out_layer_num=1 (out_layer_inter_dim irrelevant when num=1, confirmed by Path C-1 abort)
#
# Wall-clock estimate per Brain: 24-30 hrs total. After first 3 seeds I'll surface a
# checkpoint if directional progress is poor, so we don't burn the full budget on a
# null result.

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"

run_seed() {
  local SEED=$1
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local LOG="$RUNS_DIR/topogdn_wadi_c2_seed${SEED}_${STAMP}.log"
  echo "=== seed $SEED start at $(date) -> $LOG ==="
  (
    cd "$PROJECT_ROOT/repos/TopoGDN" && \
    python -u main.py \
      -dataset wadi \
      -device cpu \
      -batch 16 \
      -epoch 150 \
      -slide_win 100 \
      -slide_stride 10 \
      -dim 128 \
      -val_ratio 0.02 \
      -topk 15 \
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

echo "=== ALL TOPOGDN C-2 SEEDS DONE at $(date) ==="
