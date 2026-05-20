#!/usr/bin/env bash
# H_PI_1: GDN-WADI seed 0 under PyG 1.6.0 + Rosetta x86_64.
# Authorized via dec_01KQNYRN1EGWPHCZEQPW923KHN.
# Same H_VAR locked config; PyG-specific patches (node_dim=0, softmax keyword)
# REINSTATED because PyG 1.6.0 has the same scatter/segment_csr API as 2.5.
# This is itself a diagnostic finding — those patches are not "2.x-specific."

set -u
PROJECT_ROOT="/Users/huanbui/Desktop/adaptive-physics-attack-ad"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$RUNS_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyg1x
cd "$PROJECT_ROOT/repos/GDN_pyg1x"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RUNS_DIR/gdn_wadi_pyg1x_seed0_${STAMP}.log"
echo "=== seed 0 PYG1X start at $(date) -> $LOG ==="
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
  -random_seed 0 \
  -comment wadi_pyg1x_seed0 \
  -save_path_pattern wadi_pyg1x_seed0 \
  -report best \
  > "$LOG" 2>&1
RC=$?
echo "=== seed 0 PYG1X end at $(date) (rc=$RC) ==="
