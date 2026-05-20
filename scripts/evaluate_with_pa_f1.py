"""
Re-evaluate a trained GDN/TopoGDN checkpoint with both raw F1 and point-adjusted F1.

Authorized by Brain (jrn_01KQH1600NK00V84ZXT865K5WD) in response to
chk_01KQH11P546A3ACJ68PC0F4JGZ. Implements the canonical Xu et al. 2018 point-adjust
convention:

    For each ground-truth anomaly segment (contiguous run of label==1), if ANY point
    inside the segment is flagged as anomalous, count ALL points in that segment as
    detected. Apply BEFORE computing precision / recall / F1.

Reference: TranAD repo's `eval_methods.adjust_predicts`, Anomaly Transformer's
`adjust_anomaly_segments` — both materially identical.

Usage:
    python scripts/evaluate_with_pa_f1.py \\
        --detector gdn --dataset wadi --seed 0 \\
        --checkpoint "repos/GDN/pretrained/wadi_seed0/best_05|01-01:08:44.pt"

The script loads the checkpoint via GDN's Main class with load_model_path set, gets
the test_result + val_result tensors GDN produces, and computes both raw F1 (mirroring
GDN's `get_best_performance_data`) and PA-F1 over the same 400-step threshold sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GDN_REPO = PROJECT_ROOT / "repos/GDN"
TOPOGDN_REPO = PROJECT_ROOT / "repos/TopoGDN"


def adjust_predicts(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Xu et al. 2018 point-adjust: any-point-in-segment-detected => all-points-in-segment-detected.

    Operates on 1-D 0/1 arrays. Returns a NEW array; does not mutate input.
    """
    pred = pred.copy().astype(int)
    label = label.astype(int)
    n = len(label)
    in_seg = False
    seg_start = 0
    for i in range(n):
        if label[i] == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif label[i] == 0 and in_seg:
            in_seg = False
            if pred[seg_start:i].any():
                pred[seg_start:i] = 1
    if in_seg:
        if pred[seg_start:n].any():
            pred[seg_start:n] = 1
    return pred


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, th_steps: int = 400):
    """Mirror `eval_scores` threshold sweep, return both raw and PA-F1 across all thresholds."""
    scores_sorted = rankdata(scores, method="ordinal")
    n = len(scores)
    raw = []
    pa = []
    for i in range(th_steps):
        cur_pred = (scores_sorted > (i / th_steps) * n).astype(int)
        raw_f1 = f1_score(labels, cur_pred, zero_division=0)
        pa_pred = adjust_predicts(cur_pred, labels)
        pa_f1 = f1_score(labels, pa_pred, zero_division=0)
        raw.append((raw_f1, cur_pred))
        pa.append((pa_f1, pa_pred))
    return raw, pa


def best_metrics(name: str, sweep, labels: np.ndarray, scores: np.ndarray):
    f1s = [s[0] for s in sweep]
    best_idx = int(np.argmax(f1s))
    best_f1, best_pred = sweep[best_idx]
    pr = precision_score(labels, best_pred, zero_division=0)
    rc = recall_score(labels, best_pred, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "metric": name,
        "f1": float(best_f1),
        "precision": float(pr),
        "recall": float(rc),
        "auc_pr_proxy_roc": float(auc),
        "best_threshold_step": best_idx,
        "best_threshold_percentile": best_idx / len(sweep),
    }


def aggregate_topk(per_sensor_scores: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Aggregate per-sensor scores across sensors into a 1-D timestep score.

    GDN's `get_best_performance_data` uses top-K-sum (with default K=1, i.e. just the max).
    BETA may use a higher K. We try both sum and mean variants for each K.
    """
    total_features = per_sensor_scores.shape[0]
    if k > total_features:
        k = total_features
    topk_indices = np.argpartition(
        per_sensor_scores, range(total_features - k - 1, total_features), axis=0
    )[-k:]
    topk_vals = np.take_along_axis(per_sensor_scores, topk_indices, axis=0)
    if mode == "sum":
        return np.sum(topk_vals, axis=0)
    elif mode == "mean":
        return np.mean(topk_vals, axis=0)
    else:
        raise ValueError(mode)


def detect_out_layer_inter_dim(checkpoint_path: str) -> int:
    """Inspect checkpoint state_dict to recover out_layer_inter_dim used at training time.

    GDN/TopoGDN's OutLayer's first Linear has weight of shape (out_inter_dim, in_dim) when
    out_layer_num > 1; when out_layer_num == 1 the weight is (1, in_dim) and we cannot recover
    the inter dim — but the checkpoint will load fine either way since only the input dim
    matters for shape compatibility. Default to 256 if we can't tell.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Look for `out_layer.mlp.0.weight` shape — when out_layer_num=1, this is the only Linear,
    # and its first dim is the output (1 for forecasting). When >1 layer, first dim is inter_dim.
    for key in ("out_layer.mlp.0.weight", "out_layer.temp.weight"):
        if key in state:
            shape = state[key].shape
            if shape[0] > 1:
                return int(shape[0])
    return 256  # safe default


def evaluate(detector: str, dataset: str, seed: int, checkpoint_path: str, topk_grid: list[int] | None = None):
    if detector == "gdn":
        repo_dir = GDN_REPO
    elif detector == "topogdn":
        repo_dir = TOPOGDN_REPO
    else:
        raise ValueError(f"unknown detector: {detector}")

    # GDN/TopoGDN main.py uses relative paths (`./data/...`, `./pretrained/...`).
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    # Seed BEFORE constructing Main so the random validation split matches the original training.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Auto-detect out_layer_inter_dim from checkpoint so we can re-eval Path C (dim=64)
    # checkpoints alongside the original Path A (dim=256) ones.
    abs_ckpt = checkpoint_path
    if not Path(abs_ckpt).is_absolute():
        abs_ckpt = str(repo_dir / checkpoint_path.lstrip("./"))
    detected_inter_dim = detect_out_layer_inter_dim(abs_ckpt)

    # WADI BETA hyperparams (mirroring scripts/train_gdn_wadi_5seeds.sh).
    train_config = {
        "batch": 32,
        "epoch": 50,
        "slide_win": 100,
        "dim": 128,
        "slide_stride": 10,
        "comment": f"eval_pa_f1_seed{seed}",
        "seed": seed,
        "out_layer_num": 1,
        "out_layer_inter_dim": detected_inter_dim,
        "decay": 0.0,
        "val_ratio": 0.1,
        "topk": 30,
    }
    # TopoGDN's Main also needs these keys (upstream main.py declares them in train_config).
    if detector == "topogdn":
        train_config.update({
            "use_tcn": True,
            "use_topo": True,
            "model": "GDN",
        })
    env_config = {
        "save_path": f"{dataset}_seed{seed}",
        "dataset": dataset,
        "report": "best",
        "device": "cpu",
        "load_model_path": checkpoint_path,
    }

    from main import Main  # imports lazily so sys.path manipulation lands first

    m = Main(train_config, env_config, debug=False)
    m.run()  # loads checkpoint, runs test + val, prints raw F1 (which we re-derive below).

    from evaluate import get_full_err_scores

    test_scores, normal_scores = get_full_err_scores(m.test_result, m.val_result)
    np_test_result = np.array(m.test_result)
    test_labels = np.array(np_test_result[2, :, 0]).astype(int)

    # H5 reduction-only test: sweep top-K aggregation across sensors.
    # K=1 is GDN's default (max-sensor). Brain hypothesis: BETA uses a larger K.
    if topk_grid is None:
        topk_grid = [1, 3, 5, 10]

    topk_results: list[dict] = []
    for k in topk_grid:
        for agg_mode in ("sum", "mean"):
            test_agg = aggregate_topk(test_scores, k, agg_mode)
            val_agg = aggregate_topk(normal_scores, k, agg_mode)

            # Oracle best-F1 sweep for raw + PA.
            raw_sweep, pa_sweep = sweep_thresholds(test_agg, test_labels, th_steps=400)
            raw_best = best_metrics(f"best_raw_f1_topk{k}_{agg_mode}", raw_sweep, test_labels, test_agg)
            pa_best = best_metrics(f"best_pa_f1_topk{k}_{agg_mode}", pa_sweep, test_labels, test_agg)

            # max-validation-error threshold (BETA's stated convention).
            val_thr = float(np.max(val_agg))
            val_pred = (test_agg > val_thr).astype(int)
            val_pa_pred = adjust_predicts(val_pred, test_labels)
            val_block = {
                "threshold": val_thr,
                "raw_f1": float(f1_score(test_labels, val_pred, zero_division=0)),
                "raw_precision": float(precision_score(test_labels, val_pred, zero_division=0)),
                "raw_recall": float(recall_score(test_labels, val_pred, zero_division=0)),
                "pa_f1": float(f1_score(test_labels, val_pa_pred, zero_division=0)),
                "pa_precision": float(precision_score(test_labels, val_pa_pred, zero_division=0)),
                "pa_recall": float(recall_score(test_labels, val_pa_pred, zero_division=0)),
            }

            topk_results.append({
                "topk": k,
                "aggregation": agg_mode,
                "best_threshold_raw": raw_best,
                "best_threshold_pa": pa_best,
                "val_threshold": val_block,
            })

    # Pick the configuration with the highest oracle PA-F1 across all (K, agg) combos.
    winning = max(topk_results, key=lambda r: r["best_threshold_pa"]["f1"])

    return {
        "detector": detector,
        "dataset": dataset,
        "seed": seed,
        "checkpoint": checkpoint_path,
        "n_test_windows": int(len(test_labels)),
        "n_test_anomalies": int(test_labels.sum()),
        "test_anomaly_rate": float(test_labels.mean()),
        "topk_sweep": topk_results,
        "winning_config": {
            "topk": winning["topk"],
            "aggregation": winning["aggregation"],
            "best_pa_f1": winning["best_threshold_pa"]["f1"],
            "val_threshold_pa_f1": winning["val_threshold"]["pa_f1"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["gdn", "topogdn"], required=True)
    parser.add_argument("--dataset", choices=["swat", "wadi"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="Optional path to write the JSON report.")
    parser.add_argument("--topk-grid", default="1,3,5,10", help="Comma-separated K values for the top-K aggregation sweep.")
    args = parser.parse_args()

    topk_grid = [int(k) for k in args.topk_grid.split(",")]
    result = evaluate(args.detector, args.dataset, args.seed, args.checkpoint, topk_grid=topk_grid)
    print(json.dumps(result, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\n[write] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
