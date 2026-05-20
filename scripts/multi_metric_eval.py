"""
H_PI_3 multi-metric evaluation framework: replaces FTA as the gating
attack-effectiveness metric with a multi-metric set:

  1. Raw F1 reduction       (no-attack F1 minus attacked F1)
  2. PA-F1 reduction        (point-adjusted F1, K=10 aggregation)
  3. AUC-PR drop            (precision-recall area under curve)
  4. Continuous target-score degradation
     (mean(clean_score - attacked_score) at attacked targets)

The continuous target-score degradation is the most important — it's
metric-agnostic and provides a smooth signal that Phase 1's PPO policy
can use directly in the reward.

Authorized via dec_01KQPGTTJZX1FE3H12Y7FSDYGX (Phase 0 closeout).
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
from sklearn.metrics import f1_score, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the sanity-check setup helpers
from scripts.beta_sanity_check import (  # type: ignore
    setup_victim, collect_residuals_per_sensor, standardize_score,
)


def adjust_predicts(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Xu et al. 2018 point-adjust: any-point-in-segment-detected => all-points-in-segment-detected."""
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
    if in_seg and pred[seg_start:n].any():
        pred[seg_start:n] = 1
    return pred


def aggregate_topk(per_sensor_scores: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-K-sum aggregation across sensors. Returns (T,) score per timestep."""
    if k == 1:
        return per_sensor_scores.max(axis=1)
    topk_idx = np.argpartition(per_sensor_scores, -k, axis=1)[:, -k:]
    return np.take_along_axis(per_sensor_scores, topk_idx, axis=1).sum(axis=1)


def compute_f1_metrics(scores_1d: np.ndarray, labels: np.ndarray, n_threshold_steps: int = 400):
    """Return (best_raw_F1, best_PA_F1, AUC-PR) over a rank-percentile sweep."""
    from scipy.stats import rankdata
    ranked = rankdata(scores_1d, method="ordinal")
    n = len(scores_1d)
    best_raw = 0.0
    best_pa = 0.0
    for i in range(n_threshold_steps):
        thr = (i / n_threshold_steps) * n
        pred = (ranked > thr).astype(int)
        raw_f1 = f1_score(labels, pred, zero_division=0)
        pa_pred = adjust_predicts(pred, labels)
        pa_f1 = f1_score(labels, pa_pred, zero_division=0)
        if raw_f1 > best_raw:
            best_raw = raw_f1
        if pa_f1 > best_pa:
            best_pa = pa_f1

    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, scores_1d)
        auc_pr = float(auc(recall, precision))
    else:
        auc_pr = float("nan")

    return float(best_raw), float(best_pa), auc_pr


def evaluate(detector: str, seed: int, n_attack_trials: int, budget: int = 5):
    """Multi-metric evaluation on existing checkpoint."""
    print(f"[setup] loading {detector.upper()} seed {seed}")
    m = setup_victim(detector, seed=seed)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr

    print(f"[stats] computing per-sensor median + IQR on validation set")
    val_deltas = collect_residuals_per_sensor(model, val_loader, detector)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    flat_labels = torch.cat([b[2].float() for b in test_batches], dim=0).numpy().astype(int)

    n_sensors = flat_xs.shape[1]
    print(f"[shape] test windows={flat_xs.shape[0]}, sensors={n_sensors}, "
          f"anomaly windows={int(flat_labels.sum())} ({100*flat_labels.mean():.2f}%)")

    # ──────────────── CLEAN BASELINE ────────────────
    print(f"[clean] computing clean per-sensor scores over test set ...")
    clean_scores_per_sensor = []
    BATCH = 256
    with torch.no_grad():
        for start in range(0, flat_xs.shape[0], BATCH):
            X = flat_xs[start:start + BATCH]
            if detector == "topogdn":
                forecast, _ = model(X)
            else:
                edge_index = test_batches[0][3].float()
                forecast = model(X, edge_index)
            target_step = X[..., -1]
            s = standardize_score(forecast, target_step, median, iqr_v)
            clean_scores_per_sensor.append(s.cpu().numpy())
    clean_scores_per_sensor = np.concatenate(clean_scores_per_sensor, axis=0)  # (T, N)

    clean_top10 = aggregate_topk(clean_scores_per_sensor, k=10)
    clean_max = aggregate_topk(clean_scores_per_sensor, k=1)
    clean_raw_f1, clean_pa_f1, clean_auc_pr = compute_f1_metrics(clean_top10, flat_labels)

    print(f"[clean] raw_F1={clean_raw_f1:.4f}  PA_F1={clean_pa_f1:.4f}  AUC_PR={clean_auc_pr:.4f}")

    # ──────────────── ATTACKED ────────────────
    # We attack a sample of detected windows (where max sensor score > max-val threshold)
    # using BETA reimplementation, then PATCH the attacked window's per-sensor scores into
    # the test_scores array, recompute F1 metrics on the patched array.
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        with torch.no_grad():
            if detector == "topogdn":
                f, _ = model(x.float())
            else:
                f = model(x.float(), ei.float())
        s = standardize_score(f, y.float(), median, iqr_v)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold_max = float(val_scores_arr.max())

    detected_windows = []
    for i in range(clean_scores_per_sensor.shape[0]):
        if clean_max[i] > threshold_max:
            target = int(np.argmax(clean_scores_per_sensor[i]))
            detected_windows.append((i, target))

    print(f"[detect] detected {len(detected_windows)} test windows; "
          f"sampling {min(n_attack_trials, len(detected_windows))} for attack")

    rng = random.Random(seed)
    sample = rng.sample(detected_windows, min(n_attack_trials, len(detected_windows)))

    from attacks.beta import BETAAttack, BETAConfig

    edge_index_template = test_batches[0][3].float()

    def victim_forward(X: torch.Tensor) -> torch.Tensor:
        if detector == "topogdn":
            forecast, _ = model(X)
        else:
            forecast = model(X, edge_index_template)
        target_step = X[..., -1]
        return standardize_score(forecast, target_step, median, iqr_v)

    def learned_edge_index_fn():
        return model.learned_graph.detach()

    cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    attack = BETAAttack(victim_forward, learned_edge_index_fn, num_nodes=n_sensors, config=cfg)

    # Patched scores: clone clean scores, replace attacked windows' rows.
    attacked_scores_per_sensor = clean_scores_per_sensor.copy()

    target_score_degradations = []
    print(f"[attack] running BETA on {len(sample)} (window, target=argmax) pairs ...")
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1]
        try:
            X_pert, _ = attack.attack(X.clone(), target_idx, budget)
        except Exception as e:
            print(f"  trial {trial_i}: ATTACK FAILED ({type(e).__name__}: {e})")
            continue
        with torch.no_grad():
            attacked_scores_w = victim_forward(X_pert)[0].cpu().numpy()
        # patch the test scores
        attacked_scores_per_sensor[w_idx] = attacked_scores_w
        # continuous target-score degradation
        target_score_degradations.append(
            float(clean_scores_per_sensor[w_idx, target_idx] - attacked_scores_w[target_idx])
        )
        if (trial_i + 1) % 25 == 0:
            print(f"  trial {trial_i+1}/{len(sample)}  mean_target_degradation="
                  f"{np.mean(target_score_degradations):.4f}")

    attacked_top10 = aggregate_topk(attacked_scores_per_sensor, k=10)
    attacked_raw_f1, attacked_pa_f1, attacked_auc_pr = compute_f1_metrics(attacked_top10, flat_labels)

    print(f"[attacked] raw_F1={attacked_raw_f1:.4f}  PA_F1={attacked_pa_f1:.4f}  AUC_PR={attacked_auc_pr:.4f}")

    return {
        "detector": detector,
        "seed": seed,
        "n_attack_trials_sampled": len(sample),
        "n_attack_trials_completed": len(target_score_degradations),
        "budget": budget,
        "n_test_windows": int(flat_xs.shape[0]),
        "n_anomaly_windows": int(flat_labels.sum()),
        "n_detected_windows": len(detected_windows),
        # ─── Clean metrics (baseline) ───
        "clean": {
            "raw_F1_best": clean_raw_f1,
            "PA_F1_best": clean_pa_f1,
            "AUC_PR": clean_auc_pr,
        },
        # ─── Attacked metrics (BETA reimpl applied to sampled detected windows) ───
        "attacked": {
            "raw_F1_best": attacked_raw_f1,
            "PA_F1_best": attacked_pa_f1,
            "AUC_PR": attacked_auc_pr,
        },
        # ─── Reductions (the headline numbers) ───
        "reduction": {
            "raw_F1": clean_raw_f1 - attacked_raw_f1,
            "PA_F1": clean_pa_f1 - attacked_pa_f1,
            "AUC_PR": clean_auc_pr - attacked_auc_pr,
        },
        # ─── Continuous target-score degradation (Phase 1 PPO reward signal) ───
        "target_score_degradation": {
            "mean": float(np.mean(target_score_degradations)) if target_score_degradations else None,
            "median": float(np.median(target_score_degradations)) if target_score_degradations else None,
            "std": float(np.std(target_score_degradations)) if target_score_degradations else None,
            "min": float(np.min(target_score_degradations)) if target_score_degradations else None,
            "max": float(np.max(target_score_degradations)) if target_score_degradations else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["gdn", "topogdn", "gdn_pyg1x"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=200, help="number of (window, target) attack trials")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(args.detector, args.seed, args.n_trials, args.budget)
    print(json.dumps(result, indent=2, default=str))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # Resolve to absolute path before any chdir side effects from setup_victim.
        out_path = Path(args.out).resolve() if Path(args.out).is_absolute() else (PROJECT_ROOT / args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n[write] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
