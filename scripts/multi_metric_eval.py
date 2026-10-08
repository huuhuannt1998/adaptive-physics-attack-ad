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
    """Return (best_raw_F1, best_PA_F1, AUC-PR, best_raw_thresh, best_pa_thresh) over a rank-percentile sweep."""
    from scipy.stats import rankdata
    ranked = rankdata(scores_1d, method="ordinal")
    n = len(scores_1d)
    best_raw = 0.0
    best_pa = 0.0
    best_raw_thresh = None
    best_pa_thresh = None
    for i in range(n_threshold_steps):
        thr = (i / n_threshold_steps) * n
        pred = (ranked > thr).astype(int)
        raw_f1 = f1_score(labels, pred, zero_division=0)
        pa_pred = adjust_predicts(pred, labels)
        pa_f1 = f1_score(labels, pa_pred, zero_division=0)
        if raw_f1 > best_raw:
            best_raw = raw_f1
            # Convert rank threshold -> raw score threshold via inverse rank.
            below = scores_1d[ranked <= thr]
            best_raw_thresh = float(below.max()) if below.size else float(scores_1d.min())
        if pa_f1 > best_pa:
            best_pa = pa_f1
            below = scores_1d[ranked <= thr]
            best_pa_thresh = float(below.max()) if below.size else float(scores_1d.min())

    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, scores_1d)
        auc_pr = float(auc(recall, precision))
    else:
        auc_pr = float("nan")

    return float(best_raw), float(best_pa), auc_pr, best_raw_thresh, best_pa_thresh


def evaluate(detector: str, seed: int, n_attack_trials: int, budget: int = 5,
             agg_k: int = 10,
             defended: bool = False, hk_defense: bool = False,
             hk_refs_path: str = None, hk_projection_radius: float = 2.0,
             hk_sigma: float = 0.05, hk_persistence_clamp: float = 1.0,
             dataset: str = None, attacker: str = "beta"):
    """Multi-metric evaluation on existing checkpoint.

    Args:
      defended: if True, pre-clip all inputs (clean + attacked) to [0,1]
        before the detector forward, matching the input-clipping defense.
      hk_defense: if True, install HKStabilityWrapper on the model's
        TopologyLayer instances (requires detector='topogdn').
      hk_refs_path: path to clean PI reference .pt. Default infers from
        dataset and seed.
    """
    print(f"[setup] loading {detector.upper()} seed {seed}  defended={defended}  hk={hk_defense}")
    m = setup_victim(detector, seed=seed, dataset=(dataset or "wadi"))
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    if hk_defense:
        if detector != "topogdn":
            raise ValueError("--hk-defense requires --detector topogdn")
        from defenses.hk_stability import HKConfig, HKStabilityWrapper
        ds = dataset or ("wadi" if "wadi" in str(m.env_config.get("save_path", "")).lower() else None)
        if ds is None:
            raise ValueError("Cannot infer dataset for HK refs path; pass --dataset")
        if hk_refs_path is None:
            hk_refs_path = f"reports/hk_refs/{ds}_seed{seed}.pt"
        from pathlib import Path
        refs_path_full = hk_refs_path if Path(hk_refs_path).is_absolute() else str(PROJECT_ROOT / hk_refs_path)
        print(f"[hk] loading reference PIs from {refs_path_full}")
        refs = torch.load(refs_path_full, weights_only=False, map_location="cpu")
        ref_p0 = refs["ref_p0"]
        hk_cfg = HKConfig(
            grid_size=ref_p0.shape[-1],
            sigma=hk_sigma,
            persistence_clamp=hk_persistence_clamp,
            projection_radius=hk_projection_radius,
        )
        wrapper = HKStabilityWrapper(model, clean_references=ref_p0, config=hk_cfg)
        n_patched = 0
        for _name, mod in model.named_modules():
            if mod.__class__.__name__ == "TopologyLayer":
                wrapper._install_hook(mod)
                n_patched += 1
        print(f"[hk] patched {n_patched} TopologyLayer instance(s) "
              f"(radius={hk_cfg.projection_radius}, sigma={hk_cfg.sigma})")

    def maybe_clip(t):
        return t.clamp(0.0, 1.0) if defended else t

    from scipy.stats import iqr as scipy_iqr

    print(f"[stats] computing per-sensor median + IQR on validation set")
    val_deltas = collect_residuals_per_sensor(model, val_loader, detector)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    flat_xs = maybe_clip(flat_xs)
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

    clean_top10 = aggregate_topk(clean_scores_per_sensor, k=agg_k)
    clean_max = aggregate_topk(clean_scores_per_sensor, k=1)
    clean_raw_f1, clean_pa_f1, clean_auc_pr, clean_raw_thresh, clean_pa_thresh = compute_f1_metrics(clean_top10, flat_labels)

    print(f"[clean] raw_F1={clean_raw_f1:.4f}  PA_F1={clean_pa_f1:.4f}  AUC_PR={clean_auc_pr:.4f}")

    # ──────────────── ATTACKED ────────────────
    # We attack a sample of detected windows (where max sensor score > max-val threshold)
    # using BETA reimplementation, then PATCH the attacked window's per-sensor scores into
    # the test_scores array, recompute F1 metrics on the patched array.
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xf = maybe_clip(x.float())
        yf = maybe_clip(y.float())
        with torch.no_grad():
            if detector == "topogdn":
                f, _ = model(xf)
            else:
                f = model(xf, ei.float())
        s = standardize_score(f, yf, median, iqr_v)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold_max = float(val_scores_arr.max())

    # Attack-window selection for evasion F1-impact: target the TRUE-POSITIVE
    # windows at the F1-optimal threshold (anomalous windows the defended
    # detector correctly flags). Driving these below threshold reduces recall
    # and therefore F1 --- the realistic evasion threat. The val-max threshold
    # finds ~0 windows on a defended detector, so we use the F1-optimal
    # threshold (clean_raw_thresh) on the top-10-aggregated clean score.
    detected_windows = []
    for i in range(clean_scores_per_sensor.shape[0]):
        is_true_positive = (flat_labels[i] == 1) and (clean_top10[i] > clean_raw_thresh)
        if is_true_positive:
            target = int(np.argmax(clean_scores_per_sensor[i]))
            detected_windows.append((i, target))
    # Fallback: if no F1-opt true positives (degenerate F1), retain the
    # val-max-detected windows so the run is not silently vacuous.
    if not detected_windows:
        for i in range(clean_scores_per_sensor.shape[0]):
            if clean_max[i] > threshold_max:
                target = int(np.argmax(clean_scores_per_sensor[i]))
                detected_windows.append((i, target))

    print(f"[detect] {len(detected_windows)} F1-opt true-positive windows "
          f"(thresh={clean_raw_thresh:.4f}); "
          f"sampling {min(n_attack_trials, len(detected_windows))} for attack")

    rng = random.Random(seed)
    sample = rng.sample(detected_windows, min(n_attack_trials, len(detected_windows)))

    from attacks.beta import BETAAttack, BETAConfig

    edge_index_template = test_batches[0][3].float()

    def victim_forward(X: torch.Tensor) -> torch.Tensor:
        Xc = maybe_clip(X)
        if detector == "topogdn":
            forecast, _ = model(Xc)
        else:
            forecast = model(Xc, edge_index_template)
        target_step = Xc[..., -1]
        return standardize_score(forecast, target_step, median, iqr_v)

    def learned_edge_index_fn():
        return model.learned_graph.detach()

    if attacker == "beta":
        cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
        attack = BETAAttack(victim_forward, learned_edge_index_fn, num_nodes=n_sensors, config=cfg)
    elif attacker in ("random_search", "spsa"):
        from attacks.blackbox import RandomSearchAttack, SPSAAttack, BlackBoxConfig
        bb_cfg = BlackBoxConfig(epsilon=0.1, rs_iters=200, spsa_iters=50, seed=seed)
        attack = (RandomSearchAttack if attacker == "random_search" else SPSAAttack)(
            victim_forward, n_sensors, bb_cfg)
    else:
        raise ValueError(f"unknown attacker: {attacker}")

    # Patched scores: clone clean scores, replace attacked windows' rows.
    attacked_scores_per_sensor = clean_scores_per_sensor.copy()

    target_score_degradations = []
    print(f"[attack] running {attacker} on {len(sample)} (window, target=argmax) pairs ...")
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

    attacked_top10 = aggregate_topk(attacked_scores_per_sensor, k=agg_k)
    attacked_raw_f1, attacked_pa_f1, attacked_auc_pr, attacked_raw_thresh, attacked_pa_thresh = compute_f1_metrics(attacked_top10, flat_labels)

    print(f"[attacked] raw_F1={attacked_raw_f1:.4f}  PA_F1={attacked_pa_f1:.4f}  AUC_PR={attacked_auc_pr:.4f}")

    return {
        "detector": detector,
        "seed": seed,
        "attacker": attacker,
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
            "raw_F1_threshold": clean_raw_thresh,
            "PA_F1_threshold": clean_pa_thresh,
        },
        # ─── Attacked metrics (BETA reimpl applied to sampled detected windows) ───
        "attacked": {
            "raw_F1_best": attacked_raw_f1,
            "PA_F1_best": attacked_pa_f1,
            "AUC_PR": attacked_auc_pr,
            "raw_F1_threshold": attacked_raw_thresh,
            "PA_F1_threshold": attacked_pa_thresh,
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
    parser.add_argument("--defended", action="store_true",
                        help="Pre-clip all inputs to [0,1] (input-clipping defense).")
    parser.add_argument("--hk-defense", action="store_true",
                        help="(TopoGDN only) Install HK PI-stability defense.")
    parser.add_argument("--hk-refs", default=None,
                        help="Path to clean PI reference .pt; default: reports/hk_refs/{dataset}_seed{seed}.pt")
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"],
                        help="Dataset (used to infer HK refs path).")
    parser.add_argument("--hk-projection-radius", type=float, default=2.0)
    parser.add_argument("--hk-sigma", type=float, default=0.05)
    parser.add_argument("--hk-persistence-clamp", type=float, default=1.0)
    parser.add_argument("--attacker", default="beta",
                        choices=["beta", "random_search", "spsa"],
                        help="Attack algorithm to evaluate.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(args.detector, args.seed, args.n_trials, args.budget,
                      defended=args.defended, hk_defense=args.hk_defense,
                      hk_refs_path=args.hk_refs,
                      hk_projection_radius=args.hk_projection_radius,
                      hk_sigma=args.hk_sigma,
                      hk_persistence_clamp=args.hk_persistence_clamp,
                      dataset=args.dataset, attacker=args.attacker)
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
