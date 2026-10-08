"""
Deployment-calibration evaluation (paper Section VI-D): does the OOD-clamp finding and the
symmetry repair survive DEPLOYMENT-REALISTIC thresholds that never peek at test
labels?

The paper's adaptive section reports F1 at the F1-optimal threshold, which is
chosen with test labels. A deployed detector cannot do that. We therefore set
the alarm threshold three ways, all from BENIGN (validation) scores only:

  - val-max:       max of benign per-window aggregated scores (the strictest
                   benign threshold; what BETA's setup implicitly used).
  - benign-q:      a high percentile (default 99.5%) of benign scores.
  - EVT / POT:     extreme-value (peaks-over-threshold, Siffer et al. KDD 2017)
                   tail model fit to benign scores, set for a target false-alarm
                   probability q (default 1e-4). GPD fit by method of moments.

For GDN on WADI/SWaT we then report, at each threshold, the test-set F1 /
precision / recall / FPR for the UNDEFENDED pipeline (no input clip) vs the
REPAIRED pipeline (symmetric clip of inputs). The deployment question the repair
must pass: clipping clean inputs too (the repair) must NOT degrade benign-time
detection under a label-free threshold.

Thresholds use benign scores only; F1/precision/recall/FPR use test labels for
REPORTING (standard) but never for threshold selection.

Usage:
  python scripts/calibration_eval.py --dataset wadi --seed 0 --out reports/calibration_wadi_seed0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.beta_sanity_check import (  # noqa: E402
    setup_victim, collect_residuals_per_sensor, standardize_score,
)


def aggregate_topk(per_sensor: np.ndarray, k: int = 10) -> np.ndarray:
    if k == 1:
        return per_sensor.max(axis=1)
    idx = np.argpartition(per_sensor, -k, axis=1)[:, -k:]
    return np.take_along_axis(per_sensor, idx, axis=1).sum(axis=1)


def evt_pot_threshold(benign: np.ndarray, init_q: float = 0.98, target_q: float = 1e-4) -> float:
    """Peaks-over-threshold EVT threshold (Siffer et al. KDD 2017), GPD via
    method of moments. target_q = desired false-alarm probability."""
    n = len(benign)
    t = float(np.quantile(benign, init_q))
    exc = benign[benign > t] - t
    nt = len(exc)
    if nt < 10:
        return float(benign.max())
    m, v = float(exc.mean()), float(exc.var())
    if v <= 1e-12:
        return float(benign.max())
    gamma = 0.5 * (1.0 - m * m / v)         # GPD shape
    sigma = 0.5 * m * (m * m / v + 1.0)     # GPD scale
    ratio = target_q * n / nt
    if abs(gamma) < 1e-6:                    # exponential-tail limit
        z = t - sigma * np.log(ratio)
    else:
        z = t + (sigma / gamma) * (ratio ** (-gamma) - 1.0)
    return float(z)


def metrics_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = (scores > thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "fpr": fpr,
            "n_alarms": int(pred.sum())}


def score_test(model, detector, test_batches, median, iqr_v, clip: bool):
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    if clip:
        flat_xs = flat_xs.clamp(0.0, 1.0)
    edge_index = test_batches[0][3].float()
    out, BATCH = [], 256
    with torch.no_grad():
        for s0 in range(0, flat_xs.shape[0], BATCH):
            X = flat_xs[s0:s0 + BATCH]
            forecast = model(X)[0] if detector == "topogdn" else model(X, edge_index)
            out.append(standardize_score(forecast, X[..., -1], median, iqr_v).cpu().numpy())
    return aggregate_topk(np.concatenate(out, axis=0), k=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--detector", default="gdn", choices=["gdn", "topogdn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benign-q", type=float, default=99.5)
    p.add_argument("--evt-q", type=float, default=1e-4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from scipy.stats import iqr as scipy_iqr
    m = setup_victim(args.detector, seed=args.seed, dataset=args.dataset)
    model = m.model

    val_deltas = collect_residuals_per_sensor(model, m.val_dataloader, args.detector)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    # Benign (validation) aggregated scores -> label-free thresholds.
    val_batches = list(m.val_dataloader)
    val_edge = val_batches[0][3].float()
    val_ps, BATCH = [], 256
    with torch.no_grad():
        for b in val_batches:
            X = b[0].float()
            forecast = model(X)[0] if args.detector == "topogdn" else model(X, val_edge)
            val_ps.append(standardize_score(forecast, X[..., -1], median, iqr_v).cpu().numpy())
    val_top10 = aggregate_topk(np.concatenate(val_ps, axis=0), k=10)

    thresholds = {
        "val_max": float(val_top10.max()),
        f"benign_p{args.benign_q}": float(np.percentile(val_top10, args.benign_q)),
        "evt_pot": evt_pot_threshold(val_top10, target_q=args.evt_q),
    }
    print(f"[thresh] {json.dumps({k: round(v,4) for k,v in thresholds.items()})}")

    test_batches = list(m.test_dataloader)
    labels = torch.cat([b[2].float() for b in test_batches], dim=0).numpy().astype(int)

    results = {"dataset": args.dataset, "detector": args.detector, "seed": args.seed,
               "thresholds": thresholds, "n_test": int(len(labels)),
               "anomaly_rate": float(labels.mean()), "pipelines": {}}

    for name, clip in [("undefended", False), ("repaired", True)]:
        scores = score_test(model, args.detector, test_batches, median, iqr_v, clip)
        results["pipelines"][name] = {
            tname: metrics_at(scores, labels, thr) for tname, thr in thresholds.items()
        }

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[write] {out}")
    for tname in thresholds:
        u = results["pipelines"]["undefended"][tname]
        r = results["pipelines"]["repaired"][tname]
        print(f"  {tname:16s}  undef F1={u['f1']:.3f} FPR={u['fpr']:.4f}  |  "
              f"repaired F1={r['f1']:.3f} FPR={r['fpr']:.4f}")


if __name__ == "__main__":
    main()
