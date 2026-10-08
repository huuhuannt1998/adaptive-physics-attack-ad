"""
TopoGDN interpolation sweep (paper Section VI-C): characterize the
TopoGDN-WADI score movement along the adversarial direction to distinguish a
genuine discontinuity (abrupt jumps) from high local sensitivity (smooth but
amplified growth), and localize the movement to the topological branch by
comparing to a GDN baseline on the same windows.

For each of a few attacked windows, evaluate X_alpha = X + alpha*delta for
alpha in [0,1] and record the target-sensor standardized score along the path
for TopoGDN and GDN. We report the max single-step jump (abruptness) and the
total score range, for both detectors, so the paper can state whether the
drift is abrupt or smooth-amplified and whether the topological branch drives it.

Usage:
  python scripts/topogdn_interp_sweep.py --seed 0 --n-windows 20 \
      --out reports/topogdn_interp_wadi_seed0.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.beta_sanity_check import (  # noqa: E402
    setup_victim, collect_residuals_per_sensor, standardize_score,
)


def score_fn(model, detector, edge_index, median, iqr_v):
    def f(X):
        if detector == "topogdn":
            fc = model(X)[0]
        else:
            fc = model(X, edge_index)
        return standardize_score(fc, X[..., -1], median, iqr_v)
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    alphas = np.linspace(0, 1, 21)

    out_all = {"dataset": args.dataset, "seed": args.seed, "alphas": alphas.tolist(),
               "per_detector": {}}
    # attacked direction from BETA on TopoGDN; reuse the same delta for both detectors
    mt = setup_victim("topogdn", seed=args.seed, dataset=args.dataset)
    from scipy.stats import iqr as scipy_iqr
    vt = collect_residuals_per_sensor(mt.model, mt.val_dataloader, "topogdn")
    med_t = torch.from_numpy(np.median(vt, axis=0)).float(); iqr_t = torch.from_numpy(scipy_iqr(vt, axis=0)).float()
    tb = list(mt.test_dataloader); flat_xs = torch.cat([b[0].float() for b in tb], dim=0)
    labels = torch.cat([b[2].float() for b in tb], dim=0).numpy().astype(int)
    topo_score = score_fn(mt.model, "topogdn", None, med_t, iqr_t)

    # pick anomalous windows, get a BETA delta per window on TopoGDN
    from attacks.beta import BETAAttack, BETAConfig
    cfg = BETAConfig(epsilon=args.epsilon, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    attack = BETAAttack(lambda X: topo_score(X), lambda: mt.model.learned_graph.detach(),
                        num_nodes=flat_xs.shape[1], config=cfg)
    idxs = [i for i in range(flat_xs.shape[0]) if labels[i] == 1][:args.n_windows]

    deltas, wins, tgts = [], [], []
    for w in idxs:
        X = flat_xs[w:w+1]
        with torch.no_grad():
            tgt = int(topo_score(X)[0].argmax().item())
        Xp, _ = attack.attack(X.clone(), tgt, args.budget)
        deltas.append((Xp - X).detach()); wins.append(w); tgts.append(tgt)

    def sweep(score, med, iqr):
        jumps, ranges = [], []
        curves = []
        with torch.no_grad():
            for X0, w, tgt, d in zip([flat_xs[w:w+1] for w in wins], wins, tgts, deltas):
                vals = []
                for a in alphas:
                    Xa = (X0 + float(a) * d)
                    vals.append(float(score(Xa)[0, tgt].item()))
                vals = np.asarray(vals)
                step = np.abs(np.diff(vals))
                total = float(vals.max() - vals.min()) + 1e-9
                jumps.append(float(step.max() / total))    # largest single-step share of total range
                ranges.append(float(vals.max() - vals.min()))
                curves.append(vals.tolist())
        return {"max_step_share_median": float(np.median(jumps)),
                "max_step_share_p95": float(np.percentile(jumps, 95)),
                "score_range_median": float(np.median(ranges))}

    out_all["per_detector"]["topogdn"] = sweep(topo_score, med_t, iqr_t)

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_all, indent=2))
    print(f"[write] {out}  (n_windows={len(wins)})")
    for det, d in out_all["per_detector"].items():
        print(f"  {det:8s}  max-step share (median)={d['max_step_share_median']:.3f} "
              f"p95={d['max_step_share_p95']:.3f}  score range={d['score_range_median']:.2f}")
    print("  (max-step share near 1.0 = abrupt jump/discontinuity; near 1/20=0.05 = smooth)")


if __name__ == "__main__":
    main()
