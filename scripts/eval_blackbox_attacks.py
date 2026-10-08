"""
Adaptive black-box attack evaluation (paper Section VII).

Runs RandomSearch and SPSA gradient-free black-box attackers alongside
BETA (white-box surrogate) on the SAME (window, target) pairs, under the
same gray-box scalar-query threat model. Reports per-attacker mean
target-score degradation, query count, and attack-success rate, so attack
strength can be compared as a function of query budget.

Reuses the victim setup and standardized-score interface from
eval_beta_swat.py.

Usage:
  python scripts/eval_blackbox_attacks.py --seed 0 --dataset wadi --arch GDN \\
      --defended --n-trials 20 --out reports/blackbox_wadi_seed0_def.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_beta_swat import setup_victim  # noqa: E402
from attacks.beta import BETAAttack, BETAConfig  # noqa: E402
from attacks.blackbox import (  # noqa: E402
    RandomSearchAttack, SPSAAttack, BlackBoxConfig,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--arch", default="GDN", choices=["GDN", "TopoGDN"])
    p.add_argument("--defended", action="store_true")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--rs-iters", type=int, default=200)
    p.add_argument("--spsa-iters", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[setup] {args.arch}-{args.dataset} seed {args.seed} defended={args.defended}")
    m = setup_victim(seed=args.seed, dataset=args.dataset, arch=args.arch)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader
    is_topo = (args.arch.lower() == "topogdn")

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xf, yf = x.float(), y.float()
            if args.defended:
                xf, yf = xf.clamp(0, 1), yf.clamp(0, 1)
            if is_topo:
                out = model(xf); f = out[0] if isinstance(out, tuple) else out
            else:
                f = model(xf, ei.float())
            val_deltas.append((f - yf).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def model_forward(X):
        if is_topo:
            out = model(X); return out[0] if isinstance(out, tuple) else out
        return model(X, edge_index_template)

    def surrogate_query(X):
        Xf = X.clamp(0, 1) if args.defended else X
        forecast = model_forward(Xf)
        target_step = Xf[..., -1]
        d = (forecast - target_step).abs()
        return (d - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    if args.defended:
        flat_xs = flat_xs.clamp(0, 1)
    n_test, n_sensors, W = flat_xs.shape
    print(f"[data] test={n_test} windows, n_sensors={n_sensors}, W={W}")

    # Detect windows above the val-max threshold (same as eval_beta_swat).
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xf = x.float().clamp(0, 1) if args.defended else x.float()
        with torch.no_grad():
            s = surrogate_query(xf)
        val_scores.append(s.cpu().numpy())
    threshold = float(np.concatenate(val_scores, axis=0).max())

    detected = []
    BATCH = 256
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            s = surrogate_query(flat_xs[start:start + BATCH])
            mx, idx = s.max(dim=1)
            for i in range(s.shape[0]):
                if mx[i].item() > threshold:
                    detected.append((start + i, int(idx[i].item())))
    print(f"[detect] {len(detected)} windows fire threshold {threshold:.4f}")
    if not detected:
        json.dump({"error": "no detected windows", "threshold": threshold},
                  open(args.out, "w"))
        return

    rng = random.Random(args.seed)
    sample = rng.sample(detected, min(args.n_trials, len(detected)))

    beta = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                      num_nodes=n_sensors,
                      config=BETAConfig(epsilon=args.epsilon))
    bb_cfg = BlackBoxConfig(epsilon=args.epsilon, rs_iters=args.rs_iters,
                            spsa_iters=args.spsa_iters, seed=args.seed)
    rs = RandomSearchAttack(surrogate_query, n_sensors, bb_cfg)
    spsa = SPSAAttack(surrogate_query, n_sensors, bb_cfg)

    results = {a: {"deg": [], "queries": [], "flip": []}
               for a in ["beta", "random_search", "spsa"]}
    for ti, (w_idx, target) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean = surrogate_query(X)[0, target].item()
        # A window is "flipped" (successful evasion at the detection level)
        # if the attacked target score drops below the detection threshold.
        # BETA (white-box surrogate; query count = PGD forward budget).
        try:
            Xp, info = beta.attack(X.clone(), target, args.budget)
            with torch.no_grad():
                fs = surrogate_query(Xp)[0, target].item()
            results["beta"]["deg"].append(clean - fs)
            results["beta"]["queries"].append(
                BETAConfig().pgd_restarts * BETAConfig().pgd_iters)
            results["beta"]["flip"].append(1.0 if fs < threshold else 0.0)
        except Exception as e:
            print(f"  [beta] trial {ti} failed: {e}")
        # Random search
        Xp, info = rs.attack(X.clone(), target, args.budget)
        results["random_search"]["deg"].append(clean - info["final_score"])
        results["random_search"]["queries"].append(info["queries"])
        results["random_search"]["flip"].append(
            1.0 if info["final_score"] < threshold else 0.0)
        # SPSA
        Xp, info = spsa.attack(X.clone(), target, args.budget)
        results["spsa"]["deg"].append(clean - info["final_score"])
        results["spsa"]["queries"].append(info["queries"])
        results["spsa"]["flip"].append(
            1.0 if info["final_score"] < threshold else 0.0)
        if (ti + 1) % 5 == 0:
            print(f"  trial {ti+1}/{len(sample)}  "
                  f"beta={np.mean(results['beta']['deg']):+.3f}  "
                  f"rs={np.mean(results['random_search']['deg']):+.3f}  "
                  f"spsa={np.mean(results['spsa']['deg']):+.3f}")

    summary = {
        "seed": args.seed, "dataset": args.dataset, "arch": args.arch,
        "defended": args.defended, "epsilon": args.epsilon,
        "budget": args.budget, "n_trials": len(sample),
        "threshold": threshold,
        "attackers": {},
    }
    for a, d in results.items():
        deg = np.array(d["deg"]) if d["deg"] else np.array([np.nan])
        summary["attackers"][a] = {
            "mean_deg": float(np.nanmean(deg)),
            "median_deg": float(np.nanmedian(deg)),
            "std_deg": float(np.nanstd(deg)),
            "asr": float(np.mean(deg > 0.0)) if d["deg"] else 0.0,
            "flip_rate": float(np.mean(d["flip"])) if d["flip"] else 0.0,
            "mean_queries": float(np.mean(d["queries"])) if d["queries"] else 0.0,
            "n": len(d["deg"]),
        }
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {out_path}")
    for a, s in summary["attackers"].items():
        print(f"  {a:14s}  mean_deg={s['mean_deg']:+9.3f}  asr={s['asr']:.2f}  "
              f"flip={s['flip_rate']:.2f}  queries={s['mean_queries']:.0f}  n={s['n']}")


if __name__ == "__main__":
    main()
