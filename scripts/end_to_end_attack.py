"""
End-to-end adaptive attack (paper Section VII, decision-level attacker): the decisive test of the
defense claim. Unlike the target-sensor attackers, this one minimizes the
detector's actual top-k aggregate decision score S_k(g(X+delta)) directly,
under the same ell_inf epsilon and sensor budget b, on the DEFENDED
(symmetric-clip) detector.

Objective per window: minimize S_k(g(X+delta)) = sum of the top-k per-sensor
scores, with ||delta||_inf <= eps on a query-selected set of <= b sensors.
Gradient-free random search + a sign-coordinate refinement; every victim
query is counted. We then patch attacked windows' top-k scores back into the
test stream and recompute F1 at the F1-optimal threshold.

If F1 drops materially, the corrected detector is evadable end-to-end and
the defense claim must stay narrow. If F1 stays ~unchanged even under this
decision-level objective, the correction bounds capability more strongly
than the target-sensor attacks alone showed.

Usage:
  python scripts/end_to_end_attack.py --dataset wadi --seed 0 --n-trials 40 \
      --k 10 --budget 5 --iters 400 --out reports/e2e_wadi_seed0.json
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.beta_sanity_check import (  # noqa: E402
    setup_victim, collect_residuals_per_sensor, standardize_score,
)
from scripts.multi_metric_eval import aggregate_topk, compute_f1_metrics  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--threshold", default="f1opt", choices=["f1opt", "benign", "evt"])
    p.add_argument("--fpr-target", type=float, default=0.05)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    eps, K, B = args.epsilon, args.k, args.budget

    m = setup_victim("gdn", seed=args.seed, dataset=args.dataset)
    model = m.model
    from scipy.stats import iqr as scipy_iqr
    val_deltas = collect_residuals_per_sensor(model, m.val_dataloader, "gdn")
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    test_batches = list(m.test_dataloader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    flat_labels = torch.cat([b[2].float() for b in test_batches], dim=0).numpy().astype(int)
    edge_index = test_batches[0][3].float()

    def scores_vec(X):  # per-sensor standardized score vector on DEFENDED input
        Xc = X.clamp(0.0, 1.0)
        return standardize_score(model(Xc, edge_index), Xc[..., -1], median, iqr_v)[0]

    def Sk(X):          # top-k aggregate decision score (scalar), + query count via caller
        s = scores_vec(X)
        return float(s.topk(K).values.sum().item())

    # clean scores + F1-opt threshold + detected TPs
    BATCH, clean_ps = 256, []
    with torch.no_grad():
        for s0 in range(0, flat_xs.shape[0], BATCH):
            X = flat_xs[s0:s0 + BATCH].clamp(0.0, 1.0)
            clean_ps.append(standardize_score(model(X, edge_index), X[..., -1],
                                              median, iqr_v).cpu().numpy())
    clean_ps = np.concatenate(clean_ps, axis=0)
    clean_topk = aggregate_topk(clean_ps, k=K)
    clean_f1, _, _, thr_f1opt, _ = compute_f1_metrics(clean_topk, flat_labels)
    # deployable thresholds from benign (label==0) test-stream scores only
    benign = clean_topk[flat_labels == 0]
    thr_benign = float(np.quantile(benign, 1.0 - args.fpr_target))
    # EVT peaks-over-threshold (GPD via method of moments) at target FA prob
    def evt_thr(x, init_q=0.98, target_q=1e-4):
        t = float(np.quantile(x, init_q)); exc = x[x > t] - t
        if len(exc) < 10 or exc.var() <= 1e-12: return float(x.max())
        m, v = float(exc.mean()), float(exc.var())
        g = 0.5*(1 - m*m/v); s = 0.5*m*(m*m/v + 1); r = target_q*len(x)/len(exc)
        return t - s*np.log(r) if abs(g) < 1e-6 else t + (s/g)*(r**(-g) - 1)
    thr_evt = evt_thr(benign)
    thr = {"f1opt": thr_f1opt, "benign": thr_benign, "evt": thr_evt}[args.threshold]
    detected = [i for i in range(clean_ps.shape[0])
                if flat_labels[i] == 1 and clean_topk[i] > thr]
    rng = random.Random(args.seed)
    sample = rng.sample(detected, min(args.n_trials, len(detected)))
    print(f"[detect] clean F1={clean_f1:.4f} thr={thr:.3f}; {len(detected)} TPs, attacking {len(sample)}")

    def attack_window(w):
        """End-to-end random search + sign refinement minimizing S_k."""
        X = flat_xs[w:w + 1]
        g = torch.Generator().manual_seed(args.seed + w)
        n_q = 0
        # budget set: bump each sensor +eps, rank by S_k reduction, take top-B
        base = Sk(X); n_q += 1
        red = []
        for j in range(X.shape[1]):
            Xj = X.clone(); Xj[:, j, :] = (Xj[:, j, :] + eps).clamp(0.0, 1.0)
            red.append((base - Sk(Xj), j)); n_q += 1
        red.sort(reverse=True)
        Vbar = torch.tensor([j for _, j in red[:B]], dtype=torch.long)
        best_X, best = X.clone(), base
        # random search
        for _ in range(args.iters):
            delta = torch.zeros_like(X)
            delta[:, Vbar, :] = (torch.rand(X[:, Vbar, :].shape, generator=g) * 2 - 1) * eps
            Xp = (X + delta).clamp(0.0, 1.0)
            s = Sk(Xp); n_q += 1
            if s < best:
                best, best_X = s, Xp
        # sign refinement: try +/-eps on each budget sensor's whole window
        for j in Vbar.tolist():
            for sign in (-1.0, 1.0):
                Xp = best_X.clone()
                Xp[:, j, :] = (X[:, j, :] + sign * eps).clamp(0.0, 1.0)
                s = Sk(Xp); n_q += 1
                if s < best:
                    best, best_X = s, Xp
        return best_X, base, best, n_q

    attacked_ps = clean_ps.copy()
    queries, sk_drop, flipped = [], [], 0
    with torch.no_grad():
        for i, w in enumerate(sample):
            Xp, base, best, n_q = attack_window(w)
            attacked_ps[w] = scores_vec(Xp).cpu().numpy()
            queries.append(n_q); sk_drop.append(base - best)
            if base > thr and best <= thr:
                flipped += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample)}  median Sk drop={np.median(sk_drop):.3f}  flips={flipped}")

    attacked_topk = aggregate_topk(attacked_ps, k=K)
    attacked_f1, _, _, _, _ = compute_f1_metrics(attacked_topk, flat_labels)

    res = {
        "dataset": args.dataset, "seed": args.seed, "k": K, "budget": B,
        "epsilon": eps, "iters": args.iters, "objective": "topk_aggregate_Sk",
        "n_attacked": len(sample), "clean_F1": float(clean_f1),
        "attacked_F1": float(attacked_f1),
        "F1_reduction": float(clean_f1 - attacked_f1),
        "window_ASR": flipped / max(1, len(sample)),  # fraction pushed below thr
        "median_Sk_drop": float(np.median(sk_drop)),
        "median_queries": float(np.median(queries)),
    }
    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    print(f"  clean F1={res['clean_F1']:.4f}  attacked F1={res['attacked_F1']:.4f}  "
          f"F1 reduction={res['F1_reduction']:+.4f}")
    print(f"  window ASR (pushed below thr)={res['window_ASR']*100:.1f}%  "
          f"median Sk drop={res['median_Sk_drop']:.3f}  median queries={res['median_queries']:.0f}")


if __name__ == "__main__":
    main()
