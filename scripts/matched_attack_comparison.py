"""
Matched attacker comparison (paper Table III, Section VII): compare BETA (target-sensor) and
the decision-level top-k attacker on IDENTICAL windows, threshold, seed,
preprocessing, and stream construction, so "stronger" is claimed only on
metrics where it is actually stronger.

Defended GDN detector. Same F1-optimal threshold, same detected-TP window
sample. For each attacker report: window ASR, F1 reduction, median
target-score reduction, median top-k score reduction.

Usage:
  python scripts/matched_attack_comparison.py --dataset wadi --seed 0 \
      --n-trials 40 --out reports/matched_wadi_seed0.json
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
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    eps, K, B = args.epsilon, args.k, args.budget

    m = setup_victim("gdn", seed=args.seed, dataset=args.dataset)
    model = m.model
    from scipy.stats import iqr as scipy_iqr
    vd = collect_residuals_per_sensor(model, m.val_dataloader, "gdn")
    median = torch.from_numpy(np.median(vd, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(vd, axis=0)).float()
    tb = list(m.test_dataloader)
    flat_xs = torch.cat([b[0].float() for b in tb], dim=0)
    labels = torch.cat([b[2].float() for b in tb], dim=0).numpy().astype(int)
    edge = tb[0][3].float()

    def svec(X):  # per-sensor score on defended (clipped) input
        Xc = X.clamp(0.0, 1.0)
        return standardize_score(model(Xc, edge), Xc[..., -1], median, iqr_v)

    def Sk(X):
        return float(svec(X)[0].topk(K).values.sum().item())

    BATCH, cps = 256, []
    with torch.no_grad():
        for s0 in range(0, flat_xs.shape[0], BATCH):
            cps.append(svec(flat_xs[s0:s0+BATCH]).cpu().numpy())
    cps = np.concatenate(cps, 0)
    ctopk = aggregate_topk(cps, k=K)
    cf1, _, _, thr, _ = compute_f1_metrics(ctopk, labels)
    detected = [i for i in range(cps.shape[0]) if labels[i] == 1 and ctopk[i] > thr]
    sample = random.Random(args.seed).sample(detected, min(args.n_trials, len(detected)))
    print(f"[detect] clean F1={cf1:.4f} thr={thr:.3f}; matched sample={len(sample)}")

    from attacks.beta import BETAAttack, BETAConfig
    beta = BETAAttack(lambda X: svec(X), lambda: model.learned_graph.detach(),
                      num_nodes=flat_xs.shape[1],
                      config=BETAConfig(epsilon=eps, pgd_alpha=0.01, pgd_iters=10,
                                        pgd_restarts=5, candidate_k=32))

    def decision_attack(X, w):
        g = torch.Generator().manual_seed(args.seed + w)
        base = Sk(X); red = []
        for j in range(X.shape[1]):
            Xj = X.clone(); Xj[:, j, :] = (Xj[:, j, :] + eps).clamp(0, 1)
            red.append((base - Sk(Xj), j))
        red.sort(reverse=True); Vbar = torch.tensor([j for _, j in red[:B]])
        bestX, best = X.clone(), base
        for _ in range(args.iters):
            d = torch.zeros_like(X)
            d[:, Vbar, :] = (torch.rand(X[:, Vbar, :].shape, generator=g)*2-1)*eps
            Xp = (X + d).clamp(0, 1); s = Sk(Xp)
            if s < best: best, bestX = s, Xp
        return bestX

    def run(attacker):
        att_ps = cps.copy(); flips = 0; sk_red, tgt_red = [], []
        for w in sample:
            X = flat_xs[w:w+1]
            with torch.no_grad():
                base_sk = Sk(X); tgt = int(svec(X)[0].argmax().item())
            if attacker == "beta":          # BETA needs gradients for candidate selection
                Xp, _ = beta.attack(X.clone(), tgt, B)
            else:
                with torch.no_grad():
                    Xp = decision_attack(X, w)
            with torch.no_grad():
                sc = svec(Xp)[0].cpu().numpy()
            att_ps[w] = sc
            sk_red.append(base_sk - float(np.sort(sc)[-K:].sum()))
            tgt_red.append(float(cps[w, tgt] - sc[tgt]))
            if ctopk[w] > thr and float(np.sort(sc)[-K:].sum()) <= thr:
                flips += 1
        af1, _, _, _, _ = compute_f1_metrics(aggregate_topk(att_ps, k=K), labels)
        return {"window_ASR": flips/max(1,len(sample)), "F1_reduction": float(cf1-af1),
                "median_Sk_reduction": float(np.median(sk_red)),
                "median_target_reduction": float(np.median(tgt_red))}

    res = {"dataset": args.dataset, "seed": args.seed, "n_matched": len(sample),
           "threshold": "f1opt", "clean_F1": float(cf1),
           "beta_target_sensor": run("beta"),
           "decision_level_topk": run("decision")}
    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    for k in ["beta_target_sensor", "decision_level_topk"]:
        d = res[k]
        print(f"  {k:22s} ASR={d['window_ASR']*100:4.1f}%  F1red={d['F1_reduction']:+.4f}  "
              f"Sk_red={d['median_Sk_reduction']:.2f}  tgt_red={d['median_target_reduction']:.3f}")


if __name__ == "__main__":
    main()
