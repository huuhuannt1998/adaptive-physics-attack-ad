"""
EPB decomposition (paper Section V-C): separate the realized detector-input
displacement into attack-induced (A) and preprocessing-path-mismatch (M)
components, to test whether the large total (T) is adversarial or a
clean/attacked path mismatch.

Original asymmetric pipeline: Pi_base = identity, Pi_atk = clip(.,0,1).
Per attacked trial with clean window X and attacked X_pert = clip(X+delta):
  T = || X_pert         - X        ||_inf   (total realized displacement = EPB)
  A = || X_pert         - clip(X)  ||_inf   (attack-induced)
  M = || clip(X)        - X        ||_inf   (preprocessing-path mismatch)
Triangle bound: T <= A + M (norms are not additive).
At the dominant coordinate j* = argmax |X_pert - X|, report a* and m* so a
large T is attributed to the attack or to the path mismatch.

Windows sampled exactly as the audit/multi_metric_eval: F1-opt true-positive
windows, argmax target sensor. BETA attack, epsilon=0.1, budget b.

Usage:
  python scripts/epb_decomposition.py --dataset wadi --seed 1 --n-trials 100 \
      --out reports/epb_decomp_wadi_seed1.json
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


def linf(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--detector", default="gdn", choices=["gdn", "topogdn"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    eps = args.epsilon

    m = setup_victim(args.detector, seed=args.seed, dataset=args.dataset)
    model = m.model
    from scipy.stats import iqr as scipy_iqr
    val_deltas = collect_residuals_per_sensor(model, m.val_dataloader, args.detector)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    test_batches = list(m.test_dataloader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    flat_labels = torch.cat([b[2].float() for b in test_batches], dim=0).numpy().astype(int)
    edge_index_template = test_batches[0][3].float()

    def victim_forward(X: torch.Tensor) -> torch.Tensor:
        forecast = model(X, edge_index_template)
        return standardize_score(forecast, X[..., -1], median, iqr_v)

    # clean per-sensor scores -> F1-opt threshold + detected true-positive windows
    BATCH = 256
    clean_ps = []
    with torch.no_grad():
        for s0 in range(0, flat_xs.shape[0], BATCH):
            X = flat_xs[s0:s0 + BATCH]
            clean_ps.append(standardize_score(model(X, edge_index_template), X[..., -1],
                                              median, iqr_v).cpu().numpy())
    clean_ps = np.concatenate(clean_ps, axis=0)
    clean_top10 = aggregate_topk(clean_ps, k=10)
    _, _, _, clean_raw_thresh, _ = compute_f1_metrics(clean_top10, flat_labels)

    detected = [(i, int(np.argmax(clean_ps[i]))) for i in range(clean_ps.shape[0])
                if flat_labels[i] == 1 and clean_top10[i] > clean_raw_thresh]
    rng = random.Random(args.seed)
    sample = rng.sample(detected, min(args.n_trials, len(detected)))
    print(f"[detect] {len(detected)} F1-opt TPs; attacking {len(sample)}")

    from attacks.beta import BETAAttack, BETAConfig
    cfg = BETAConfig(epsilon=eps, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    attack = BETAAttack(victim_forward, lambda: model.learned_graph.detach(),
                        num_nodes=flat_xs.shape[1], config=cfg)

    T, A, M, aStar, mStar, score_deg = [], [], [], [], [], []
    for k, (w, tgt) in enumerate(sample):
        X = flat_xs[w:w + 1]
        try:
            X_pert, _ = attack.attack(X.clone(), tgt, args.budget)
        except Exception as e:
            print(f"  trial {k}: FAILED {type(e).__name__}"); continue
        Xc = X.clamp(0.0, 1.0)
        d_total = (X_pert - X)          # T vector
        d_attack = (X_pert - Xc)        # A vector
        d_path = (Xc - X)               # M vector
        T.append(linf(d_total)); A.append(linf(d_attack)); M.append(linf(d_path))
        j = int(d_total.abs().flatten().argmax().item())
        aStar.append(float(d_attack.flatten()[j].abs().item()))
        mStar.append(float(d_path.flatten()[j].abs().item()))
        with torch.no_grad():
            sc = victim_forward(X_pert)[0, tgt].item()
            sd = float(clean_ps[w, tgt] - sc)
        score_deg.append(sd)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(sample)}  T~{np.median(T):.2f} A~{np.median(A):.3f} M~{np.median(M):.2f}")

    def stats(x):
        x = np.asarray(x)
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                "max": float(np.max(x)), "mean": float(np.mean(x)),
                "frac_gt_eps": float((x > eps).mean())}

    from scipy.stats import spearmanr
    T_, A_, M_, sd_ = map(np.asarray, (T, A, M, score_deg))
    # dominant-coordinate attribution: at j*, is the displacement attack or path?
    aStar_, mStar_ = np.asarray(aStar), np.asarray(mStar)
    dom_is_path = float((mStar_ > aStar_).mean())
    path_share = float(np.median(mStar_ / (aStar_ + mStar_ + 1e-12)))

    res = {
        "dataset": args.dataset, "detector": args.detector, "seed": args.seed,
        "epsilon": eps, "budget": args.budget, "n_trials": len(T),
        "T_total": stats(T_), "A_attack": stats(A_), "M_path": stats(M_),
        "dominant_coord_is_path_fraction": dom_is_path,
        "dominant_coord_path_share_median": path_share,
        "spearman_T_vs_scoredeg": float(spearmanr(T_, sd_)[0]),
        "spearman_A_vs_scoredeg": float(spearmanr(A_, sd_)[0]),
        "spearman_M_vs_scoredeg": float(spearmanr(M_, sd_)[0]),
    }
    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    print(f"  T median={res['T_total']['median']:.2f}  A median={res['A_attack']['median']:.3f}  "
          f"M median={res['M_path']['median']:.2f}")
    print(f"  dominant-coord is PATH in {dom_is_path*100:.1f}% of trials; "
          f"A>eps in {res['A_attack']['frac_gt_eps']*100:.1f}%")
    print(f"  Spearman(score-deg): T={res['spearman_T_vs_scoredeg']:.2f} "
          f"A={res['spearman_A_vs_scoredeg']:.2f} M={res['spearman_M_vs_scoredeg']:.2f}")


if __name__ == "__main__":
    main()
