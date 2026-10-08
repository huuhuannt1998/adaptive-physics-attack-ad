"""
Correction baselines (paper Table I, Section VI-A): compare three preprocessing pipelines
under the decision-level (top-k) attack, to separate fixing the invalid
attack projection from globally changing the clean detector input.

Pipelines (detector input):
  original : clean = X (raw),  attacked = clip(X+delta, 0, 1)   [asymmetric bug]
  clip     : clean = clip(X,0,1), attacked = clip(X+delta,0,1)  [symmetric global clip]
  center   : clean = X (raw),  attacked = X+delta (delta eps-bounded) [center-proj]
             (center-preserving projection clip(X+delta, X-eps, X+eps) = X+delta
              since ||delta||_inf <= eps; corrects the budget WITHOUT snapping OOR
              clean values to the boundary)

For each pipeline we report the EPB decomposition medians (T,A,M), clean F1,
decision-level attacked F1, and window-evasion rate (ASR) at the pipeline's
own F1-optimal threshold. Decision-level attack = gradient-free random search
minimizing the top-k aggregate S_k, budget b, eps.

Usage:
  python scripts/correction_baselines.py --dataset wadi --seed 0 --n-trials 40 \
      --out reports/corr_baselines_wadi_seed0.json
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
    p.add_argument("--iters", type=int, default=250)
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

    def proj(X, pipeline, attacked):
        """detector input under each pipeline."""
        if pipeline == "clip":
            return X.clamp(0.0, 1.0)
        if pipeline == "center":
            return X                      # raw (delta already eps-bounded upstream)
        # original: clean raw, attacked clipped
        return X.clamp(0.0, 1.0) if attacked else X

    def score(X, pipeline, attacked):
        Xi = proj(X, pipeline, attacked)
        return standardize_score(model(Xi, edge), Xi[..., -1], median, iqr_v)

    def clean_scores(pipeline):
        out, BATCH = [], 256
        with torch.no_grad():
            for s0 in range(0, flat_xs.shape[0], BATCH):
                out.append(score(flat_xs[s0:s0+BATCH], pipeline, attacked=False).cpu().numpy())
        return aggregate_topk(np.concatenate(out, 0), k=K), np.concatenate(out, 0)

    results = {}
    for pipeline in ["original", "clip", "center"]:
        c_agg, c_ps = clean_scores(pipeline)
        cf1, _, _, thr, _ = compute_f1_metrics(c_agg, labels)
        detected = [i for i in range(c_ps.shape[0]) if labels[i] == 1 and c_agg[i] > thr]
        rng = random.Random(args.seed)
        sample = rng.sample(detected, min(args.n_trials, len(detected)))

        def Sk(X, attacked=True):
            return float(score(X, pipeline, attacked)[0].topk(K).values.sum().item())

        att_ps = c_ps.copy()
        flips, Ts, As, Ms = 0, [], [], []
        with torch.no_grad():
            for w in sample:
                X = flat_xs[w:w+1]; g = torch.Generator().manual_seed(args.seed + w)
                # budget set by Sk-reduction saliency
                base = Sk(X); red = []
                for j in range(X.shape[1]):
                    Xj = X.clone(); Xj[:, j, :] = (Xj[:, j, :] + eps)
                    red.append((base - Sk(Xj), j))
                red.sort(reverse=True); Vbar = torch.tensor([j for _, j in red[:B]])
                bestX, best = X.clone(), base
                for _ in range(args.iters):
                    d = torch.zeros_like(X)
                    d[:, Vbar, :] = (torch.rand(X[:, Vbar, :].shape, generator=g)*2-1)*eps
                    Xp = X + d                    # eps-bounded perturbation
                    s = Sk(Xp)
                    if s < best: best, bestX = s, Xp
                att_ps[w] = score(bestX, pipeline, attacked=True)[0].cpu().numpy()
                if base > thr and best <= thr: flips += 1
                # decomposition on this trial for this pipeline
                Yb = proj(X, pipeline, attacked=False); Yc = proj(X, pipeline, attacked=True)
                Ya = proj(bestX, pipeline, attacked=True)
                Ts.append(float((Ya-Yb).abs().max())); As.append(float((Ya-Yc).abs().max()))
                Ms.append(float((Yc-Yb).abs().max()))
        a_agg = aggregate_topk(att_ps, k=K)
        af1, _, _, _, _ = compute_f1_metrics(a_agg, labels)
        results[pipeline] = {
            "clean_F1": float(cf1), "attacked_F1": float(af1),
            "F1_reduction": float(cf1 - af1),
            "window_ASR": flips / max(1, len(sample)),
            "T_median": float(np.median(Ts)), "A_median": float(np.median(As)),
            "M_median": float(np.median(Ms)), "n_attacked": len(sample), "thr": float(thr),
        }

    res = {"dataset": args.dataset, "seed": args.seed, "k": K, "budget": B,
           "epsilon": eps, "pipelines": results}
    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    for pl, d in results.items():
        print(f"  {pl:9s}  cleanF1={d['clean_F1']:.3f} attF1={d['attacked_F1']:.3f} "
              f"F1red={d['F1_reduction']:+.3f} ASR={d['window_ASR']*100:4.1f}%  "
              f"T={d['T_median']:.2f} A={d['A_median']:.3f} M={d['M_median']:.2f}")


if __name__ == "__main__":
    main()
