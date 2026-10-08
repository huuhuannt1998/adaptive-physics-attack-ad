"""
Clipping side-effects (paper Section VI-D): does symmetric clipping correct the
evaluation path, or does it partially defend by suppressing informative
out-of-range anomaly evidence?

Partition test windows into four slices: {benign, anomalous} x {in-range,
out-of-range} (out-of-range = the window contains a post-MinMax cell > 1).
For each slice, score with the original (unclipped) pipeline and the clipped
pipeline, and report detection recall (anomalous slices) / false-positive
rate (benign slices) at a benign-calibrated threshold. If clipping materially
reduces recall on anomalous out-of-range windows, it suppresses anomaly
evidence and should not be called a broadly deployable defense.

Usage:
  python scripts/clipping_side_effects.py --dataset wadi --seed 0 \
      --out reports/clip_sideeffects_wadi_seed0.json
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
from scripts.multi_metric_eval import aggregate_topk  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--fpr-target", type=float, default=0.05)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    m = setup_victim("gdn", seed=args.seed, dataset=args.dataset)
    model = m.model
    from scipy.stats import iqr as scipy_iqr
    val_deltas = collect_residuals_per_sensor(model, m.val_dataloader, "gdn")
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    test_batches = list(m.test_dataloader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    labels = torch.cat([b[2].float() for b in test_batches], dim=0).numpy().astype(int)
    edge_index = test_batches[0][3].float()
    win_max = flat_xs.amax(dim=(1, 2)).numpy()          # per-window max post-MinMax value
    is_oor = win_max > 1.0                               # window contains an OOR cell

    def topk_scores(clip: bool):
        out, BATCH = [], 256
        with torch.no_grad():
            for s0 in range(0, flat_xs.shape[0], BATCH):
                X = flat_xs[s0:s0 + BATCH]
                if clip:
                    X = X.clamp(0.0, 1.0)
                s = standardize_score(model(X, edge_index), X[..., -1], median, iqr_v)
                out.append(s.cpu().numpy())
        return aggregate_topk(np.concatenate(out, axis=0), k=args.k)

    clean_agg = topk_scores(clip=False)
    clip_agg = topk_scores(clip=True)

    # benign-calibrated threshold: choose per-pipeline threshold hitting fpr-target
    # on benign windows (deployment-realistic, no anomaly labels used).
    def thr_for_fpr(agg):
        benign = agg[labels == 0]
        return float(np.quantile(benign, 1.0 - args.fpr_target))

    thr_clean, thr_clip = thr_for_fpr(clean_agg), thr_for_fpr(clip_agg)

    slices = {
        "benign_inrange":    (labels == 0) & (~is_oor),
        "benign_oor":        (labels == 0) & (is_oor),
        "anom_inrange":      (labels == 1) & (~is_oor),
        "anom_oor":          (labels == 1) & (is_oor),
    }

    def rates(mask, agg, thr, positive):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        fired = (agg[mask] > thr)
        if positive:   # anomalous slice -> recall
            return {"n": n, "recall": float(fired.mean())}
        else:          # benign slice -> FPR
            return {"n": n, "fpr": float(fired.mean())}

    res = {"dataset": args.dataset, "seed": args.seed, "k": args.k,
           "fpr_target": args.fpr_target,
           "thr_clean": thr_clean, "thr_clip": thr_clip,
           "frac_windows_oor": float(is_oor.mean()),
           "slices": {}}
    for name, mask in slices.items():
        positive = name.startswith("anom")
        res["slices"][name] = {
            "clean": rates(mask, clean_agg, thr_clean, positive),
            "clipped": rates(mask, clip_agg, thr_clip, positive),
        }

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    print(f"  OOR windows: {100*res['frac_windows_oor']:.2f}%  "
          f"(thr clean={thr_clean:.2f} clip={thr_clip:.2f}, FPR target={args.fpr_target})")
    for name, d in res["slices"].items():
        c, cl = d["clean"], d["clipped"]
        key = "recall" if name.startswith("anom") else "fpr"
        cv = c.get(key); clv = cl.get(key)
        print(f"  {name:16s} n={c.get('n',0):5d}  clean {key}={cv if cv is None else round(cv,3)}"
              f"  clipped {key}={clv if clv is None else round(clv,3)}")


if __name__ == "__main__":
    main()
