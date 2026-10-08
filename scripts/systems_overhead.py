"""
Systems overhead (paper Section VI-D): can EPB auditing and the corrected
preprocessing path run in online ICS monitoring without material overhead?

Microbenchmarks (per window, batch=1, single detector instance) the online
pipeline components and reports median / p95 / p99 latency:
  1. baseline preprocessing (identity pass-through of a window)
  2. corrected preprocessing (symmetric clip)
  3. EPB computation from a retained (X, delta) trace (the audit)
  4. detector inference (GDN forward + standardized score)
  5. threshold aggregation (top-k)
  6. end-to-end (corrected preprocess + inference + aggregation)

The audit (3) is offline/retroactive; (1,2,4,5,6) are the online path. We
report the correction's overhead vs. baseline and check it against the
dataset sampling cadence.

Usage:
  python scripts/systems_overhead.py --dataset wadi --seed 0 --reps 1000 \
      --out reports/overhead_wadi_seed0.json
"""
from __future__ import annotations

import argparse, json, sys, time, platform
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.beta_sanity_check import (  # noqa: E402
    setup_victim, collect_residuals_per_sensor, standardize_score,
)
from scripts.multi_metric_eval import aggregate_topk  # noqa: E402


def timeit(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)  # ms
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(np.mean(a))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)  # single-thread, online-per-window setting

    m = setup_victim("gdn", seed=args.seed, dataset=args.dataset)
    model = m.model.eval()
    from scipy.stats import iqr as scipy_iqr
    val_deltas = collect_residuals_per_sensor(model, m.val_dataloader, "gdn")
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    tb = list(m.test_dataloader)
    flat_xs = torch.cat([b[0].float() for b in tb], dim=0)
    edge_index = tb[0][3].float()
    N, W = flat_xs.shape[1], flat_xs.shape[2]
    X = flat_xs[0:1]
    delta = (torch.rand_like(X) * 2 - 1) * 0.1

    def f_baseline_pre():
        return X.clone()

    def f_corrected_pre():
        return X.clamp(0.0, 1.0)

    def f_epb():  # audit: ||clip(X+delta) - X||_inf from a retained trace
        return float(((X + delta).clamp(0, 1) - X).abs().max().item())

    def f_infer():
        with torch.no_grad():
            fc = model(X.clamp(0, 1), edge_index)
            return standardize_score(fc, X.clamp(0, 1)[..., -1], median, iqr_v)

    def f_aggregate():
        with torch.no_grad():
            s = standardize_score(model(X.clamp(0, 1), edge_index),
                                  X.clamp(0, 1)[..., -1], median, iqr_v)[0].cpu().numpy()
        return aggregate_topk(s[None, :], k=args.k)

    def f_end2end():
        with torch.no_grad():
            Xc = X.clamp(0, 1)
            s = standardize_score(model(Xc, edge_index), Xc[..., -1], median, iqr_v)[0].cpu().numpy()
            return float(aggregate_topk(s[None, :], k=args.k)[0])

    comps = {
        "baseline_preprocess": f_baseline_pre,
        "corrected_preprocess": f_corrected_pre,
        "epb_audit_per_trace": f_epb,
        "detector_inference": f_infer,
        "end_to_end_corrected": f_end2end,
    }
    res = {"dataset": args.dataset, "seed": args.seed, "N_sensors": N, "W_window": W,
           "reps": args.reps, "threads": 1,
           "env": {"platform": platform.platform(), "python": platform.python_version(),
                   "torch": torch.__version__},
           "components_ms": {}}
    for name, fn in comps.items():
        res["components_ms"][name] = timeit(fn, args.reps, args.warmup)

    base = res["components_ms"]["baseline_preprocess"]["median_ms"]
    corr = res["components_ms"]["corrected_preprocess"]["median_ms"]
    e2e = res["components_ms"]["end_to_end_corrected"]["median_ms"]
    res["correction_overhead_ms_median"] = corr - base
    res["end_to_end_median_ms"] = e2e

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}  (N={N}, W={W})")
    for name, d in res["components_ms"].items():
        print(f"  {name:24s} median={d['median_ms']:.4f}ms  p95={d['p95_ms']:.4f}  p99={d['p99_ms']:.4f}")
    print(f"  correction overhead (median): {res['correction_overhead_ms_median']*1000:.2f} us; "
          f"end-to-end {e2e:.3f} ms/window")


if __name__ == "__main__":
    main()
