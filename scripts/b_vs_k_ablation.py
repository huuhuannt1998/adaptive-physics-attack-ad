"""
b-vs-k ablation (paper Section VII): separate the defense effect from the
attacker-budget / aggregation-size interaction.

The defended detector shows ~0 F1 reduction under target-sensor attacks.
Is that because the correction bounds the attacker, or because the attacker
modifies b sensors while detection aggregates the top-k (with b<k, driving
one sensor down leaves k others above threshold)? We sweep b and k on the
DEFENDED GDN detector and report F1 reduction as a b x k grid. If F1
reduction stays low even when b>=k, the correction has stronger evidence;
if it rises sharply at b>=k, the earlier zero-F1 result was partly structural.

Usage:
  python scripts/b_vs_k_ablation.py --dataset wadi --seeds 0 1 2 \
      --n-trials 40 --out reports/b_vs_k_wadi.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.multi_metric_eval import evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--detector", default="gdn")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bs", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--attacker", default="beta")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # grid[b][k] -> list of per-seed F1 reductions (defended detector)
    grid = {b: {k: [] for k in args.ks} for b in args.bs}
    tgt = {b: {k: [] for k in args.ks} for b in args.bs}
    for seed in args.seeds:
        for b in args.bs:
            for k in args.ks:
                try:
                    r = evaluate(args.detector, seed, args.n_trials, budget=b,
                                 agg_k=k, defended=True, dataset=args.dataset,
                                 attacker=args.attacker)
                    f1red = r["reduction"]["raw_F1"]
                    td = r["target_score_degradation"]["mean"]
                    grid[b][k].append(float(f1red))
                    tgt[b][k].append(float(td) if td is not None else 0.0)
                    print(f"[{args.dataset} s{seed}] b={b} k={k}  F1red={f1red:+.4f}  tgt_deg={td}")
                except Exception as e:
                    print(f"[{args.dataset} s{seed}] b={b} k={k}  FAILED {type(e).__name__}: {e}")

    def agg(d):
        return {b: {k: (float(np.mean(v)) if v else None) for k, v in row.items()}
                for b, row in d.items()}

    res = {"dataset": args.dataset, "detector": args.detector, "seeds": args.seeds,
           "bs": args.bs, "ks": args.ks, "n_trials": args.n_trials,
           "attacker": args.attacker,
           "f1_reduction_mean": agg(grid), "target_deg_mean": agg(tgt)}
    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    # print grid
    print("\nF1 reduction (mean over seeds), rows=b, cols=k:")
    print("      " + "".join(f"k={k:<8}" for k in args.ks))
    for b in args.bs:
        row = res["f1_reduction_mean"][b]
        print(f"  b={b:<3}" + "".join(
            f"{(row[k] if row[k] is not None else float('nan')):+.4f}   " for k in args.ks))


if __name__ == "__main__":
    main()
