"""
Preprocessing-variant analysis (paper Section V-E).

Demonstrates that the OOD-clamp shortcut is a property of *train-fit
normalization + clip*, not specific to MinMax. For each of three
train-fit normalizations (MinMax, z-score/standard, robust median-IQR),
we fit per-sensor statistics on the TRAIN split, transform the TEST
split, and measure how far test data drifts beyond the train-fit
support --- the quantity the post-clip step amplifies into effective
perturbation.

No detector and no attack are involved: this isolates the preprocessing
pipeline, which is where the OOD-clamp mechanism lives. The deployed
"clip range" for each scheme is the range the TRAINING data occupies
after that scheme's transform (the range a train-fit clip enforces).

Reports, per (dataset, scheme):
  - frac_oor: fraction of test cells outside the train-fit range
    (scale-invariant; directly comparable across schemes)
  - max_drift_rel: max test drift beyond the train range, as a multiple
    of the train range width (scale-invariant amplification factor)
  - max_drift_native: max drift in the scheme's native units

Usage:
  python scripts/preprocessing_variant_analysis.py --out reports/preprocessing_variants.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fit_transform(scheme, train, test):
    """Per-sensor fit on train, transform test. Returns (train_t, test_t)."""
    if scheme == "minmax":
        lo = train.min(axis=0)
        rng = (train.max(axis=0) - lo).replace(0, 1.0)
        return (train - lo) / rng, (test - lo) / rng
    if scheme == "zscore":
        mu = train.mean(axis=0)
        sd = train.std(axis=0).replace(0, 1.0)
        return (train - mu) / sd, (test - mu) / sd
    if scheme == "robust":
        med = train.median(axis=0)
        q1 = train.quantile(0.25, axis=0)
        q3 = train.quantile(0.75, axis=0)
        iqr = (q3 - q1).replace(0, 1.0)
        return (train - med) / iqr, (test - med) / iqr
    raise ValueError(scheme)


def analyze(dataset, train, test):
    out = {}
    for scheme in ["minmax", "zscore", "robust"]:
        train_t, test_t = fit_transform(scheme, train, test)
        # Deployed clip range = per-sensor range the TRAIN occupies post-transform.
        lo = train_t.min(axis=0)
        hi = train_t.max(axis=0)
        width = (hi - lo).replace(0, 1.0)

        below = (test_t < lo.values).values
        above = (test_t > hi.values).values
        oor = below | above
        frac_oor = float(oor.mean())

        # Drift beyond range, per cell, in native units.
        drift = np.zeros_like(test_t.values, dtype=float)
        drift[above] = (test_t.values - hi.values)[above]
        drift[below] = (lo.values - test_t.values)[below]
        max_drift_native = float(drift.max())

        # Scale-invariant: drift relative to the train range width (per sensor).
        rel = drift / width.values  # broadcast per-column
        max_drift_rel = float(np.nanmax(rel))

        # The single most-out-of-range sensor (for the worked example).
        per_sensor_max_rel = np.nanmax(rel, axis=0)
        worst_sensor = int(np.argmax(per_sensor_max_rel))

        out[scheme] = {
            "frac_oor_test_cells": frac_oor,
            "max_drift_relative_to_train_range": max_drift_rel,
            "max_drift_native_units": max_drift_native,
            "worst_sensor_index": worst_sensor,
            "worst_sensor_max_rel_drift": float(per_sensor_max_rel[worst_sensor]),
        }
    return out


def load_csv(dataset):
    path = PROJECT_ROOT / f"repos/GDN/data/{dataset}"
    train = pd.read_csv(path / "train.csv", index_col=0)
    test = pd.read_csv(path / "test.csv", index_col=0)
    for c in ["attack", "label"]:
        if c in train.columns:
            train = train.drop(columns=[c])
        if c in test.columns:
            test = test.drop(columns=[c])
    # Align columns (test may carry the attack label only).
    common = [c for c in train.columns if c in test.columns]
    return train[common].astype(float), test[common].astype(float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["wadi", "swat"])
    p.add_argument("--out", default="reports/preprocessing_variants.json")
    args = p.parse_args()

    results = {}
    for ds in args.datasets:
        print(f"[load] {ds}")
        train, test = load_csv(ds)
        print(f"  train={train.shape}  test={test.shape}")
        results[ds] = analyze(ds, train, test)
        print(f"=== {ds} ===")
        for scheme, r in results[ds].items():
            print(f"  {scheme:8s}  frac_OOR={r['frac_oor_test_cells']*100:6.3f}%  "
                  f"max_drift_rel={r['max_drift_relative_to_train_range']:10.1f}x  "
                  f"native={r['max_drift_native_units']:10.2f}  "
                  f"worst_sensor={r['worst_sensor_index']}")

    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
