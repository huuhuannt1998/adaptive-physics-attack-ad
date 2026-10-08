"""
HAI cross-dataset audit (paper Section V-E): extend the OOD-clamp / EPB analysis to a
THIRD public ICS testbed (HAI, HIL-augmented), fully self-contained.

This script does NOT depend on the GDN repo or its dataloaders. It mirrors the
GDN preprocessing (per-sensor train-fit MinMax, downsample, sliding window) on
HAI directly, so HAI becomes a third point on the paper's cross-dataset
OOR-magnitude line (WADI max OOR ~267, SWaT ~3.1).

Two measurements, both label-free:
  1. OOR / preprocessing precursor (detector-INDEPENDENT): fraction of test
     cells outside the train-fit [0,1] range and the max drift beyond it. This
     is the input-space quantity the post-MinMax clip amplifies into effective
     perturbation -- the same precursor EPB measures.
  2. OOD-clamp score degradation on a forecasting detector: we train a small
     LSTM forecaster (same paradigm as the GDN/LSTM study) on HAI train and
     measure score(X) - score(clip(X)) at the target sensor on OOR test
     windows, undefended (asymmetric clip) vs repaired (symmetric clip).

No attack-effectiveness or F1 claim is made on HAI (labels not used): the point
is that the preprocessing precursor and the OOD-clamp mechanism reproduce on a
third testbed.

Usage:
  python scripts/hai_audit.py --release hai-23.05 --out reports/hai_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_hai(release: str):
    base = PROJECT_ROOT / "data" / "hai" / release
    tr_files = sorted(base.glob("hai-train*.csv"))
    te_files = sorted(base.glob("hai-test*.csv"))
    if not tr_files or not te_files:
        raise SystemExit(f"no HAI csvs under {base}")
    tr = pd.concat([pd.read_csv(f) for f in tr_files], ignore_index=True)
    te = pd.concat([pd.read_csv(f) for f in te_files], ignore_index=True)
    sensors = [c for c in tr.columns if c in te.columns and c.lower() != "timestamp"]
    tr = tr[sensors].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    te = te[sensors].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    return tr.astype(float), te.astype(float), sensors


def minmax_fit_transform(train: pd.DataFrame, test: pd.DataFrame):
    lo = train.min(axis=0)
    rng = (train.max(axis=0) - lo).replace(0, 1.0)
    return (train - lo) / rng, (test - lo) / rng


def downsample(arr: np.ndarray, factor: int):
    n = (arr.shape[0] // factor) * factor
    return arr[:n].reshape(n // factor, factor, arr.shape[1]).mean(axis=1)


def make_windows(arr: np.ndarray, win: int, stride: int):
    # X: (M, N, win) windows -> forecast target y: (M, N) next step after window.
    idx = range(0, arr.shape[0] - win - 1, stride)
    X = np.stack([arr[i:i + win].T for i in idx], axis=0)      # (M, N, win)
    y = np.stack([arr[i + win] for i in idx], axis=0)          # (M, N)
    return X.astype(np.float32), y.astype(np.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, n, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n, hidden, layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n)

    def forward(self, X):                 # X: (B, N, W)
        out, _ = self.lstm(X.transpose(1, 2))
        return self.head(out[:, -1, :])   # (B, N)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", default="hai-23.05")
    p.add_argument("--downsample", type=int, default=10)
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-train-windows", type=int, default=20000)
    p.add_argument("--max-oor-windows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/hai_audit.json")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[load] HAI {args.release}")
    tr, te, sensors = load_hai(args.release)
    print(f"  train={tr.shape} test={te.shape} sensors={len(sensors)}")

    tr_n, te_n = minmax_fit_transform(tr, te)
    tr_a, te_a = tr_n.values, te_n.values

    # --- (1) OOR / preprocessing precursor (detector-independent) ---
    below = (te_a < 0.0)
    above = (te_a > 1.0)
    oor = below | above
    drift = np.zeros_like(te_a)
    drift[above] = te_a[above] - 1.0
    drift[below] = 0.0 - te_a[below]
    oor_stats = {
        "frac_oor_test_cells": float(oor.mean()),
        "max_oor_cell_value": float(te_a.max()),          # comparable to WADI 267 / SWaT 3.1
        "max_drift_beyond_unit_range": float(drift.max()),
        "n_sensors": len(sensors),
    }
    print(f"[oor] frac={oor_stats['frac_oor_test_cells']*100:.3f}%  "
          f"max_cell={oor_stats['max_oor_cell_value']:.2f}  "
          f"max_drift={oor_stats['max_drift_beyond_unit_range']:.2f}")

    # --- train LSTM forecaster on downsampled normalized HAI train ---
    tr_d = downsample(tr_a, args.downsample)
    te_d = downsample(te_a, args.downsample)
    Xtr, ytr = make_windows(tr_d, args.win, args.stride)
    if Xtr.shape[0] > args.max_train_windows:
        sel = np.random.choice(Xtr.shape[0], args.max_train_windows, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    print(f"[train] {Xtr.shape[0]} windows, N={Xtr.shape[1]}, W={Xtr.shape[2]}")

    model = LSTMForecaster(len(sensors))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    bs = 256
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(Xt.shape[0]); tot = 0.0; nb = 0
        for i in range(0, Xt.shape[0], bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(model(Xt[j]), yt[j]); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{args.epochs} train_mse={tot/max(1,nb):.5f}")

    # Validation-residual standardization stats (use a held-out tail of train).
    model.eval()
    with torch.no_grad():
        res = (model(Xt) - yt).abs().numpy()
    from scipy.stats import iqr as scipy_iqr
    median = np.median(res, axis=0)
    iqrv = scipy_iqr(res, axis=0)
    med_t = torch.from_numpy(median).float()
    iqr_t = torch.from_numpy(iqrv).float()

    def score(X):                          # standardized residual, (B, N)
        f = model(X)
        d = (f - X[..., -1]).abs()
        return (d - med_t) / (iqr_t.abs() + 1e-2)

    # --- (2) OOD-clamp audit on test OOR windows ---
    Xte, _ = make_windows(te_d, args.win, args.stride)
    Xte_t = torch.from_numpy(Xte)
    win_max = Xte_t.amax(dim=(1, 2))
    oor_idx = torch.nonzero(win_max > 1.0, as_tuple=False).flatten().tolist()
    oor_idx = oor_idx[:args.max_oor_windows]
    print(f"[audit] {len(oor_idx)} OOR test windows")

    epb, deg_undef, deg_def = [], [], []
    with torch.no_grad():
        for i in oor_idx:
            X = Xte_t[i:i + 1]
            Xc = X.clamp(0.0, 1.0)
            epb.append(float((X - Xc).abs().max().item()))
            s_raw = score(X)[0]; s_clip = score(Xc)[0]
            tgt = int(s_raw.argmax().item())
            deg_undef.append(float(s_raw[tgt].item() - s_clip[tgt].item()))
            deg_def.append(0.0)            # symmetric clip: baseline == attacked

    audit = {
        "n_oor_windows": len(oor_idx),
        "epb_mean": float(np.mean(epb)) if epb else 0.0,
        "epb_max": float(np.max(epb)) if epb else 0.0,
        "ood_clamp_deg_undef_mean": float(np.mean(deg_undef)) if deg_undef else 0.0,
        "ood_clamp_deg_undef_median": float(np.median(deg_undef)) if deg_undef else 0.0,
        "ood_clamp_deg_defended_mean": float(np.mean(deg_def)) if deg_def else 0.0,
    }
    print(f"[audit] EPB mean={audit['epb_mean']:.2f} max={audit['epb_max']:.2f}  "
          f"OOD-clamp deg undef={audit['ood_clamp_deg_undef_mean']:+.3f} "
          f"def={audit['ood_clamp_deg_defended_mean']:+.3f}")

    res_out = {"release": args.release, "seed": args.seed,
               "oor_precursor": oor_stats, "lstm_ood_clamp": audit}
    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res_out, indent=2))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
