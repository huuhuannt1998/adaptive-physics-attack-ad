"""
Train the LSTM forecaster on WADI/SWaT (detector-agnostic study, paper Section V-E).

Reuses the GDN data pipeline (same MinMax+downsample+window preprocessing,
same train/val/test splits) so the comparison to GDN is apples-to-apples.
Trains an LSTM to forecast the next-step sensor vector, saves the checkpoint,
and reports clean F1 to confirm it is a working detector.

Usage:
  python scripts/train_lstm_forecaster.py --seed 0 --dataset wadi --epochs 15 \\
      --out detectors/ckpts/lstm_wadi_seed0.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.beta_sanity_check import setup_victim  # noqa: E402
from detectors.lstm_forecaster import LSTMForecaster  # noqa: E402


def f1_sweep(scores_1d, labels, steps=200):
    from sklearn.metrics import f1_score
    from scipy.stats import rankdata
    ranked = rankdata(scores_1d, method="ordinal")
    n = len(scores_1d)
    best = 0.0
    for i in range(steps):
        pred = (ranked > (i / steps) * n).astype(int)
        best = max(best, f1_score(labels, pred, zero_division=0))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Borrow GDN's fully-wired data pipeline (we use only its dataloaders).
    print(f"[setup] borrowing GDN data pipeline for {args.dataset} seed {args.seed}")
    m = setup_victim("gdn", seed=args.seed, dataset=args.dataset)
    train_loader, val_loader, test_loader = (
        m.train_dataloader, m.val_dataloader, m.test_dataloader)
    n_sensors = next(iter(train_loader))[0].shape[1]
    print(f"[data] n_sensors={n_sensors}")

    model = LSTMForecaster(n_sensors, hidden=args.hidden, layers=args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossfn = nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y, _, _ in train_loader:
            opt.zero_grad()
            pred = model(x.float())
            loss = lossfn(pred, y.float())
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        # Quick val loss
        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                vtot += lossfn(model(x.float()), y.float()).item(); vnb += 1
        print(f"  epoch {ep+1}/{args.epochs}  train_mse={tot/max(1,nb):.5f}  val_mse={vtot/max(1,vnb):.5f}")

    # Validation residual stats for standardization.
    from scipy.stats import iqr as scipy_iqr
    model.eval()
    val_res = []
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            val_res.append((model(x.float()) - y.float()).abs().cpu().numpy())
    val_res = np.concatenate(val_res, axis=0)
    median = np.median(val_res, axis=0)
    iqrv = scipy_iqr(val_res, axis=0)

    # Clean F1 on the test stream (top-10 aggregation, F1-optimal sweep).
    test_scores, test_labels = [], []
    with torch.no_grad():
        for x, y, lab, _ in test_loader:
            r = (model(x.float()) - y.float()).abs().cpu().numpy()
            s = (r - median) / (np.abs(iqrv) + 1e-2)
            test_scores.append(s); test_labels.append(lab.cpu().numpy())
    test_scores = np.concatenate(test_scores, axis=0)
    test_labels = np.concatenate(test_labels, axis=0).astype(int)
    k = min(10, test_scores.shape[1])
    topk = np.sort(test_scores, axis=1)[:, -k:].sum(axis=1)
    clean_f1 = f1_sweep(topk, test_labels)
    print(f"[clean] F1(top-{k}) = {clean_f1:.4f}  (anomaly rate {100*test_labels.mean():.2f}%)")

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_sensors": n_sensors, "hidden": args.hidden, "layers": args.layers,
        "median": median, "iqr": iqrv,
        "seed": args.seed, "dataset": args.dataset, "clean_f1": float(clean_f1),
    }, out)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
