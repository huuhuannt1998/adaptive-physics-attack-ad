"""
Detector-agnostic OOD-clamp audit (paper Section V-E).

Demonstrates that the OOD-clamp shortcut and its repair are NOT specific
to GDN by measuring them on a non-GNN detector (LSTM forecaster) and on
GDN, side by side, on the same out-of-range (OOR) windows.

Core measurement, per OOR window (a window whose max post-MinMax cell
value exceeds 1):
  - EPB: ||X - clip(X,0,1)||_inf, the input-space displacement the
    asymmetric clip produces. Detector-INDEPENDENT by construction.
  - OOD-clamp score degradation: score(X) - score(clip(X)) at the
    window's argmax (target) sensor. This is the anomaly-score drop an
    attacker obtains "for free" from the baseline-vs-attacked clip
    asymmetry. Detector-DEPENDENT; we compare GDN vs LSTM.

The symmetry repair clips BOTH paths, so baseline == attacked and the
degradation vanishes for any detector --- we confirm this too.

Usage:
  python scripts/lstm_ood_clamp_audit.py --seed 0 --dataset wadi \\
      --lstm detectors/ckpts/lstm_wadi_seed0.pt --out reports/lstm_audit_wadi_seed0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.beta_sanity_check import setup_victim  # noqa: E402
from detectors.lstm_forecaster import LSTMForecaster  # noqa: E402


def standardized_score_fn(forecast_fn, median, iqr_v):
    med = torch.from_numpy(np.asarray(median)).float()
    iqr = torch.from_numpy(np.asarray(iqr_v)).float()

    def score(X):
        f = forecast_fn(X)
        target = X[..., -1]
        d = (f - target).abs()
        return (d - med) / (iqr.abs() + 1e-2)
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--lstm", required=True)
    p.add_argument("--max-windows", type=int, default=2000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # GDN gives the data pipeline AND the GDN victim for comparison.
    m = setup_victim("gdn", seed=args.seed, dataset=args.dataset)
    gdn = m.model
    test_batches = list(m.test_dataloader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    edge_index = test_batches[0][3].float()
    n_test, n_sensors, W = flat_xs.shape

    # GDN validation standardization stats.
    from scipy.stats import iqr as scipy_iqr
    gdn_res = []
    with torch.no_grad():
        for x, y, _, ei in m.val_dataloader:
            gdn_res.append((gdn(x.float(), ei.float()) - y.float()).abs().cpu().numpy())
    gdn_res = np.concatenate(gdn_res, axis=0)
    gdn_score = standardized_score_fn(
        lambda X: gdn(X, edge_index), np.median(gdn_res, axis=0), scipy_iqr(gdn_res, axis=0))

    # LSTM victim.
    ck = torch.load(PROJECT_ROOT / args.lstm if not Path(args.lstm).is_absolute()
                    else args.lstm, weights_only=False, map_location="cpu")
    lstm = LSTMForecaster(ck["n_sensors"], hidden=ck["hidden"], layers=ck["layers"])
    lstm.load_state_dict(ck["state_dict"]); lstm.eval()
    lstm_score = standardized_score_fn(lstm, ck["median"], ck["iqr"])

    # OOR windows: max raw post-MinMax cell value exceeds 1.
    win_max = flat_xs.amax(dim=(1, 2))
    oor_idx = torch.nonzero(win_max > 1.0, as_tuple=False).flatten().tolist()
    oor_idx = oor_idx[:args.max_windows]
    print(f"[oor] {len(oor_idx)} out-of-range windows (of {n_test})")

    def audit(score_fn, label):
        epb, deg_undef, deg_def = [], [], []
        with torch.no_grad():
            for i in oor_idx:
                X = flat_xs[i:i + 1]
                Xc = X.clamp(0.0, 1.0)
                # input displacement from the clamp (detector-independent)
                epb.append(float((X - Xc).abs().max().item()))
                s_raw = score_fn(X)[0]
                s_clip = score_fn(Xc)[0]
                tgt = int(s_raw.argmax().item())
                # undefended: baseline raw vs attacked-clipped (asymmetric)
                deg_undef.append(float(s_raw[tgt].item() - s_clip[tgt].item()))
                # defended: both paths clipped -> symmetric -> no clamp gap
                deg_def.append(float(s_clip[tgt].item() - s_clip[tgt].item()))
        return {
            "epb_mean": float(np.mean(epb)), "epb_max": float(np.max(epb)),
            "ood_clamp_deg_undef_mean": float(np.mean(deg_undef)),
            "ood_clamp_deg_undef_median": float(np.median(deg_undef)),
            "ood_clamp_deg_defended_mean": float(np.mean(deg_def)),
            "n_oor_windows": len(oor_idx),
        }

    res = {"dataset": args.dataset, "seed": args.seed,
           "gdn": audit(gdn_score, "GDN"),
           "lstm": audit(lstm_score, "LSTM"),
           "lstm_clean_f1": ck.get("clean_f1")}

    out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")
    for det in ["gdn", "lstm"]:
        r = res[det]
        print(f"  {det.upper():5s}  EPB mean={r['epb_mean']:8.2f} max={r['epb_max']:8.2f}  "
              f"OOD-clamp deg (undef)={r['ood_clamp_deg_undef_mean']:+9.3f}  "
              f"(defended)={r['ood_clamp_deg_defended_mean']:+.3f}")


if __name__ == "__main__":
    main()
