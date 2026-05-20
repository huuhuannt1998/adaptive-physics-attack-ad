"""
Calibration-split experiment: deployable threshold selection without test labels.

Protocol (per detector, dataset, seed under input-clip defense):
  1. Compute clipped val median/IQR for residual standardization.
  2. Compute clipped-test window scores + labels.
  3. For K random Cal/HO splits (50/50 stratified by label):
       a. Pick threshold tau that maximizes F1 on (scores[Cal], labels[Cal]).
       b. Report F1 + Precision + Recall + FPR on HO with that tau.
  4. Aggregate mean +/- std across K splits.

Comparators (also reported, same config):
  - F1-optimal on full test (oracle, non-deployable upper bound).
  - val-99.5 percentile recipe (max-aggregated standardized residual).

Output: per-seed JSON to reports/calibration_split/<det>_<dataset>_seed<s>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EPS = 1e-2


def setup_victim(seed: int, dataset: str, arch: str = "GDN"):
    repo_name = "TopoGDN" if arch.lower() == "topogdn" else "GDN"
    repo = PROJECT_ROOT / f"repos/{repo_name}"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ckpt_dir = repo / f"pretrained/{dataset}_seed{seed}"
    ckpts = sorted(ckpt_dir.glob("best_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no best_*.pt under {ckpt_dir}")
    ckpt_path = "./" + str(ckpts[-1].relative_to(repo).as_posix())
    if dataset == "swat":
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 64,
                        "slide_stride": 10, "comment": "thresh", "seed": seed,
                        "out_layer_num": 1, "out_layer_inter_dim": 128,
                        "decay": 0.0, "val_ratio": 0.1, "topk": 15}
    else:
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
                        "slide_stride": 10, "comment": "thresh", "seed": seed,
                        "out_layer_num": 1, "out_layer_inter_dim": 256,
                        "decay": 0.0, "val_ratio": 0.1, "topk": 30}
    if arch.lower() == "topogdn":
        train_config["use_tcn"] = True
        train_config["use_topo"] = True
        train_config["model"] = "GDN"
    env_config = {"save_path": f"{dataset}_seed{seed}", "dataset": dataset, "report": "best",
                  "device": "cpu", "load_model_path": ckpt_path}
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()
    return m


def compute_clipped_scores(m, defended: bool, arch: str):
    from scipy.stats import iqr as scipy_iqr
    model = m.model
    is_topo = arch.lower() == "topogdn"

    def fwd(X, ei=None):
        if is_topo:
            out = model(X)
            return out[0] if isinstance(out, tuple) else out
        return model(X, ei)

    val_loader = m.val_dataloader
    test_loader = m.test_dataloader

    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xf, yf = x.float(), y.float()
            if defended:
                xf = xf.clamp(0, 1); yf = yf.clamp(0, 1)
            f = fwd(xf, ei.float())
            val_deltas.append((f - yf).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    labels = torch.cat([b[2].float() for b in test_batches], dim=0).cpu().numpy().astype(int)
    edge_index_template = test_batches[0][3].float()
    if defended:
        flat_xs = flat_xs.clamp(0, 1)
    n_test = flat_xs.shape[0]
    BATCH = 256
    scores_max = np.zeros(n_test, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            if defended:
                X = X.clamp(0, 1)
            forecast = fwd(X, edge_index_template)
            target_step = X[..., -1]
            delta = (forecast - target_step).abs()
            s = (delta - median) / (iqr_v.abs() + EPS)
            mx = s.max(dim=1)[0].cpu().numpy()
            scores_max[start:start + len(mx)] = mx

    # val-99.5 recipe: 99.5pct of max-aggregated standardized val residuals
    val_std = ((val_deltas - median.numpy()) / (iqr_v.numpy() + EPS))
    val_max = val_std.max(axis=1)
    val_995 = float(np.percentile(val_max, 99.5))

    return scores_max, labels, val_995


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float):
    pred = scores > t
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return {"f1": 0.0, "p": 0.0, "r": 0.0, "fpr": fp / max(fp + tn, 1)}
    p = tp / (tp + fp); r = tp / (tp + fn)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"f1": f1, "p": p, "r": r, "fpr": fp / max(fp + tn, 1)}


def best_threshold(scores: np.ndarray, labels: np.ndarray):
    candidates = list(np.linspace(scores.min(), scores.max(), 200))
    candidates += list(np.percentile(scores, [50, 75, 80, 85, 90, 95, 97, 98, 99, 99.5, 99.9]))
    best = {"f1": -1.0, "t": float(scores.min()), "p": 0.0, "r": 0.0, "fpr": 0.0}
    for t in candidates:
        m = f1_at_threshold(scores, labels, float(t))
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "t": float(t), "p": m["p"], "r": m["r"], "fpr": m["fpr"]}
    return best


def stratified_split(labels: np.ndarray, rng: np.random.RandomState, cal_frac: float = 0.5):
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_cal_pos = int(len(pos) * cal_frac); n_cal_neg = int(len(neg) * cal_frac)
    cal_idx = np.concatenate([pos[:n_cal_pos], neg[:n_cal_neg]])
    ho_idx = np.concatenate([pos[n_cal_pos:], neg[n_cal_neg:]])
    return cal_idx, ho_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arch", default="GDN", choices=["GDN", "TopoGDN"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"[setup] {args.arch}-{args.dataset} seed {args.seed} defended=True")
    m = setup_victim(seed=args.seed, dataset=args.dataset, arch=args.arch)
    scores, labels, val_995 = compute_clipped_scores(m, defended=True, arch=args.arch)
    print(f"[scores] n_test={len(labels)} anomaly_rate={labels.mean():.4f}")
    print(f"[val-995] threshold={val_995:.4f}")

    # Oracle (non-deployable upper bound): F1-optimal on full test
    oracle = best_threshold(scores, labels)
    val995_full = f1_at_threshold(scores, labels, val_995)
    print(f"[oracle] F1={oracle['f1']:.4f} P={oracle['p']:.4f} R={oracle['r']:.4f} FPR={oracle['fpr']:.4f}")
    print(f"[val995] F1={val995_full['f1']:.4f} P={val995_full['p']:.4f} R={val995_full['r']:.4f} FPR={val995_full['fpr']:.4f}")

    # Calibration-split: K splits
    rng = np.random.RandomState(args.split_seed)
    split_results = []
    for k in range(args.n_splits):
        cal_idx, ho_idx = stratified_split(labels, rng)
        cal_scores, cal_labels = scores[cal_idx], labels[cal_idx]
        ho_scores, ho_labels = scores[ho_idx], labels[ho_idx]
        cal_best = best_threshold(cal_scores, cal_labels)
        ho_metrics = f1_at_threshold(ho_scores, ho_labels, cal_best["t"])
        split_results.append({
            "split": k,
            "cal_best_t": cal_best["t"],
            "cal_f1": cal_best["f1"],
            "ho_f1": ho_metrics["f1"],
            "ho_p": ho_metrics["p"],
            "ho_r": ho_metrics["r"],
            "ho_fpr": ho_metrics["fpr"],
        })
        print(f"[split {k}] cal_t={cal_best['t']:.4f} cal_F1={cal_best['f1']:.4f} -> HO: F1={ho_metrics['f1']:.4f} P={ho_metrics['p']:.4f} R={ho_metrics['r']:.4f} FPR={ho_metrics['fpr']:.4f}")

    ho_f1s = np.array([r["ho_f1"] for r in split_results])
    ho_ps = np.array([r["ho_p"] for r in split_results])
    ho_rs = np.array([r["ho_r"] for r in split_results])
    ho_fprs = np.array([r["ho_fpr"] for r in split_results])

    summary = {
        "arch": args.arch, "dataset": args.dataset, "seed": args.seed, "defended": True,
        "n_test": int(len(labels)), "anomaly_rate": float(labels.mean()),
        "oracle": oracle, "val_995": {"threshold": val_995, **val995_full},
        "cal_split": {
            "n_splits": args.n_splits, "split_seed": args.split_seed,
            "ho_f1_mean": float(ho_f1s.mean()), "ho_f1_std": float(ho_f1s.std()),
            "ho_p_mean": float(ho_ps.mean()), "ho_r_mean": float(ho_rs.mean()),
            "ho_fpr_mean": float(ho_fprs.mean()),
            "splits": split_results,
        },
    }
    print(f"[summary] cal-split HO F1 = {summary['cal_split']['ho_f1_mean']:.4f} +/- {summary['cal_split']['ho_f1_std']:.4f}")
    print(f"          cal-split HO FPR = {summary['cal_split']['ho_fpr_mean']:.4f}")
    print(f"          vs oracle F1 {oracle['f1']:.4f}, val-995 F1 {val995_full['f1']:.4f}")

    out_path = Path(args.out) if args.out else (PROJECT_ROOT / f"reports/calibration_split/{args.arch.lower()}_{args.dataset}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
