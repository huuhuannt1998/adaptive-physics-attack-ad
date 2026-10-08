"""
Sweep thresholds and find F1-optimal threshold for a given (dataset, seed) detector.
Output: threshold value + F1 + Precision + Recall at that threshold.

The published GDN F1 of 0.41 on WADI is achieved with point-adjusted F1 + a specific
threshold; raw F1 at max-of-val is ~0. This script finds the raw-F1-optimal threshold.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_gdn_victim(seed: int, dataset: str, arch: str = "GDN"):
    repo_name = "TopoGDN" if arch.lower() == "topogdn" else "GDN"
    repo = PROJECT_ROOT / f"repos/{repo_name}"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ckpt_dir = repo / f"pretrained/{dataset}_seed{seed}"
    ckpt_path = "./" + str(list(ckpt_dir.glob("best_*.pt"))[-1].relative_to(repo).as_posix())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="wadi")
    parser.add_argument("--defended", action="store_true")
    parser.add_argument("--arch", default="GDN", choices=["GDN", "TopoGDN"])
    parser.add_argument("--hk-defense", action="store_true",
                        help="(TopoGDN only) Install HK PI-stability defense.")
    parser.add_argument("--hk-refs", default=None)
    parser.add_argument("--hk-projection-radius", type=float, default=2.0)
    parser.add_argument("--hk-sigma", type=float, default=0.05)
    parser.add_argument("--hk-persistence-clamp", type=float, default=1.0)
    args = parser.parse_args()

    print(f"[setup] {args.arch}-{args.dataset} seed {args.seed} defended={args.defended} hk={args.hk_defense}")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset, arch=args.arch)
    model = m.model
    is_topo = args.arch.lower() == "topogdn"

    if args.hk_defense:
        if not is_topo:
            raise ValueError("--hk-defense requires --arch TopoGDN")
        from defenses.hk_stability import HKConfig, HKStabilityWrapper
        refs_path = args.hk_refs or f"reports/hk_refs/{args.dataset}_seed{args.seed}.pt"
        refs_path_full = refs_path if Path(refs_path).is_absolute() else str(PROJECT_ROOT / refs_path)
        print(f"[hk] loading reference PIs from {refs_path_full}")
        refs = torch.load(refs_path_full, weights_only=False, map_location="cpu")
        ref_p0 = refs["ref_p0"]
        hk_cfg = HKConfig(
            grid_size=ref_p0.shape[-1],
            sigma=args.hk_sigma,
            persistence_clamp=args.hk_persistence_clamp,
            projection_radius=args.hk_projection_radius,
        )
        wrapper = HKStabilityWrapper(model, clean_references=ref_p0, config=hk_cfg)
        n_patched = 0
        for _name, mod in model.named_modules():
            if mod.__class__.__name__ == "TopologyLayer":
                wrapper._install_hook(mod)
                n_patched += 1
        print(f"[hk] patched {n_patched} TopologyLayer instance(s)")
    def model_fwd(X, ei=None):
        if is_topo:
            out = model(X)
            return out[0] if isinstance(out, tuple) else out
        return model(X, ei)
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            x, y, _, ei = batch
            xf, yf = x.float(), y.float()
            if args.defended:
                xf = xf.clamp(0, 1); yf = yf.clamp(0, 1)
            f = model_fwd(xf, ei.float())
            if bi == 0:
                print(f"  [debug] f.shape={tuple(f.shape)} f.dtype={f.dtype} yf.dtype={yf.dtype}")
            d = (f - yf).abs()
            if bi == 0:
                print(f"  [debug] (f-yf).abs().dtype={d.dtype}")
            try:
                arr = d.detach().float().cpu().numpy()
            except Exception as e:
                # Fallback: route through Python list to bypass torch->numpy interop quirk.
                print(f"  [warn] .numpy() failed at batch {bi}: {type(e).__name__}: {e}; falling back to .tolist()")
                arr = np.asarray(d.detach().float().cpu().tolist(), dtype=np.float32)
            val_deltas.append(arr)
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    test_labels = torch.cat([b[2].float() for b in test_batches], dim=0).cpu().numpy().astype(int)
    if args.defended:
        flat_xs = flat_xs.clamp(0, 1)
    n_test, n_sensors, W = flat_xs.shape

    # Compute per-window max-sensor standardized score
    BATCH = 256
    scores_max = np.zeros(n_test, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            if args.defended:
                X = X.clamp(0, 1)
            forecast = model_fwd(X, edge_index_template)
            target_step = X[..., -1]
            delta = (forecast - target_step).abs()
            s = (delta - median) / (iqr_v.abs() + EPS)
            mx = s.max(dim=1)[0].detach().float().cpu().numpy()
            scores_max[start:start + len(mx)] = mx

    # Sweep thresholds
    candidates = list(np.linspace(scores_max.min(), scores_max.max(), 200))
    candidates += list(np.percentile(scores_max, [50, 75, 80, 85, 90, 95, 97, 98, 99, 99.5, 99.9]))
    best_f1 = -1.0; best_thresh = None; best_p = None; best_r = None
    for t in candidates:
        pred = scores_max > t
        tp = int(((pred == 1) & (test_labels == 1)).sum())
        fp = int(((pred == 1) & (test_labels == 0)).sum())
        fn = int(((pred == 0) & (test_labels == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1; best_thresh = float(t); best_p = p; best_r = r

    print(f"[result] F1-optimal threshold = {best_thresh:.4f}")
    print(f"         F1 = {best_f1:.4f}  (P={best_p:.4f}  R={best_r:.4f})")
    print(f"         compare: max-of-val = {val_deltas.max():.4f}")
    print(f"         compare: 99.5 percentile of val = {np.percentile(np.concatenate([(val_deltas - median.numpy())/(iqr_v.numpy()+EPS)], axis=0).max(axis=1), 99.5):.4f}")


if __name__ == "__main__":
    main()
