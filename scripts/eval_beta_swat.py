"""
BETA-only sanity eval on GDN-SWaT seed 0. No policy involved — just measures
BETA's attack effectiveness on SWaT to test whether the OOD-clamp dominance
observed on WADI (sensor 106 driving 18,900 mean degradation) generalizes.

Reports per-target-sensor BETA degradation distribution to identify SWaT's
equivalent OOD-attackable sensors, if any.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attacks.beta import BETAAttack, BETAConfig  # noqa: E402


def setup_victim(seed: int, dataset: str = "swat", arch: str = "GDN"):
    repo_name = "TopoGDN" if arch.lower() == "topogdn" else "GDN"
    repo = PROJECT_ROOT / f"repos/{repo_name}"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ckpt_dir = repo / f"pretrained/{dataset}_seed{seed}"
    candidates = list(ckpt_dir.glob("best_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    ckpt_path = "./" + str(candidates[-1].relative_to(repo).as_posix())

    if dataset == "swat":
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 64,
            "slide_stride": 10, "comment": "beta_eval", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "beta_eval", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 256,
            "decay": 0.0, "val_ratio": 0.1, "topk": 30,
        }
    if arch.lower() == "topogdn":
        train_config["use_tcn"] = True
        train_config["use_topo"] = True
        train_config["model"] = "GDN"  # TopoGDN's main.py routes "GDN" -> TopoGDN with use_topo
    env_config = {
        "save_path": f"{dataset}_seed{seed}", "dataset": dataset, "report": "best",
        "device": "cpu", "load_model_path": ckpt_path,
    }
    # IMPORTANT: clear cached `main` module if previously imported from another repo
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()
    return m


# Backward-compat alias
setup_gdn_victim = setup_victim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="swat", choices=["wadi", "swat"])
    parser.add_argument("--arch", default="GDN", choices=["GDN", "TopoGDN"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out", default="reports/beta_eval.json")
    parser.add_argument("--defended", action="store_true",
                        help="Pre-clip test inputs to [0,1] before BETA (defended-detector eval)")
    parser.add_argument("--threshold-percentile", type=float, default=100.0,
                        help="Percentile of val score distribution for threshold (100=max-of-val)")
    args = parser.parse_args()

    print(f"[setup] loading {args.arch}-{args.dataset} seed {args.seed}")
    m = setup_victim(seed=args.seed, dataset=args.dataset, arch=args.arch)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    is_topo = (args.arch.lower() == "topogdn")
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xf, yf = x.float(), y.float()
            if args.defended:
                xf = xf.clamp(0.0, 1.0)
                yf = yf.clamp(0.0, 1.0)
            if is_topo:
                out = model(xf)
                f = out[0] if isinstance(out, tuple) else out
            else:
                f = model(xf, ei.float())
            val_deltas.append((f - yf).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def model_forward(X):
        if is_topo:
            out = model(X)
            return out[0] if isinstance(out, tuple) else out
        return model(X, edge_index_template)

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        Xf = X.clamp(0.0, 1.0) if args.defended else X
        forecast = model_forward(Xf)
        target_step = Xf[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    n_oor_pre = ((flat_xs < 0) | (flat_xs > 1)).sum().item()
    if args.defended:
        flat_xs = flat_xs.clamp(0.0, 1.0)
        print(f"[defended] clipped {n_oor_pre} OOR cells in test inputs")
    n_test, n_sensors, W = flat_xs.shape
    print(f"[data] test={n_test} windows, n_sensors={n_sensors}, W={W}")

    # Pre-clip stats
    n_oor = ((flat_xs < 0) | (flat_xs > 1)).sum().item()
    sensor_max = flat_xs.amax(dim=(0, 2))
    sensor_min = flat_xs.amin(dim=(0, 2))
    out_of_unit_sensors = [(int(i), float(sensor_min[i]), float(sensor_max[i]))
                           for i in range(n_sensors)
                           if sensor_max[i] > 1.5 or sensor_min[i] < -0.5]
    print(f"[data] OOR cells: {n_oor} ({100*n_oor/(n_test*n_sensors*W):.3f}%); "
          f"OOR sensors: {[(i, f'{lo:.2f}/{hi:.2f}') for i, lo, hi in out_of_unit_sensors[:10]]}")

    # Detect on test
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xf, yf = x.float(), y.float()
        if args.defended:
            xf = xf.clamp(0.0, 1.0)
            yf = yf.clamp(0.0, 1.0)
        with torch.no_grad():
            f = model_forward(xf)
        s = (f - yf).abs() - median
        s = s / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    if args.threshold_percentile >= 100.0:
        threshold = float(val_scores_arr.max())
    else:
        threshold = float(np.percentile(val_scores_arr, args.threshold_percentile))

    detected: list[tuple[int, int]] = []
    BATCH = 256
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            scores = surrogate_query(X)
            max_scores, max_idx = scores.max(dim=1)
            for i in range(X.shape[0]):
                global_i = start + i
                if max_scores[i].item() > threshold:
                    detected.append((global_i, int(max_idx[i].item())))
    print(f"[detect] {len(detected)} windows fire threshold ({threshold:.4f})")

    rng = random.Random(args.seed)
    sample = rng.sample(detected, min(args.n_trials, len(detected)))
    print(f"[eval] running BETA on {len(sample)} (window, argmax-target) trials")

    beta_cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    beta_attack = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                             num_nodes=n_sensors, config=beta_cfg)

    per_target_degradations = defaultdict(list)
    all_degradations = []
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean_score = surrogate_query(X)[0, target_idx].item()
        try:
            X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            deg = clean_score - attacked_score
        except Exception:
            deg = 0.0
        per_target_degradations[target_idx].append(deg)
        all_degradations.append(deg)
        if (trial_i + 1) % 10 == 0:
            print(f"  trial {trial_i+1}/{len(sample)}  mean={np.mean(all_degradations):+.3f}  "
                  f"unique targets={len(per_target_degradations)}")

    print()
    print("=== Per-target BETA degradation profile ===")
    for tgt, degs in sorted(per_target_degradations.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print(f"  sensor {tgt:3d}  n={len(degs):2d}  mean_deg={np.mean(degs):+12.4f}  "
              f"max_deg={max(degs):+12.4f}")

    result = {
        "seed": args.seed,
        "dataset": args.dataset,
        "arch": args.arch,
        "n_trials": len(sample),
        "n_unique_targets": len(per_target_degradations),
        "n_oor_cells": n_oor,
        "out_of_unit_sensors": out_of_unit_sensors,
        "threshold": threshold,
        "n_detected": len(detected),
        "all_degradations_mean": float(np.mean(all_degradations)),
        "all_degradations_std": float(np.std(all_degradations)),
        "per_target_mean": {int(k): float(np.mean(v)) for k, v in per_target_degradations.items()},
        "per_target_n": {int(k): len(v) for k, v in per_target_degradations.items()},
    }
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[write] {out_path}")
    print(f"\n=== Summary ===")
    print(f"  All-target BETA mean degradation: {np.mean(all_degradations):+.4f} ± {np.std(all_degradations):.4f}")


if __name__ == "__main__":
    main()
