"""
Defense component ablation on GDN-WADI: which of (input clip, threshold
recalibration) is necessary/sufficient for BETA-effectiveness reduction?

Variants:
  full           : clip(X)@inference  +  median/iqr from CLIPPED val   +  thr=99.5pct CLIPPED val
  clip_only      : clip(X)@inference  +  median/iqr from CLIPPED val   +  thr=MAX of UNCLIPPED val
  threshold_only : no clip            +  median/iqr from UNCLIPPED val +  thr=99.5pct UNCLIPPED val

Reports BETA mean target-score degradation per (window, target) trial,
matching Table 6 protocol (30 trials, --n-trials default). Sanity-checks
Full against existing reports/eval_defended_seed*.json values.

Usage:
  python scripts/defense_ablation.py --seed 0 --variant full --out reports/ablation/wadi_s0_full.json
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

from attack_env.test_split import get_anomaly_ranges_ds, split_ranges  # noqa: E402
from attacks.beta import BETAAttack, BETAConfig  # noqa: E402


def setup_gdn_victim(seed: int, dataset: str = "wadi"):
    repo = PROJECT_ROOT / "repos/GDN"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    ckpt_dir = repo / f"pretrained/{dataset}_seed{seed}"
    ckpt_path = "./" + str(list(ckpt_dir.glob("best_*.pt"))[-1].relative_to(repo).as_posix())

    if dataset == "swat":
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 64,
                        "slide_stride": 10, "comment": "ablation", "seed": seed,
                        "out_layer_num": 1, "out_layer_inter_dim": 128,
                        "decay": 0.0, "val_ratio": 0.1, "topk": 15}
    else:
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
                        "slide_stride": 10, "comment": "ablation", "seed": seed,
                        "out_layer_num": 1, "out_layer_inter_dim": 256,
                        "decay": 0.0, "val_ratio": 0.1, "topk": 30}

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
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    parser.add_argument("--variant", required=True,
                        choices=["full", "clip_only", "threshold_only"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    apply_clip = args.variant in ("full", "clip_only")
    val_clipped_for_stats = args.variant in ("full", "clip_only")

    print(f"[setup] GDN-{args.dataset} seed {args.seed}  variant={args.variant}  "
          f"clip@inference={apply_clip}  stats_from_clipped_val={val_clipped_for_stats}")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    val_scores_premedian = []  # for max-of-unclipped-val threshold
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xf = x.float()
            yf = y.float()
            if val_clipped_for_stats:
                xf_in = xf.clamp(0.0, 1.0); yf_in = yf.clamp(0.0, 1.0)
            else:
                xf_in = xf; yf_in = yf
            f = model(xf_in, ei.float())
            val_deltas.append((f - yf_in).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    EPS = 1e-2

    edge_index_template = next(iter(test_loader))[3].float()

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        if apply_clip:
            X_in = X.clamp(0.0, 1.0)
        else:
            X_in = X
        forecast = model(X_in, edge_index_template)
        target_step = X_in[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    if apply_clip:
        n_oor = ((flat_xs < 0) | (flat_xs > 1)).sum().item()
        flat_xs = flat_xs.clamp(0.0, 1.0)
        print(f"[clip] clipped {n_oor} out-of-[0,1] cells in test inputs")
    n_test, n_sensors, W = flat_xs.shape

    # ---- threshold computation per variant ----
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xf = x.float(); yf = y.float()
        if val_clipped_for_stats:
            xf_in = xf.clamp(0.0, 1.0); yf_in = yf.clamp(0.0, 1.0)
        else:
            xf_in = xf; yf_in = yf
        with torch.no_grad():
            f = model(xf_in, ei.float())
        s = ((f - yf_in).abs() - median) / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)

    if args.variant == "full":
        threshold = float(np.percentile(val_scores_arr, 99.5))
    elif args.variant == "clip_only":
        # max-of-val on clipped val (the undefended stand-in: max of whatever val produces)
        # Per mission spec: "use the original max-of-val threshold from undefended training (no recalibration)".
        # The undefended training used max of UNCLIPPED val; we recompute that here under no-clip val.
        unclipped_val_scores = []
        with torch.no_grad():
            for batch in val_loader:
                x, y, _, ei = batch
                xf = x.float(); yf = y.float()
                f = model(xf, ei.float())
                s = ((f - yf).abs() - median) / (iqr_v.abs() + EPS)
                unclipped_val_scores.append(s.cpu().numpy())
        unclipped_arr = np.concatenate(unclipped_val_scores, axis=0)
        threshold = float(unclipped_arr.max())
    else:  # threshold_only
        threshold = float(np.percentile(val_scores_arr, 99.5))

    print(f"[threshold] variant={args.variant} threshold={threshold:.4f}  "
          f"(val_max={val_scores_arr.max():.4f})")

    # ---- build holdout pool + target rotation under this variant's surrogate ----
    if args.dataset == "wadi":
        ranges = get_anomaly_ranges_ds("wadi", num_test_rows_ds=n_test + W - 1)
    else:
        ranges = get_anomaly_ranges_ds("swat")
    _, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start_target_row = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    holdout_pool = list(range(max(0, holdout_start_target_row - W + 1), n_test))

    detected_in_holdout = []
    threshold_crossings = np.zeros(n_sensors, dtype=np.int64)
    BATCH = 256
    holdout_set = set(holdout_pool)
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            scores = surrogate_query(X)
            for i in range(X.shape[0]):
                gi = start + i
                if gi not in holdout_set:
                    continue
                if scores[i].max().item() > threshold:
                    detected_in_holdout.append(gi)
                threshold_crossings += (scores[i] > threshold).cpu().numpy().astype(np.int64)

    crossing_counts = [(int(s), int(threshold_crossings[s])) for s in range(n_sensors)
                       if threshold_crossings[s] > 0]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:args.n_targets]]
    print(f"[holdout] detected={len(detected_in_holdout)}  rotation={target_rotation}")

    if not detected_in_holdout or not target_rotation:
        result = {"variant": args.variant, "seed": args.seed, "dataset": args.dataset,
                  "n_detected_holdout": len(detected_in_holdout),
                  "target_rotation": target_rotation,
                  "beta_degradation_mean": 0.0, "beta_degradation_std": 0.0,
                  "threshold": threshold, "n_trials_completed": 0,
                  "note": "no detection under this variant — degenerate"}
        out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print("[result] no detection → degenerate output written")
        return

    rng = random.Random(args.seed)
    pairs = [(w, t) for w in detected_in_holdout for t in target_rotation]
    rng.shuffle(pairs)
    sample = pairs[:args.n_trials]

    beta_cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    beta_attack = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                             num_nodes=n_sensors, config=beta_cfg)

    print(f"[eval] {len(sample)} trials × BETA")
    beta_degradations = []
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean_score = surrogate_query(X)[0, target_idx].item()
        try:
            X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            beta_degradations.append(clean_score - attacked_score)
        except Exception as e:
            print(f"  [warn] trial {trial_i}: BETA exception {type(e).__name__}: {e}")
            beta_degradations.append(0.0)
        if (trial_i + 1) % 10 == 0:
            print(f"  trial {trial_i+1}/{len(sample)}  BETA mean={np.mean(beta_degradations):+.4f}")

    beta_mean = float(np.mean(beta_degradations))
    beta_std = float(np.std(beta_degradations))
    result = {
        "variant": args.variant, "seed": args.seed, "dataset": args.dataset,
        "n_trials": len(sample), "budget": args.budget,
        "n_detected_holdout": len(detected_in_holdout),
        "target_rotation": target_rotation,
        "threshold": threshold,
        "beta_degradation_mean": beta_mean,
        "beta_degradation_std": beta_std,
        "beta_degradation_per_trial": beta_degradations,
    }
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[result] BETA mean={beta_mean:+.4f}±{beta_std:.4f}  → {out_path}")


if __name__ == "__main__":
    main()
