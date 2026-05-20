"""
FGSM-with-budget defended-distribution eval (W5 of mis P-reviewer2-response).
Reuses the same defense + holdout + matched-eval protocol as
eval_defended_detector.py; swaps BETA-PGD for the FGSM-single-step attack.

Outputs per-trial degradations to populate the M5 success-rate column for
the FGSM attack as well as mean/std target-score degradation.
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
from attacks.fgsm import FGSMAttack, FGSMConfig  # noqa: E402


def setup_gdn_victim(seed: int, dataset: str = "wadi"):
    repo = PROJECT_ROOT / "repos/GDN"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ckpt_dir = repo / f"pretrained/{dataset}_seed{seed}"
    ckpt_path = "./" + str(list(ckpt_dir.glob("best_*.pt"))[-1].relative_to(repo).as_posix())
    if dataset == "swat":
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 64,
                        "slide_stride": 10, "comment": "fgsm", "seed": seed,
                        "out_layer_num": 1, "out_layer_inter_dim": 128,
                        "decay": 0.0, "val_ratio": 0.1, "topk": 15}
    else:
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
                        "slide_stride": 10, "comment": "fgsm", "seed": seed,
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
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[setup] FGSM defended eval GDN-{args.dataset} seed {args.seed} eps={args.epsilon}")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xc = x.float().clamp(0.0, 1.0); yc = y.float().clamp(0.0, 1.0)
            f = model(xc, ei.float())
            val_deltas.append((f - yc).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def surrogate_query(X):
        Xc = X.clamp(0.0, 1.0)
        forecast = model(Xc, edge_index_template)
        target_step = Xc[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    n_oor = ((flat_xs < 0) | (flat_xs > 1)).sum().item()
    flat_xs = flat_xs.clamp(0.0, 1.0)
    n_test, n_sensors, W = flat_xs.shape

    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xc = x.float().clamp(0.0, 1.0); yc = y.float().clamp(0.0, 1.0)
        with torch.no_grad():
            f = model(xc, ei.float())
        s = ((f - yc).abs() - median) / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold = float(np.percentile(val_scores_arr, 99.5))

    if args.dataset == "wadi":
        ranges = get_anomaly_ranges_ds("wadi", num_test_rows_ds=n_test + W - 1)
    else:
        ranges = get_anomaly_ranges_ds("swat")
    _, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    holdout_pool = list(range(max(0, holdout_start - W + 1), n_test))

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
                if gi not in holdout_set: continue
                if scores[i].max().item() > threshold:
                    detected_in_holdout.append(gi)
                threshold_crossings += (scores[i] > threshold).cpu().numpy().astype(np.int64)
    crossing_counts = [(int(s), int(threshold_crossings[s])) for s in range(n_sensors)
                       if threshold_crossings[s] > 0]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:args.n_targets]]

    rng = random.Random(args.seed)
    pairs = [(w, t) for w in detected_in_holdout for t in target_rotation]
    rng.shuffle(pairs)
    sample = pairs[:args.n_trials]

    fgsm_cfg = FGSMConfig(epsilon=args.epsilon)
    fgsm_attack = FGSMAttack(surrogate_query, lambda: model.learned_graph.detach(),
                             num_nodes=n_sensors, config=fgsm_cfg)

    fgsm_degradations = []
    fgsm_flips = []
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean_score = surrogate_query(X)[0, target_idx].item()
        try:
            X_pert, _ = fgsm_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            fgsm_degradations.append(clean_score - attacked_score)
            fgsm_flips.append(1.0 if (clean_score > threshold and attacked_score < threshold) else 0.0)
        except Exception as e:
            print(f"  [warn] trial {trial_i}: FGSM exception {type(e).__name__}: {e}")
            fgsm_degradations.append(0.0)
            fgsm_flips.append(0.0)

    fgsm_mean = float(np.mean(fgsm_degradations))
    fgsm_std = float(np.std(fgsm_degradations))
    result = {
        "defended": True, "attack": "fgsm",
        "seed": args.seed, "dataset": args.dataset,
        "n_trials": len(sample), "budget": args.budget,
        "epsilon": args.epsilon,
        "n_oor_clipped": n_oor,
        "threshold": threshold,
        "fgsm_degradation_mean": fgsm_mean,
        "fgsm_degradation_std": fgsm_std,
        "fgsm_per_trial_degradations": [float(d) for d in fgsm_degradations],
        "fgsm_flip_rate": float(np.mean(fgsm_flips)),
        "fgsm_success_rate": float(np.mean([d > 0 for d in fgsm_degradations])),
    }
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[result] FGSM defended mean={fgsm_mean:+.4f}±{fgsm_std:.4f}  success_rate={result['fgsm_success_rate']:.1%}  -> {out_path}")


if __name__ == "__main__":
    main()
