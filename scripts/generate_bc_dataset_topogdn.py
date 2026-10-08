"""
Generate Behavior Cloning dataset by running BETA reimpl on train-range
anomaly windows and logging (state, action) pairs.

Protocol:
  1. Load victim checkpoint (GDN-WADI seed 0 for now; iterate per seed at PPO time).
  2. Set up GreyBoxVictim wrapper as the surrogate.
  3. Get train-range anomaly windows (per test_split).
  4. For each detected window in train-set ranges:
     a. Pick target_idx = argmax sensor.
     b. Run BETA: select_candidate_nodes → eigenvector_centrality prune → PGD.
     c. Reset MDP env on this (window, target).
     d. For each of B=5 influencer sensors in BETA's V_bar order:
        - Snapshot MDP state.
        - Step env with action = (sensor_idx_in_categorical_form, BETA's delta for that sensor).
        - Log (state, action) pair.
  5. Save dataset to disk as .pt.

Output dataset format:
  - List of dicts with keys: state (MDP state dict), action (categorical idx, delta tensor).
  - Saved as Python list via torch.save.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attack_env.test_split import (  # noqa: E402
    get_anomaly_ranges_ds, split_ranges, windows_for_ranges,
)
from attack_env.mdp import AdaptiveAttackEnv, MDPConfig, compute_nominal_distributions  # noqa: E402
from attack_env.grey_box import GreyBoxVictim, StandardizationStats  # noqa: E402
from attacks.beta import (  # noqa: E402
    BETAConfig, eigenvector_centrality, prune_to_budget, select_candidate_nodes,
)


def setup_gdn_victim(seed: int = 0, dataset: str = "wadi"):
    """Load TopoGDN victim for given dataset/seed. Returns Main object.
    Function name preserved for minimal diff vs generate_bc_dataset.py.
    """
    repo = PROJECT_ROOT / "repos/TopoGDN"
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
            "slide_stride": 10, "comment": "bc_gen_topogdn", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
            "use_tcn": True, "use_topo": True, "model": "GDN",
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "bc_gen_topogdn", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 256,
            "decay": 0.0, "val_ratio": 0.1, "topk": 30,
            "use_tcn": True, "use_topo": True, "model": "GDN",
        }
    env_config = {
        "save_path": f"{dataset}_seed{seed}", "dataset": dataset, "report": "best",
        "device": "cpu", "load_model_path": ckpt_path,
    }

    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()

    return m


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="Output dataset path (default: bc_dataset_gdn_{dataset}_seed{seed}.pt)")
    parser.add_argument("--max-windows", type=int, default=10000,
                        help="Cap on number of windows to process (default 10K)")
    parser.add_argument("--n-targets", type=int, default=10,
                        help="Number of target sensors per window")
    parser.add_argument("--max-pairs", type=int, default=50000,
                        help="Subsample to this many pairs after generation")
    parser.add_argument("--exclude-sensor", type=int, default=-1,
                        help="Sensor to exclude from rotation (-1 = none; for WADI use 106)")
    args = parser.parse_args()
    if args.exclude_sensor < 0:
        args.exclude_sensor = 106 if args.dataset == "wadi" else -1

    print(f"[setup] loading GDN-{args.dataset} seed {args.seed}")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    # Build standardization stats (median + IQR over val residuals).
    from scipy.stats import iqr as scipy_iqr

    def _model_forward(X, edge_index=None):
        """TopoGDN forward: model(X) returns (forecast, learned_graph). ei unused."""
        out = model(X)
        return out[0] if isinstance(out, tuple) else out

    print("[stats] computing val-set per-sensor median + IQR")
    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            forecast = _model_forward(x.float(), ei.float())
            val_deltas.append((forecast - y.float()).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()

    # Set up surrogate query function: standardized scores from model + median/iqr.
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        forecast = _model_forward(X, edge_index_template)
        target_step = X[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    # Get test windows; apply temporal pre-holdout filter.
    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    n_test, n_sensors, W = flat_xs.shape
    print(f"[data] test set: {n_test} windows, {n_sensors} sensors, window={W}")

    if args.dataset == "wadi":
        ranges = get_anomaly_ranges_ds("wadi", num_test_rows_ds=n_test + W - 1)
    else:
        ranges = get_anomaly_ranges_ds("swat")
    train_ranges, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start_target_row = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    # Pre-holdout temporal pool: any window whose prediction-target row falls before holdout starts.
    # window_i predicts row i + W - 1; we want i + W - 1 < holdout_start_target_row.
    pre_holdout_windows = list(range(0, max(0, holdout_start_target_row - W + 1)))
    pre_holdout_set = set(pre_holdout_windows)
    print(f"[split] {len(train_ranges)} train ranges, {len(holdout_ranges)} holdout ranges; "
          f"pre-holdout temporal pool: {len(pre_holdout_set)} candidate windows "
          f"(holdout starts at ds-row {holdout_start_target_row})")

    # Filter to "detected" windows (max sensor score > val-score-max threshold).
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        with torch.no_grad():
            f = _model_forward(x.float(), ei.float())
        delta = (f - y.float()).abs()
        s = (delta - median) / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold = float(val_scores_arr.max())
    print(f"[threshold] max-of-val standardized score = {threshold:.4f}")

    # Detect via max-score threshold; ALSO track per-sensor threshold-crossing frequency
    # over the full pre-holdout pool (not just argmax) so target rotation reflects
    # diversity rather than collapsing to sensor 106's chronic dominance.
    detected_train_windows: list[tuple[int, int]] = []  # (window_idx, argmax_sensor)
    threshold_crossings = np.zeros(n_sensors, dtype=np.int64)
    BATCH = 256
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            scores = surrogate_query(X)
            max_scores, max_idx = scores.max(dim=1)
            for i in range(X.shape[0]):
                global_i = start + i
                if global_i not in pre_holdout_set:
                    continue
                if max_scores[i].item() > threshold:
                    detected_train_windows.append((global_i, int(max_idx[i].item())))
                # accumulate per-sensor crossings on this window regardless of argmax
                crossings_i = (scores[i] > threshold).cpu().numpy()
                threshold_crossings += crossings_i.astype(np.int64)
    print(f"[detect] detected pre-holdout windows: {len(detected_train_windows)}")

    if len(detected_train_windows) > args.max_windows:
        rng = random.Random(args.seed)
        detected_train_windows = rng.sample(detected_train_windows, args.max_windows)
        print(f"[subsample] capped at {args.max_windows} windows")

    # Target rotation: pick top-N by threshold-crossing frequency (not argmax),
    # excluding sensor 106. argmax is dominated by sensor 106 (chronic mispredict),
    # while threshold-crossing reveals genuinely attackable sensors.
    crossing_counts = [(int(s), int(threshold_crossings[s])) for s in range(n_sensors)
                       if threshold_crossings[s] > 0 and s != args.exclude_sensor]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:args.n_targets]]
    print(f"[targets] target rotation ({len(target_rotation)} sensors, excl sensor {args.exclude_sensor}, "
          f"by threshold-crossing freq): {[(s, threshold_crossings[s]) for s in target_rotation]}")
    if len(target_rotation) == 0:
        raise RuntimeError("No target sensors after exclusion")

    # Compute nominal distributions (for KS stealth term — env needs this even though λ_2 not active in BC).
    print("[nominal] computing per-sensor nominal distributions (sample 1024 per sensor)")
    train_orig = pd.read_csv(PROJECT_ROOT / f"repos/TopoGDN/data/{args.dataset}/train.csv", index_col=0)
    train_x = torch.from_numpy(train_orig.drop(columns=["attack"]).values).float()  # (T, N)
    # Reshape to (T-W+1, N, W) windowed... simpler: just use flat sensor values.
    nominal = torch.zeros(n_sensors, 1024)
    for i in range(n_sensors):
        sample = train_x[:, i]
        if len(sample) < 1024:
            nominal[i] = sample.repeat((1024 // len(sample) + 1))[:1024]
        else:
            idx = torch.randperm(len(sample), generator=torch.Generator().manual_seed(args.seed))[:1024]
            nominal[i] = sample[idx]

    # Set up MDP env.
    cfg = MDPConfig(
        epsilon_pgd=0.1,
        target_score_history_len=5,
        epsilon_reward_floor=0.1,
        lambda_1_sparsity=0.0,  # BC dataset doesn't penalize sparsity
        lambda_2_stealth=0.0,
        lambda_3_physics=0.0,
        per_window_budget_default=args.budget,
        seed=args.seed,
    )
    env = AdaptiveAttackEnv(surrogate_query, nominal, cfg)

    # Run BETA pipeline per window, log (state, action) pairs.
    beta_cfg = BETAConfig(
        epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32,
    )

    from attacks.beta import pgd_attack

    pairs: list[dict] = []
    n_window_attacks = 0
    for trial_i, (w_idx, _argmax_target_unused) in enumerate(detected_train_windows):
        X = flat_xs[w_idx:w_idx + 1]

        for target_idx in target_rotation:
            # 1. BETA selection + prune.
            try:
                candidates = select_candidate_nodes(surrogate_query, X, target_idx, k=beta_cfg.candidate_k)
                with torch.no_grad():
                    _ = surrogate_query(X)  # populate model.learned_graph
                edge_index = model.learned_graph.detach()
                V_bar = prune_to_budget(candidates, edge_index, n_sensors, args.budget)
            except Exception:
                continue

            # 2. PGD.
            with torch.no_grad():
                clean_target = surrogate_query(X)[0, target_idx].item()
            try:
                X_pert, _ = pgd_attack(
                    surrogate_query, X, V_bar, target_idx,
                    torch.tensor([clean_target]), beta_cfg,
                )
            except Exception:
                continue

            delta_per_sensor = (X_pert[0] - X[0]).cpu()  # (N, W)

            # 3. Replay as MDP actions.
            env.reset(X.clone(), target_idx, budget=args.budget)
            for sensor_in_v_bar in V_bar.tolist():
                sensor_in_v_bar = int(sensor_in_v_bar)
                if sensor_in_v_bar == target_idx:
                    continue  # never attack the target itself
                cat_action = sensor_in_v_bar if sensor_in_v_bar < target_idx else sensor_in_v_bar - 1
                delta_for_sensor = delta_per_sensor[sensor_in_v_bar].numpy()
                state_before = env._encode_state()
                env.step((cat_action, delta_for_sensor))
                pairs.append({
                    "window_idx": int(w_idx),
                    "target_idx": int(target_idx),
                    "state": {
                        "current_window": state_before["current_window"].clone(),
                        "target_score_history": state_before["target_score_history"].clone(),
                        "target_idx": state_before["target_idx"],
                        "budget_remaining": state_before["budget_remaining"],
                        "step": state_before["step"],
                        "clean_target_score": state_before["clean_target_score"],
                    },
                    "action": {
                        "categorical": int(cat_action),
                        "delta": torch.from_numpy(delta_for_sensor).float(),
                    },
                })
            n_window_attacks += 1

        if (trial_i + 1) % 100 == 0:
            print(f"  window {trial_i+1}/{len(detected_train_windows)}  attacks={n_window_attacks}  pairs={len(pairs)}")

    print(f"[gen] generated {len(pairs)} pairs from {n_window_attacks} (window, target) attacks")
    if len(pairs) > args.max_pairs:
        rng = random.Random(args.seed + 1)  # different seed than window subsample
        pairs = rng.sample(pairs, args.max_pairs)
        print(f"[subsample] capped pairs at {args.max_pairs}")

    out_path = args.out or f"bc_dataset_gdn_{args.dataset}_seed{args.seed}.pt"
    if not Path(out_path).is_absolute():
        out_path = str(PROJECT_ROOT / out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, out_path)
    print(f"[write] {len(pairs)} BC pairs → {out_path}")


if __name__ == "__main__":
    main()
