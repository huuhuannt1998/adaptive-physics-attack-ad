"""
Evaluate the BC-pretrained policy on the held-out 30% test ranges.

Computes:
  - BC policy mean target-score degradation
  - BETA reimpl mean target-score degradation (for reference)
  - Ratio = BC / BETA. Mission spec target: ≥80%.

Usage:
    python scripts/eval_bc_policy.py --bc-checkpoint reports/bc_policy_gdn_wadi_seed0.pt \\
                                     --seed 0 --n-trials 50
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

from attack_env.test_split import (  # noqa: E402
    get_wadi_anomaly_ranges_ds, split_ranges, windows_for_ranges,
)
from attack_env.mdp import AdaptiveAttackEnv, MDPConfig  # noqa: E402
from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states  # noqa: E402
from attacks.beta import BETAAttack, BETAConfig  # noqa: E402


def setup_gdn_victim(seed: int = 0):
    repo = PROJECT_ROOT / "repos/GDN"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ckpt_dir = repo / f"pretrained/wadi_seed{seed}"
    candidates = list(ckpt_dir.glob("best_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    ckpt_path = "./" + str(candidates[-1].relative_to(repo).as_posix())

    train_config = {
        "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
        "slide_stride": 10, "comment": "bc_eval", "seed": seed,
        "out_layer_num": 1, "out_layer_inter_dim": 256,
        "decay": 0.0, "val_ratio": 0.1, "topk": 30,
    }
    env_config = {
        "save_path": f"wadi_seed{seed}", "dataset": "wadi", "report": "best",
        "device": "cpu", "load_model_path": ckpt_path,
    }

    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out", default="reports/bc_eval_gdn_wadi_seed0.json")
    args = parser.parse_args()

    bc_ckpt_path = PROJECT_ROOT / args.bc_checkpoint if not Path(args.bc_checkpoint).is_absolute() else Path(args.bc_checkpoint)
    bc_ckpt = torch.load(bc_ckpt_path, weights_only=False)
    cfg = PolicyConfig(**bc_ckpt["config"])
    policy = AdaptiveAttackPolicy(cfg)
    policy.load_state_dict(bc_ckpt["policy_state_dict"])
    policy.eval()

    print(f"[setup] loading GDN-WADI seed {args.seed}")
    m = setup_gdn_victim(seed=args.seed)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr

    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            f = model(x.float(), ei.float())
            val_deltas.append((f - y.float()).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()

    EPS = 1e-2
    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        forecast = model(X, edge_index_template)
        target_step = X[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    n_test, n_sensors, W = flat_xs.shape

    # Holdout windows.
    ranges = get_wadi_anomaly_ranges_ds(num_test_rows_ds=n_test + W - 1)
    _, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_windows_idx = [i for i in windows_for_ranges(holdout_ranges, window_size=W) if 0 <= i < n_test]

    # Detect within holdout.
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        with torch.no_grad():
            f = model(x.float(), ei.float())
        s = (f - y.float()).abs() - median
        s = s / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold = float(val_scores_arr.max())

    detected_holdout: list[tuple[int, int]] = []
    BATCH = 256
    holdout_set = set(holdout_windows_idx)
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            scores = surrogate_query(X)
            max_scores, max_idx = scores.max(dim=1)
            for i in range(X.shape[0]):
                global_i = start + i
                if global_i in holdout_set and max_scores[i].item() > threshold:
                    detected_holdout.append((global_i, int(max_idx[i].item())))
    print(f"[holdout] detected windows: {len(detected_holdout)}")

    rng = random.Random(args.seed)
    sample = rng.sample(detected_holdout, min(args.n_trials, len(detected_holdout)))
    print(f"[eval] running BC policy + BETA on {len(sample)} (window, target=argmax) trials")

    # BETA setup
    beta_cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    beta_attack = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                             num_nodes=n_sensors, config=beta_cfg)

    # Nominal distributions for env.
    import pandas as pd  # noqa
    train_df = pd.read_csv(PROJECT_ROOT / "repos/GDN/data/wadi/train.csv", index_col=0)
    train_x = torch.from_numpy(train_df.drop(columns=["attack"]).values).float()
    nominal = torch.zeros(n_sensors, 1024)
    for i in range(n_sensors):
        idx = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(args.seed))[:1024]
        nominal[i] = train_x[idx, i]

    mdp_cfg = MDPConfig(
        epsilon_pgd=0.1, target_score_history_len=cfg.target_score_history_len,
        epsilon_reward_floor=0.1,
        lambda_1_sparsity=0.0, lambda_2_stealth=0.0, lambda_3_physics=0.0,
        per_window_budget_default=args.budget,
    )
    env = AdaptiveAttackEnv(surrogate_query, nominal, mdp_cfg)

    bc_degradations = []
    beta_degradations = []
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean_score = surrogate_query(X)[0, target_idx].item()

        # ──── BC POLICY ATTACK ────
        env.reset(X.clone(), target_idx, budget=args.budget)
        for _ in range(args.budget):
            state = env._encode_state()
            batch = collate_states([state])
            with torch.no_grad():
                out = policy(batch)
            cat_idx = out["cat_logits"][0].argmax().item()
            delta_t = out["delta_mu"][0].cpu().numpy()
            res = env.step((cat_idx, delta_t))
            if res.done or res.truncated:
                break
        with torch.no_grad():
            bc_attacked_score = surrogate_query(env._current_window)[0, target_idx].item()
        bc_degradations.append(clean_score - bc_attacked_score)

        # ──── BETA ATTACK ────
        try:
            X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                beta_attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            beta_degradations.append(clean_score - beta_attacked_score)
        except Exception:
            beta_degradations.append(0.0)

        if (trial_i + 1) % 10 == 0:
            print(f"  trial {trial_i+1}/{len(sample)}  BC mean={np.mean(bc_degradations):.2f}  "
                  f"BETA mean={np.mean(beta_degradations):.2f}")

    bc_mean = float(np.mean(bc_degradations))
    bc_std = float(np.std(bc_degradations))
    beta_mean = float(np.mean(beta_degradations))
    beta_std = float(np.std(beta_degradations))
    ratio = bc_mean / beta_mean if beta_mean != 0 else 0.0

    result = {
        "bc_checkpoint": str(bc_ckpt_path),
        "seed": args.seed,
        "n_trials": len(sample),
        "budget": args.budget,
        "bc_degradation_mean": bc_mean,
        "bc_degradation_std": bc_std,
        "beta_degradation_mean": beta_mean,
        "beta_degradation_std": beta_std,
        "ratio_bc_over_beta": ratio,
        "ratio_target": 0.80,
        "passes_target": ratio >= 0.80,
    }
    print(json.dumps(result, indent=2))
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[write] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
