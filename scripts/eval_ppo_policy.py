"""
Evaluate PPO-trained policy with the *same target-selection and action-sampling regime*
it was trained under, plus a head-to-head BETA reference on identical (window, target) pairs.

Differences vs eval_bc_policy.py:
  1. Filters eval trials to target ∈ training rotation (excludes sensor 106 etc.).
  2. Stochastic action sampling: cat ~ Categorical(masked logits); sign ~ Bernoulli.
  3. Reports per-(window, target) BETA degradation alongside PPO degradation.
  4. Multiple rollouts per (window, target) to average over policy stochasticity.
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
from torch.distributions import Categorical, Bernoulli

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attack_env.test_split import (  # noqa: E402
    get_wadi_anomaly_ranges_ds, split_ranges,
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
        "slide_stride": 10, "comment": "ppo_eval", "seed": seed,
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
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--n-rollouts-per-trial", type=int, default=3)
    parser.add_argument("--exclude-sensor", type=int, default=106)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--out", default="reports/ppo_eval_matched_seed0.json")
    args = parser.parse_args()

    bc_ckpt_path = PROJECT_ROOT / args.ppo_checkpoint if not Path(args.ppo_checkpoint).is_absolute() else Path(args.ppo_checkpoint)
    ckpt = torch.load(bc_ckpt_path, weights_only=False)
    cfg = PolicyConfig(**ckpt["config"])
    policy = AdaptiveAttackPolicy(cfg)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"[load] policy from {bc_ckpt_path}")

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

    ranges = get_wadi_anomaly_ranges_ds(num_test_rows_ds=n_test + W - 1)
    _, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start_target_row = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    holdout_pool = list(range(max(0, holdout_start_target_row - W + 1), n_test))

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

    detected_in_holdout: list[int] = []
    threshold_crossings = np.zeros(n_sensors, dtype=np.int64)
    BATCH = 256
    holdout_set = set(holdout_pool)
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            scores = surrogate_query(X)
            for i in range(X.shape[0]):
                global_i = start + i
                if global_i not in holdout_set:
                    continue
                max_score, _ = scores[i].max(dim=0)
                if max_score.item() > threshold:
                    detected_in_holdout.append(global_i)
                threshold_crossings += (scores[i] > threshold).cpu().numpy().astype(np.int64)

    crossing_counts = [(int(s), int(threshold_crossings[s])) for s in range(n_sensors)
                       if threshold_crossings[s] > 0 and s != args.exclude_sensor]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:args.n_targets]]
    print(f"[holdout] detected: {len(detected_in_holdout)}; target rotation (matched to PPO training): {target_rotation}")

    rng = random.Random(args.seed)
    pairs = [(w, t) for w in detected_in_holdout for t in target_rotation]
    rng.shuffle(pairs)
    sample = pairs[:args.n_trials]
    print(f"[eval] running {len(sample)} (window, target) trials, {args.n_rollouts_per_trial} stochastic rollouts each")

    import pandas as pd
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

    beta_cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    beta_attack = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                             num_nodes=n_sensors, config=beta_cfg)

    ppo_degradations = []
    beta_degradations = []
    eps = 0.1
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean_score = surrogate_query(X)[0, target_idx].item()

        # PPO policy: stochastic sampling, average over n_rollouts
        rollout_degradations = []
        for r in range(args.n_rollouts_per_trial):
            torch.manual_seed(args.seed * 1000 + trial_i * 10 + r)
            env.reset(X.clone(), target_idx, budget=args.budget)
            for _ in range(args.budget):
                state = env._encode_state()
                batch = collate_states([state])
                with torch.no_grad():
                    out = policy(batch)
                cat_logits = out["cat_logits"][0].clone()
                cat_logits[target_idx] = -float("inf")
                cat_dist = Categorical(logits=cat_logits)
                full_idx = int(cat_dist.sample().item())
                cat_in_v_minus_u = full_idx if full_idx < target_idx else full_idx - 1

                sign_logits = out["sign_logits"][0]
                sign_dist = Bernoulli(logits=sign_logits)
                sign_sample = sign_dist.sample()
                delta_signed = ((sign_sample * 2 - 1) * eps).cpu().numpy()

                res = env.step((cat_in_v_minus_u, delta_signed))
                if res.done or res.truncated:
                    break
            with torch.no_grad():
                attacked_score = surrogate_query(env._current_window)[0, target_idx].item()
            rollout_degradations.append(clean_score - attacked_score)
        ppo_degradations.append(float(np.mean(rollout_degradations)))

        # BETA reference on the same (window, target)
        try:
            X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            beta_degradations.append(clean_score - attacked_score)
        except Exception:
            beta_degradations.append(0.0)

        if (trial_i + 1) % 10 == 0:
            print(f"  trial {trial_i+1}/{len(sample)}  PPO_mean={np.mean(ppo_degradations):.3f}  "
                  f"BETA_mean={np.mean(beta_degradations):.3f}")

    ppo_mean = float(np.mean(ppo_degradations))
    ppo_std = float(np.std(ppo_degradations))
    beta_mean = float(np.mean(beta_degradations))
    beta_std = float(np.std(beta_degradations))
    ratio = ppo_mean / beta_mean if beta_mean != 0 else 0.0

    result = {
        "ppo_checkpoint": str(bc_ckpt_path),
        "seed": args.seed,
        "n_trials": len(sample),
        "n_rollouts_per_trial": args.n_rollouts_per_trial,
        "budget": args.budget,
        "target_rotation": target_rotation,
        "exclude_sensor": args.exclude_sensor,
        "ppo_degradation_mean": ppo_mean,
        "ppo_degradation_std": ppo_std,
        "beta_degradation_mean": beta_mean,
        "beta_degradation_std": beta_std,
        "ratio_ppo_over_beta": ratio,
    }
    print(json.dumps(result, indent=2))
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[write] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
