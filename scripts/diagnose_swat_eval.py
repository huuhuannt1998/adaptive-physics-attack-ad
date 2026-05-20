"""
Diagnostic: re-run SWaT seed 0 defended eval but filter (window, target) pairs
to those where clean_target_score > threshold (i.e., genuinely-attackable pairs).

Hypothesis: matched eval samples target ∈ rotation, but rotation is built from
threshold-crossing-frequency (sensors that ARE threshold-crossers in SOME windows).
On a specific (window, target) pair, the target sensor's clean score may already be
below threshold, in which case "attack" has nothing to do and degradation can only
go one direction (down) or stay zero.

Filter eval to clean_target_score > threshold pairs and report:
  1. What fraction of original 30 (window, target) pairs are genuinely attackable
  2. PPO and BETA mean degradation on filtered (attackable-only) pairs
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

from attack_env.test_split import get_anomaly_ranges_ds, split_ranges  # noqa: E402
from attack_env.mdp import AdaptiveAttackEnv, MDPConfig  # noqa: E402
from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states  # noqa: E402
from attacks.beta import BETAAttack, BETAConfig  # noqa: E402


def setup_gdn_victim(seed: int, dataset: str = "swat"):
    repo = PROJECT_ROOT / "repos/GDN"
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
            "slide_stride": 10, "comment": "diag", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "diag", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 256,
            "decay": 0.0, "val_ratio": 0.1, "topk": 30,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="swat")
    parser.add_argument("--n-trials", type=int, default=100)  # bigger pool, then filter
    parser.add_argument("--n-rollouts-per-trial", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--out", default="reports/diagnose_swat_seed0.json")
    args = parser.parse_args()

    bc_ckpt_path = PROJECT_ROOT / args.ppo_checkpoint if not Path(args.ppo_checkpoint).is_absolute() else Path(args.ppo_checkpoint)
    ckpt = torch.load(bc_ckpt_path, weights_only=False)
    cfg = PolicyConfig(**ckpt["config"])
    policy = AdaptiveAttackPolicy(cfg)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    print(f"[setup] {args.dataset} seed {args.seed} (defended)")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xc, yc = x.float().clamp(0, 1), y.float().clamp(0, 1)
            f = model(xc, ei.float())
            val_deltas.append((f - yc).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        Xc = X.clamp(0, 1)
        forecast = model(Xc, edge_index_template)
        target_step = Xc[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0).clamp(0, 1)
    n_test, n_sensors, W = flat_xs.shape

    if args.dataset == "wadi":
        ranges = get_anomaly_ranges_ds("wadi", num_test_rows_ds=n_test + W - 1)
    else:
        ranges = get_anomaly_ranges_ds("swat")
    _, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start_target_row = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    holdout_pool = list(range(max(0, holdout_start_target_row - W + 1), n_test))

    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xc, yc = x.float().clamp(0, 1), y.float().clamp(0, 1)
        with torch.no_grad():
            f = model(xc, ei.float())
        s = (f - yc).abs() - median
        s = s / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold = float(np.percentile(val_scores_arr, args.threshold_percentile))
    print(f"[threshold] {threshold:.4f}")

    detected_in_holdout = []
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
                       if threshold_crossings[s] > 0]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:10]]

    rng = random.Random(args.seed)
    pairs = [(w, t) for w in detected_in_holdout for t in target_rotation]
    rng.shuffle(pairs)
    sample = pairs[:args.n_trials]
    print(f"[eval] sampling {len(sample)} (window, target) pairs from rotation {target_rotation}")

    # First pass: classify each pair by clean_target_score vs threshold
    clean_target_scores = []
    for w_idx, target_idx in sample:
        with torch.no_grad():
            cs = surrogate_query(flat_xs[w_idx:w_idx + 1])[0, target_idx].item()
        clean_target_scores.append((w_idx, target_idx, cs))

    above_threshold = [(w, t, c) for w, t, c in clean_target_scores if c > threshold]
    below_threshold = [(w, t, c) for w, t, c in clean_target_scores if c <= threshold]
    print(f"[diagnostic] {len(above_threshold)}/{len(sample)} pairs have clean_target > threshold ({threshold:.3f})")
    print(f"[diagnostic]   {len(above_threshold)} attackable; {len(below_threshold)} non-attackable")

    if not above_threshold:
        print("[result] NO ATTACKABLE PAIRS in sampled rotation — eval methodology issue confirmed")
        out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.write_text(json.dumps({
            "no_attackable_pairs": True,
            "n_sample": len(sample),
            "n_above_threshold": 0,
            "threshold": threshold,
            "target_rotation": target_rotation,
        }, indent=2))
        return

    # Second pass: run PPO + BETA on attackable pairs only
    import pandas as pd
    train_df = pd.read_csv(PROJECT_ROOT / f"repos/GDN/data/{args.dataset}/train.csv", index_col=0)
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

    ppo_attackable, beta_attackable = [], []
    ppo_flips, beta_flips = [], []
    eps = 0.1
    for trial_i, (w_idx, target_idx, clean_score) in enumerate(above_threshold[:30]):
        X = flat_xs[w_idx:w_idx + 1].clone()

        # PPO with stochastic sampling, 3 rollouts averaged
        rollout_d = []
        rollout_f = []
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
            rollout_d.append(clean_score - attacked_score)
            rollout_f.append(1.0 if attacked_score < threshold else 0.0)
        ppo_attackable.append(float(np.mean(rollout_d)))
        ppo_flips.append(float(np.mean(rollout_f)))

        try:
            X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            beta_attackable.append(clean_score - attacked_score)
            beta_flips.append(1.0 if attacked_score < threshold else 0.0)
        except Exception:
            beta_attackable.append(0.0)
            beta_flips.append(0.0)

    result = {
        "diagnostic": True,
        "dataset": args.dataset,
        "seed": args.seed,
        "n_sampled": len(sample),
        "n_attackable": len(above_threshold),
        "fraction_attackable": len(above_threshold) / len(sample),
        "threshold": threshold,
        "target_rotation": target_rotation,
        "n_evaluated": min(30, len(above_threshold)),
        "ppo_mean_attackable": float(np.mean(ppo_attackable)),
        "ppo_std_attackable": float(np.std(ppo_attackable)),
        "beta_mean_attackable": float(np.mean(beta_attackable)),
        "beta_std_attackable": float(np.std(beta_attackable)),
        "ppo_flip_rate": float(np.mean(ppo_flips)),
        "beta_flip_rate": float(np.mean(beta_flips)),
    }
    print(json.dumps(result, indent=2))
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
