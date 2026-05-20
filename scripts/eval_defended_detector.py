"""
Defended-detector head-to-head: PPO vs BETA when test inputs are pre-clipped to [0,1]
(deployment-realistic defense that removes the OOD-clamp exploit BETA leverages).

Key differences vs eval_ppo_policy.py:
  - Test inputs (flat_xs) clipped to [0,1] at load time.
  - Validation residuals also computed on clipped val inputs (to match the deployed
    detector's distribution).
  - Threshold recalibrated on clipped val scores (typically much lower than unclipped).
  - Detection is on clipped distribution → fires on real anomalies, not OOD scale violations.

Predicted outcome under this defense:
  - BETA loses its OOD-clamp shortcut (was 1000× ε on out-of-range cells).
  - Both attacks operate within true ε=0.1 budget in [0,1] feature space.
  - Whether either achieves substantial degradation is the empirical question.

Per chk_01KQY51RMTXKK4Q06W0Z2SVKE9 option B (highest-value paper experiment).
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
    get_anomaly_ranges_ds, split_ranges,
)
from attack_env.mdp import AdaptiveAttackEnv, MDPConfig  # noqa: E402
from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states  # noqa: E402
from attacks.beta import BETAAttack, BETAConfig  # noqa: E402


def setup_gdn_victim(seed: int = 0, dataset: str = "wadi"):
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
            "slide_stride": 10, "comment": "defended_eval", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "defended_eval", "seed": seed,
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
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--n-rollouts-per-trial", type=int, default=3)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--threshold-percentile", type=float, default=99.5,
                        help="Percentile of val score distribution to use as threshold")
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    parser.add_argument("--out", default="reports/eval_defended_seed0.json")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Perturbation budget for both BETA and PPO eval (default 0.1).")
    parser.add_argument("--undefended", action="store_true",
                        help="If set, bypass the input-clip defense (no clamp). Used for "
                        "Brain dec_01KRCF1NBGGZC5F40F99Q71MJP undefended-baseline measurement.")
    args = parser.parse_args()

    bc_ckpt_path = PROJECT_ROOT / args.ppo_checkpoint if not Path(args.ppo_checkpoint).is_absolute() else Path(args.ppo_checkpoint)
    ckpt = torch.load(bc_ckpt_path, weights_only=False)
    cfg = PolicyConfig(**ckpt["config"])
    policy = AdaptiveAttackPolicy(cfg)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"[load] policy from {bc_ckpt_path}")

    mode = "UNDEFENDED" if args.undefended else "DEFENDED"
    print(f"[setup] loading GDN-{args.dataset} seed {args.seed} ({mode}: input clip [0,1] {'BYPASSED' if args.undefended else 'ACTIVE'})")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    # _clip is identity in undefended mode, clamp(0,1) in defended mode.
    if args.undefended:
        _clip = lambda x: x
    else:
        _clip = lambda x: x.clamp(0.0, 1.0)

    from scipy.stats import iqr as scipy_iqr

    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            x_clip = _clip(x.float())
            y_clip = _clip(y.float())
            f = model(x_clip, ei.float())
            val_deltas.append((f - y_clip).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        # In defended mode: clip inputs to [0,1]. In undefended mode: pass through.
        X_clip = _clip(X)
        forecast = model(X_clip, edge_index_template)
        target_step = X_clip[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    n_oor = ((flat_xs < 0) | (flat_xs > 1)).sum().item()
    flat_xs = _clip(flat_xs)
    n_test, n_sensors, W = flat_xs.shape
    print(f"[{mode.lower()}-clip] {'would-clip' if args.undefended else 'clipped'} {n_oor} out-of-[0,1] cells in test inputs")

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
        x_clip = _clip(x.float())
        y_clip = _clip(y.float())
        with torch.no_grad():
            f = model(x_clip, ei.float())
        s = (f - y_clip).abs() - median
        s = s / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    threshold = float(np.percentile(val_scores_arr, args.threshold_percentile))
    print(f"[threshold] defended val score percentile {args.threshold_percentile} = {threshold:.4f}  "
          f"(max={val_scores_arr.max():.4f})")

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
                       if threshold_crossings[s] > 0]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:args.n_targets]]
    print(f"[holdout] detected: {len(detected_in_holdout)}; rotation: {target_rotation}")

    if not detected_in_holdout or not target_rotation:
        print("[result] DEFENDED: 0 detected windows or no targets — defense eliminates attack surface")
        result = {"defended": True, "n_detected_holdout": len(detected_in_holdout),
                  "target_rotation": target_rotation, "ppo_degradation_mean": 0.0,
                  "beta_degradation_mean": 0.0, "note": "no detection under defense"}
        out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        return

    rng = random.Random(args.seed)
    pairs = [(w, t) for w in detected_in_holdout for t in target_rotation]
    rng.shuffle(pairs)
    sample = pairs[:args.n_trials]

    import pandas as pd
    train_df = pd.read_csv(PROJECT_ROOT / f"repos/GDN/data/{args.dataset}/train.csv", index_col=0)
    train_x = torch.from_numpy(train_df.drop(columns=["attack"]).values).float()
    nominal = torch.zeros(n_sensors, 1024)
    for i in range(n_sensors):
        idx = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(args.seed))[:1024]
        nominal[i] = train_x[idx, i]

    mdp_cfg = MDPConfig(
        epsilon_pgd=args.epsilon, target_score_history_len=cfg.target_score_history_len,
        epsilon_reward_floor=0.1,
        lambda_1_sparsity=0.0, lambda_2_stealth=0.0, lambda_3_physics=0.0,
        per_window_budget_default=args.budget,
    )
    env = AdaptiveAttackEnv(surrogate_query, nominal, mdp_cfg)

    beta_cfg = BETAConfig(epsilon=args.epsilon, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    beta_attack = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                             num_nodes=n_sensors, config=beta_cfg)

    print(f"[eval] {len(sample)} trials × {args.n_rollouts_per_trial} rollouts under DEFENSE")
    ppo_degradations = []
    beta_degradations = []
    ppo_flips = []  # 1 if attack drove score below threshold; 0 otherwise
    beta_flips = []
    eps = args.epsilon
    for trial_i, (w_idx, target_idx) in enumerate(sample):
        X = flat_xs[w_idx:w_idx + 1].clone()
        with torch.no_grad():
            clean_score = surrogate_query(X)[0, target_idx].item()

        rollout_degradations = []
        rollout_flips = []
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
            # Flip = attack drove target score below threshold (assuming clean was above)
            rollout_flips.append(1.0 if (clean_score > threshold and attacked_score < threshold) else 0.0)
        ppo_degradations.append(float(np.mean(rollout_degradations)))
        ppo_flips.append(float(np.mean(rollout_flips)))

        try:
            X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
            with torch.no_grad():
                attacked_score = surrogate_query(X_pert)[0, target_idx].item()
            beta_degradations.append(clean_score - attacked_score)
            beta_flips.append(1.0 if (clean_score > threshold and attacked_score < threshold) else 0.0)
        except Exception:
            beta_degradations.append(0.0)
            beta_flips.append(0.0)

        if (trial_i + 1) % 10 == 0:
            print(f"  trial {trial_i+1}/{len(sample)}  PPO={np.mean(ppo_degradations):+.4f}  "
                  f"BETA={np.mean(beta_degradations):+.4f}")

    ppo_mean = float(np.mean(ppo_degradations))
    ppo_std = float(np.std(ppo_degradations))
    beta_mean = float(np.mean(beta_degradations))
    beta_std = float(np.std(beta_degradations))

    result = {
        "defended": (not args.undefended),
        "ppo_checkpoint": str(bc_ckpt_path),
        "seed": args.seed,
        "n_trials": len(sample),
        "n_rollouts_per_trial": args.n_rollouts_per_trial,
        "budget": args.budget,
        "target_rotation": target_rotation,
        "n_detected_holdout": len(detected_in_holdout),
        "threshold": threshold,
        "n_oor_clipped": n_oor,
        "ppo_degradation_mean": ppo_mean,
        "ppo_degradation_std": ppo_std,
        "beta_degradation_mean": beta_mean,
        "beta_degradation_std": beta_std,
        "ratio_ppo_over_beta": ppo_mean / beta_mean if beta_mean != 0 else 0.0,
        "ppo_flip_rate": float(np.mean(ppo_flips)),
        "beta_flip_rate": float(np.mean(beta_flips)),
        "n_eligible_for_flip": int(sum(1 for d in beta_degradations if d != 0.0 or beta_flips)),  # rough sanity
        # M5 (per-trial logging — Task 8 of mis P-reviewer2-response):
        "ppo_per_trial_degradations": [float(d) for d in ppo_degradations],
        "beta_per_trial_degradations": [float(d) for d in beta_degradations],
        "ppo_per_trial_flips": [float(f) for f in ppo_flips],
        "beta_per_trial_flips": [float(f) for f in beta_flips],
        # Per-window success rate: fraction of trials with positive degradation
        "ppo_success_rate": float(np.mean([d > 0 for d in ppo_degradations])),
        "beta_success_rate": float(np.mean([d > 0 for d in beta_degradations])),
        "trial_pairs": [{"window_idx": int(w), "target_idx": int(t)} for (w, t) in sample],
    }
    print(json.dumps(result, indent=2))
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
