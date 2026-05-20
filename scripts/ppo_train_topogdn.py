"""
P1.T5: PPO fine-tune of the BC-pretrained adaptive attack policy.

Loads BC checkpoint (cat-head 99.7% accuracy warm-start), continues training under
target-score-degradation reward in AdaptiveAttackEnv.

Action: hybrid (categorical sensor, per-channel sign vector).
  cat ~ Categorical(softmax(cat_logits))
  sign ~ Bernoulli(sigmoid(sign_logits))   per channel
  delta = (sign * 2 - 1) * epsilon_pgd     ∈ {-ε, +ε}^W

Reward: from env (target-score degradation, ε-floor normalized).

Loss: PPO clipped surrogate + value MSE - entropy bonus.

Per resolved chk_01KQVXF9GSP5HPXBV706JCW0NN: 200K-step initial budget on
GDN-WADI seed 0 sanity check before 5-seed × 2-detector sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states  # noqa: E402
from attack_env.mdp import AdaptiveAttackEnv, MDPConfig  # noqa: E402


@dataclass
class PPOConfig:
    n_total_steps: int = 200_000
    n_steps_per_rollout: int = 256
    n_epochs: int = 4
    minibatch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03


def setup_gdn_victim(seed: int = 0, dataset: str = "wadi"):
    """TopoGDN setup (variable name preserved for minimal diff vs ppo_train.py)."""
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
            "slide_stride": 10, "comment": "ppo_topogdn", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
            "use_tcn": True, "use_topo": True, "model": "GDN",
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "ppo_topogdn", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 256,
            "decay": 0.0, "val_ratio": 0.1, "topk": 30,
            "use_tcn": True, "use_topo": True, "model": "GDN",
        }
    env_config = {
        "save_path": f"{dataset}_seed{seed}", "dataset": dataset, "report": "best",
        "device": "cpu", "load_model_path": ckpt_path,
    }
    # IMPORTANT: clear cached `main` module from GDN repo if previously imported
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()
    return m


def _topogdn_forward(model, X, ei=None):
    """TopoGDN forward: model(X) returns (forecast, learned_graph). ei unused."""
    out = model(X)
    return out[0] if isinstance(out, tuple) else out


def make_env_for_seed(m, seed: int, budget: int, defended: bool = False, threshold_percentile: float = 100.0,
                      dataset: str = "wadi"):
    """Builds AdaptiveAttackEnv + a window-sampling iterator over detected pre-holdout windows.

    If defended=True: pre-clip val + test inputs to [0,1], recalibrate threshold on percentile.
    """
    from scipy.stats import iqr as scipy_iqr
    import pandas as pd

    from attack_env.test_split import (
        get_anomaly_ranges_ds, split_ranges,
    )

    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xf, yf = x.float(), y.float()
            if defended:
                xf = xf.clamp(0.0, 1.0)
                yf = yf.clamp(0.0, 1.0)
            f = _topogdn_forward(model, xf)
            val_deltas.append((f - yf).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        Xf = X.clamp(0.0, 1.0) if defended else X
        forecast = _topogdn_forward(model, Xf)
        target_step = Xf[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    if defended:
        n_oor = ((flat_xs < 0) | (flat_xs > 1)).sum().item()
        flat_xs = flat_xs.clamp(0.0, 1.0)
        print(f"[defended] clipped {n_oor} OOR cells")
    n_test, n_sensors, W = flat_xs.shape

    if dataset == "wadi":
        ranges = get_anomaly_ranges_ds("wadi", num_test_rows_ds=n_test + W - 1)
    else:
        ranges = get_anomaly_ranges_ds("swat")
    train_ranges, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start_target_row = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    pre_holdout_set = set(range(0, max(0, holdout_start_target_row - W + 1)))

    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xf, yf = x.float(), y.float()
        if defended:
            xf = xf.clamp(0.0, 1.0)
            yf = yf.clamp(0.0, 1.0)
        with torch.no_grad():
            f = _topogdn_forward(model, xf)
        s = (f - yf).abs() - median
        s = s / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    if threshold_percentile >= 100.0:
        threshold = float(val_scores_arr.max())
    else:
        threshold = float(np.percentile(val_scores_arr, threshold_percentile))

    detected_pool: list[tuple[int, int]] = []
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
                    detected_pool.append((global_i, int(max_idx[i].item())))
                threshold_crossings += (scores[i] > threshold).cpu().numpy().astype(np.int64)

    if dataset == "wadi":
        EXCLUDE = 106 if not defended else None
    else:
        EXCLUDE = None  # SWaT: don't pre-exclude any sensor
    crossing_counts = [(int(s), int(threshold_crossings[s])) for s in range(n_sensors)
                       if threshold_crossings[s] > 0 and s != EXCLUDE]
    crossing_counts.sort(key=lambda x: -x[1])
    target_rotation = [s for s, _ in crossing_counts[:10]]

    train_df = pd.read_csv(PROJECT_ROOT / f"repos/GDN/data/{dataset}/train.csv", index_col=0)
    train_x = torch.from_numpy(train_df.drop(columns=["attack"]).values).float()
    nominal = torch.zeros(n_sensors, 1024)
    for i in range(n_sensors):
        idx = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed))[:1024]
        nominal[i] = train_x[idx, i]

    cfg = MDPConfig(
        epsilon_pgd=0.1, target_score_history_len=5,
        epsilon_reward_floor=0.1,
        lambda_1_sparsity=0.0, lambda_2_stealth=0.0, lambda_3_physics=0.0,
        per_window_budget_default=budget, seed=seed,
    )
    env = AdaptiveAttackEnv(surrogate_query, nominal, cfg)

    print(f"[env] flat_xs={flat_xs.shape}, detected={len(detected_pool)}, targets={target_rotation}")
    return env, flat_xs, target_rotation, n_sensors, W


def sample_action(policy_out: dict, target_idx: int, n_sensors: int):
    """Sample full-dim cat (sensor index in [0, N-1] excluding target_idx) + per-channel sign."""
    cat_logits = policy_out["cat_logits"][0].clone()
    cat_logits[target_idx] = -float("inf")
    cat_dist = Categorical(logits=cat_logits)
    cat_action = cat_dist.sample()  # full-dim sensor index
    cat_log_prob = cat_dist.log_prob(cat_action)

    sign_logits = policy_out["sign_logits"][0]
    sign_dist = Bernoulli(logits=sign_logits)
    sign_sample = sign_dist.sample()
    sign_log_prob = sign_dist.log_prob(sign_sample).sum()
    delta_signed = (sign_sample * 2 - 1) * 0.1

    full_idx = int(cat_action.item())
    # env expects index in V\{u}; convert by removing target_idx slot
    cat_in_v_minus_u = full_idx if full_idx < target_idx else full_idx - 1
    value = policy_out["value"][0]
    return {
        "cat_idx_in_full": full_idx,
        "cat_idx_in_v_minus_u": cat_in_v_minus_u,
        "delta_signed": delta_signed.cpu().numpy(),
        "sign_sample": sign_sample.detach(),
        "log_prob": (cat_log_prob + sign_log_prob).detach(),
        "value": value.detach(),
    }


def evaluate_action_logprob(policy, batch_states, batch_cat_in_full, batch_sign_samples, batch_target_idx):
    """Re-evaluate log_prob at the FULL-dim cat index (matches sampling distribution)."""
    out = policy(batch_states)
    B = out["cat_logits"].shape[0]
    cat_logits = out["cat_logits"].clone()
    for b in range(B):
        cat_logits[b, batch_target_idx[b]] = -float("inf")
    cat_dist = Categorical(logits=cat_logits)
    cat_log_prob = cat_dist.log_prob(batch_cat_in_full)
    sign_dist = Bernoulli(logits=out["sign_logits"])
    sign_log_prob = sign_dist.log_prob(batch_sign_samples).sum(dim=-1)
    entropy = cat_dist.entropy() + sign_dist.entropy().sum(dim=-1)
    return cat_log_prob + sign_log_prob, out["value"], entropy


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", default="reports/bc_policy_gdn_wadi_seed0_v3.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-total-steps", type=int, default=200_000)
    parser.add_argument("--n-steps-per-rollout", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--out-checkpoint", default="reports/ppo_policy_gdn_wadi_seed0.pt")
    parser.add_argument("--out-report", default="reports/ppo_train_report.json")
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--defended", action="store_true",
                        help="Train PPO under input-clipping defense")
    parser.add_argument("--threshold-percentile", type=float, default=100.0)
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    args = parser.parse_args()

    cfg = PPOConfig(
        n_total_steps=args.n_total_steps,
        n_steps_per_rollout=args.n_steps_per_rollout,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, value_coef=args.value_coef,
        entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bc_ckpt_path = PROJECT_ROOT / args.bc_checkpoint if not Path(args.bc_checkpoint).is_absolute() else Path(args.bc_checkpoint)
    bc_ckpt = torch.load(bc_ckpt_path, weights_only=False)
    pcfg = PolicyConfig(**bc_ckpt["config"])
    policy = AdaptiveAttackPolicy(pcfg)
    policy.load_state_dict(bc_ckpt["policy_state_dict"])
    print(f"[load] BC warm-start: cat_acc={bc_ckpt.get('final_val_cat_acc'):.3f}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    print(f"[setup] loading GDN-{args.dataset} seed", args.seed)
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    env, flat_xs, target_rotation, n_sensors, W = make_env_for_seed(
        m, args.seed, args.budget, defended=args.defended,
        threshold_percentile=args.threshold_percentile,
        dataset=args.dataset,
    )

    rng = random.Random(args.seed)
    history = {"step": [], "mean_episode_return": [], "policy_loss": [],
               "value_loss": [], "entropy": [], "approx_kl": []}

    step = 0
    t0 = time.time()
    while step < cfg.n_total_steps:
        # Rollout
        states_buf, log_probs_buf, values_buf = [], [], []
        rewards_buf, dones_buf, target_idx_buf = [], [], []
        sign_samples_buf, cat_in_full_buf = [], []
        episode_returns = []
        ep_return = 0.0

        # episode init
        w_idx = rng.choice(range(flat_xs.shape[0] - 1))
        target = rng.choice(target_rotation)
        env.reset(flat_xs[w_idx:w_idx + 1].clone(), target, budget=args.budget)
        s = env._encode_state()

        rollout_step = 0
        while rollout_step < cfg.n_steps_per_rollout:
            batch = collate_states([s])
            policy.eval()
            with torch.no_grad():
                out = policy(batch)
            sa = sample_action(out, target, n_sensors)

            states_buf.append(s)
            log_probs_buf.append(sa["log_prob"].item())
            values_buf.append(sa["value"].item())
            target_idx_buf.append(target)
            sign_samples_buf.append(sa["sign_sample"])
            cat_in_full_buf.append(sa["cat_idx_in_full"])

            res = env.step((sa["cat_idx_in_v_minus_u"], sa["delta_signed"]))
            rewards_buf.append(res.reward)
            done = bool(res.done or res.truncated)
            dones_buf.append(1.0 if done else 0.0)
            ep_return += res.reward

            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                w_idx = rng.choice(range(flat_xs.shape[0] - 1))
                target = rng.choice(target_rotation)
                env.reset(flat_xs[w_idx:w_idx + 1].clone(), target, budget=args.budget)
                s = env._encode_state()
            else:
                s = res.state

            rollout_step += 1
            step += 1

        # Bootstrap final value
        with torch.no_grad():
            last_batch = collate_states([s])
            last_value = policy(last_batch)["value"][0].item()

        advantages, returns = compute_gae(
            np.array(rewards_buf, dtype=np.float32),
            np.array(values_buf, dtype=np.float32),
            np.array(dones_buf, dtype=np.float32),
            last_value, cfg.gamma, cfg.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_t = torch.from_numpy(returns)
        advantages_t = torch.from_numpy(advantages)
        old_log_probs_t = torch.tensor(log_probs_buf, dtype=torch.float32)
        sign_samples_t = torch.stack(sign_samples_buf, dim=0)
        cat_in_full_t = torch.tensor(cat_in_full_buf, dtype=torch.long)
        target_idx_t = torch.tensor(target_idx_buf, dtype=torch.long)

        # PPO update
        n_rollout = len(states_buf)
        idx_all = np.arange(n_rollout)
        approx_kls, p_losses, v_losses, ent_means = [], [], [], []
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx_all)
            for mb_start in range(0, n_rollout, cfg.minibatch_size):
                mb = idx_all[mb_start:mb_start + cfg.minibatch_size]
                if len(mb) == 0:
                    continue
                mb_states = collate_states([states_buf[i] for i in mb])
                mb_cat = cat_in_full_t[mb]
                mb_sign = sign_samples_t[mb]
                mb_target = target_idx_t[mb]
                mb_old_log = old_log_probs_t[mb]
                mb_adv = advantages_t[mb]
                mb_ret = returns_t[mb]

                policy.train()
                new_log_prob, value, entropy = evaluate_action_logprob(
                    policy, mb_states, mb_cat, mb_sign, mb_target
                )
                ratio = (new_log_prob - mb_old_log).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                approx_kl = (mb_old_log - new_log_prob).mean().item()
                approx_kls.append(approx_kl)
                p_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                ent_means.append(entropy_mean.item())

            if np.mean(approx_kls[-(n_rollout // cfg.minibatch_size + 1):]) > cfg.target_kl:
                break  # early stop epoch on KL

        mean_ep_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        history["step"].append(step)
        history["mean_episode_return"].append(mean_ep_return)
        history["policy_loss"].append(float(np.mean(p_losses)))
        history["value_loss"].append(float(np.mean(v_losses)))
        history["entropy"].append(float(np.mean(ent_means)))
        history["approx_kl"].append(float(np.mean(approx_kls)))

        elapsed = time.time() - t0
        print(f"  step {step}/{cfg.n_total_steps}  ep_ret={mean_ep_return:+.3f}  "
              f"p_loss={np.mean(p_losses):+.4f}  v_loss={np.mean(v_losses):.4f}  "
              f"ent={np.mean(ent_means):.3f}  kl={np.mean(approx_kls):.4f}  "
              f"elapsed={elapsed:.0f}s")

        if step % cfg.n_total_steps == 0 or step >= cfg.n_total_steps or step % args.checkpoint_every < cfg.n_steps_per_rollout:
            out_ckpt = PROJECT_ROOT / args.out_checkpoint if not Path(args.out_checkpoint).is_absolute() else Path(args.out_checkpoint)
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "config": vars(pcfg),
                "step": step,
                "history_tail": {k: v[-5:] for k, v in history.items()},
            }, out_ckpt)

    out_ckpt = PROJECT_ROOT / args.out_checkpoint if not Path(args.out_checkpoint).is_absolute() else Path(args.out_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "config": vars(pcfg),
        "step": step,
        "history": history,
    }, out_ckpt)
    print(f"[write] PPO policy checkpoint -> {out_ckpt}")

    out_report = PROJECT_ROOT / args.out_report if not Path(args.out_report).is_absolute() else Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps({
        "n_total_steps": cfg.n_total_steps,
        "history": history,
        "ppo_config": vars(cfg),
        "policy_config": vars(pcfg),
    }, indent=2, default=str))
    print(f"[write] PPO report -> {out_report}")


if __name__ == "__main__":
    main()
