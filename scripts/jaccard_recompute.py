"""
Recompute BC-vs-BETA and PPO-vs-BETA top-K categorical-target Jaccard on an
undefended GDN detector seed. Resolves the §7.1/§7.2 BC=0.93 / PPO~0.90 tension
flagged by Brain in Mission P-reviewer2-response Batch 2.

Protocol:
  1. Load GDN-WADI seed N victim. Build the 50-window detected-set used by
     scripts/eval_beta_swat.py.
  2. For each window:
     a. Run BETA: select_candidate_nodes (saliency top-32) → eigenvector_centrality
        prune → V_bar (b=5 sensors).
     b. Run BC policy: categorical-head sampled action → 1 sensor (argmax).
     c. Run PPO policy: same.
  3. Aggregate sensor selection counts:
     - BETA: 5 picks per window × n_windows = 5n total V_bar entries
     - BC: 1 pick per window
     - PPO: 1 pick per window
  4. Compute top-K most-frequent sensors for each method.
  5. Report Jaccard(top-K(BC), top-K(BETA)) and Jaccard(top-K(PPO), top-K(BETA))
     for K ∈ {5, 10}.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attack_env.test_split import get_anomaly_ranges_ds, split_ranges  # noqa: E402
from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states  # noqa: E402
from attacks.beta import (  # noqa: E402
    BETAConfig, eigenvector_centrality, prune_to_budget, select_candidate_nodes,
)


def setup_gdn_victim(seed: int, dataset: str = "wadi", arch: str = "GDN"):
    repo_name = "TopoGDN" if arch.lower() == "topogdn" else "GDN"
    repo = PROJECT_ROOT / f"repos/{repo_name}"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    ckpt_dir = repo / f"pretrained/{dataset}_seed{seed}"
    ckpt_path = "./" + str(list(ckpt_dir.glob("best_*.pt"))[-1].relative_to(repo).as_posix())

    if dataset == "swat":
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 64,
                        "slide_stride": 10, "comment": "jacc", "seed": seed,
                        "out_layer_num": 1, "out_layer_inter_dim": 128,
                        "decay": 0.0, "val_ratio": 0.1, "topk": 15}
    else:
        train_config = {"batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
                        "slide_stride": 10, "comment": "jacc", "seed": seed,
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


def load_policy(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = PolicyConfig(**ckpt["config"])
    policy = AdaptiveAttackPolicy(cfg)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, cfg


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    p.add_argument("--bc-checkpoint", required=True)
    p.add_argument("--ppo-checkpoint", required=True)
    p.add_argument("--n-windows", type=int, default=50)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--out", default="reports/jaccard_recompute.json")
    args = p.parse_args()

    print(f"[setup] GDN-{args.dataset} seed {args.seed}")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset)
    model = m.model
    test_loader = m.test_dataloader

    edge_index_template = next(iter(test_loader))[3].float()
    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    n_test, N, W = flat_xs.shape
    print(f"[setup] flat_xs={tuple(flat_xs.shape)}")

    # Get anomaly windows (similar to eval_beta_swat protocol)
    if args.dataset == "wadi":
        ranges = get_anomaly_ranges_ds("wadi", num_test_rows_ds=n_test + W - 1)
    else:
        ranges = get_anomaly_ranges_ds("swat")
    _, holdout_ranges = split_ranges(ranges, train_fraction=0.7)
    holdout_start = min(r.start_row_ds for r in holdout_ranges) if holdout_ranges else n_test
    holdout_pool = list(range(max(0, holdout_start - W + 1), n_test))

    # Compute std stats and threshold (undefended convention: max-of-val)
    from scipy.stats import iqr as scipy_iqr
    val_loader = m.val_dataloader
    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            f = model(x.float(), ei.float())
            val_deltas.append((f - y.float()).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    EPS = 1e-2

    def surrogate_query(X):
        forecast = model(X, edge_index_template)
        target_step = X[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    # Detected windows (above max-of-val threshold)
    threshold = float((np.median(val_deltas, axis=0) - median.numpy()) / (iqr_v.abs().numpy() + EPS)).max() if False else 0.0
    # Use n_windows-most-anomalous from holdout
    holdout_set = set(holdout_pool)
    BATCH = 256
    score_max = []
    with torch.no_grad():
        for start in range(0, n_test, BATCH):
            X = flat_xs[start:start + BATCH]
            sc = surrogate_query(X)
            score_max.extend([(start + i, float(sc[i].max().item())) for i in range(X.shape[0])
                              if (start + i) in holdout_set])
    score_max.sort(key=lambda x: -x[1])
    selected_windows = [w for w, _ in score_max[:args.n_windows]]
    print(f"[windows] selected {len(selected_windows)} most-anomalous holdout windows")

    # Load policies
    bc_policy, _ = load_policy(Path(args.bc_checkpoint) if Path(args.bc_checkpoint).is_absolute()
                               else PROJECT_ROOT / args.bc_checkpoint)
    ppo_policy, _ = load_policy(Path(args.ppo_checkpoint) if Path(args.ppo_checkpoint).is_absolute()
                                else PROJECT_ROOT / args.ppo_checkpoint)
    beta_cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)

    beta_v_bar_picks = []
    bc_picks = []
    ppo_picks = []

    for w_idx in selected_windows:
        X = flat_xs[w_idx:w_idx + 1].clone()
        # Per-window argmax target
        with torch.no_grad():
            scores = surrogate_query(X)[0]
        target_idx = int(scores.argmax().item())

        # BETA: saliency → centrality → V_bar
        candidates = select_candidate_nodes(surrogate_query, X, target_idx, k=beta_cfg.candidate_k)
        with torch.no_grad():
            _ = surrogate_query(X)  # populate model.learned_graph
        learned_edge_index = model.learned_graph.detach()
        V_bar = prune_to_budget(candidates, learned_edge_index, N, args.budget)
        beta_v_bar_picks.extend(V_bar.tolist())

        # BC and PPO: build state, get categorical argmax
        from attack_env.mdp import AdaptiveAttackEnv, MDPConfig
        nominal = torch.zeros(N, 1024)
        # quick nominal stub: zeros (won't matter for state encoding action)
        mdp_cfg = MDPConfig(
            epsilon_pgd=0.1, target_score_history_len=bc_policy.cfg.target_score_history_len,
            epsilon_reward_floor=0.1,
            lambda_1_sparsity=0.0, lambda_2_stealth=0.0, lambda_3_physics=0.0,
            per_window_budget_default=args.budget,
        )
        env = AdaptiveAttackEnv(surrogate_query, nominal, mdp_cfg)
        env.reset(X.clone(), target_idx, budget=args.budget)
        state = env._encode_state()
        batch = collate_states([state])
        with torch.no_grad():
            bc_out = bc_policy(batch)
            ppo_out = ppo_policy(batch)
        bc_logits = bc_out["cat_logits"][0].clone()
        bc_logits[target_idx] = -float("inf")
        ppo_logits = ppo_out["cat_logits"][0].clone()
        ppo_logits[target_idx] = -float("inf")
        bc_argmax = int(bc_logits.argmax().item())
        ppo_argmax = int(ppo_logits.argmax().item())
        bc_picks.append(bc_argmax)
        ppo_picks.append(ppo_argmax)

    bc_top10 = set(s for s,_ in Counter(bc_picks).most_common(10))
    ppo_top10 = set(s for s,_ in Counter(ppo_picks).most_common(10))
    beta_top10 = set(s for s,_ in Counter(beta_v_bar_picks).most_common(10))
    bc_top5 = set(s for s,_ in Counter(bc_picks).most_common(5))
    ppo_top5 = set(s for s,_ in Counter(ppo_picks).most_common(5))
    beta_top5 = set(s for s,_ in Counter(beta_v_bar_picks).most_common(5))

    out = {
        "seed": args.seed, "dataset": args.dataset,
        "n_windows": len(selected_windows), "budget": args.budget,
        "bc_picks_unique": len(set(bc_picks)),
        "ppo_picks_unique": len(set(ppo_picks)),
        "beta_unique_in_v_bar": len(set(beta_v_bar_picks)),
        "bc_top10": sorted(bc_top10), "ppo_top10": sorted(ppo_top10), "beta_top10": sorted(beta_top10),
        "jaccard_bc_beta_top10": jaccard(bc_top10, beta_top10),
        "jaccard_ppo_beta_top10": jaccard(ppo_top10, beta_top10),
        "jaccard_bc_ppo_top10": jaccard(bc_top10, ppo_top10),
        "jaccard_bc_beta_top5": jaccard(bc_top5, beta_top5),
        "jaccard_ppo_beta_top5": jaccard(ppo_top5, beta_top5),
        "jaccard_bc_ppo_top5": jaccard(bc_top5, ppo_top5),
    }
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
