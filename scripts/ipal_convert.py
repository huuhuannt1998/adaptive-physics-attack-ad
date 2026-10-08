"""
Convert our defense-paper experimental outputs into IPAL state file + attacks.json
for evaluation via fkie-cad/ipal_evaluate.

Generates a JSONL state file where each line is:
  {"timestamp": <int>, "malicious": <bool>, "ids": <bool>}

And an attacks.json describing each anomaly time-range.

Six scenarios per (dataset, seed):
  1. clean_undefended:   no attack, no defense (detector baseline F1)
  2. clean_defended:     no attack, with defense (defense doesn't break clean F1)
  3. beta_undefended:    BETA attack, no defense (worst case for detector)
  4. beta_defended:      BETA attack, with our defense (defense recovery)
  5. ppo_undefended:     PPO attack, no defense  [optional, requires PPO ckpt]
  6. ppo_defended:       PPO attack, with our defense

Usage:
  python scripts/ipal_convert.py --seed 0 --dataset wadi --scenario beta_undefended \
    --out reports/ipal/wadi_seed0_beta_undefended.ipal
"""
from __future__ import annotations

import argparse
import gzip
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


def setup_gdn_victim(seed: int, dataset: str, arch: str = "GDN"):
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
            "slide_stride": 10, "comment": "ipal", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "ipal", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 256,
            "decay": 0.0, "val_ratio": 0.1, "topk": 30,
        }
    if arch.lower() == "topogdn":
        train_config["use_tcn"] = True
        train_config["use_topo"] = True
        train_config["model"] = "GDN"
    env_config = {
        "save_path": f"{dataset}_seed{seed}", "dataset": dataset, "report": "best",
        "device": "cpu", "load_model_path": ckpt_path,
    }
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()
    return m


def build_attacks_json(test_labels: np.ndarray, base_timestamp: int = 1451293200) -> list:
    """Each contiguous run of attack==1 becomes one entry in attacks.json.

    Timestamps shifted to start at base_timestamp (SWaT-like) for IPAL compatibility.
    """
    attacks = []
    in_run = False
    s = 0
    aid = 1
    for i, v in enumerate(test_labels):
        if v == 1 and not in_run:
            in_run = True
            s = i
        elif v == 0 and in_run:
            in_run = False
            attacks.append({
                "id": str(aid),
                "attack_point": [],
                "description": "",
                "start": int(base_timestamp + s),
                "end": int(base_timestamp + i - 1),
            })
            aid += 1
    if in_run:
        attacks.append({
            "id": str(aid),
            "attack_point": [],
            "description": "",
            "start": int(base_timestamp + s),
            "end": int(base_timestamp + len(test_labels) - 1),
        })
    return attacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    parser.add_argument("--scenario", required=True,
                        choices=["clean_undefended", "clean_defended",
                                 "beta_undefended", "beta_defended",
                                 "ppo_undefended", "ppo_defended"])
    parser.add_argument("--ppo-checkpoint", default=None,
                        help="Required for ppo_* scenarios")
    parser.add_argument("--threshold-percentile", type=float, default=99.5,
                        help="100=max-of-val (strictest); 99.5=defended convention; 95=more sensitive baseline")
    parser.add_argument("--threshold-abs", type=float, default=None,
                        help="Use this absolute threshold value (overrides percentile)")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit number of test windows for time (default: full test)")
    parser.add_argument("--max-attacks", type=int, default=2000,
                        help="For attack scenarios: cap BETA/PPO calls to N detected windows (rest stay clean)")
    parser.add_argument("--out", required=True, help="Output base path (will append .ipal.gz, .attacks.json)")
    parser.add_argument("--arch", default="GDN", choices=["GDN", "TopoGDN"])
    parser.add_argument("--ph-clip", type=float, default=None,
                        help="PH-feature magnitude clip (TopoGDN-only): "
                             "register a forward-hook on TopologyLayer to clamp "
                             "topoOut to [-c, c]. Phase 4.2 PH-regularization.")
    parser.add_argument("--hk-defense", action="store_true",
                        help="(TopoGDN only) Install HK PI-stability defense on TopologyLayer "
                             "instances (exploratory; not part of the IPCCC 2026 release).")
    parser.add_argument("--hk-refs", default=None,
                        help="Path to clean PI reference .pt; default: reports/hk_refs/{dataset}_seed{seed}.pt")
    parser.add_argument("--hk-projection-radius", type=float, default=2.0)
    parser.add_argument("--hk-sigma", type=float, default=0.05)
    parser.add_argument("--hk-persistence-clamp", type=float, default=1.0)
    args = parser.parse_args()

    defended = args.scenario.endswith("_defended")
    attacked = "beta" in args.scenario or "ppo" in args.scenario
    use_ppo = "ppo" in args.scenario
    if use_ppo and not args.ppo_checkpoint:
        raise ValueError("--ppo-checkpoint required for ppo_* scenarios")

    print(f"[setup] {args.arch}-{args.dataset} seed {args.seed}  defended={defended} attacked={attacked}")
    m = setup_gdn_victim(seed=args.seed, dataset=args.dataset, arch=args.arch)
    model = m.model
    is_topo = args.arch.lower() == "topogdn"

    # Phase 4.2: PH-feature magnitude clip via forward-hook on TopologyLayer
    if args.ph_clip is not None and args.ph_clip > 0 and is_topo:
        _clip = float(args.ph_clip)
        def _topo_clip_hook(module, inputs, output):
            out_act, ga1, filt = output
            return out_act.clamp(-_clip, _clip), ga1, filt
        n_hooks = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "TopologyLayer":
                mod.register_forward_hook(_topo_clip_hook)
                n_hooks += 1
        print(f"[ph-reg] magnitude-clip={_clip} on {n_hooks} TopologyLayer instance(s)")

    # Optional HK PI-stability defense via topoPooling.forward patch (exploratory)
    if args.hk_defense:
        if not is_topo:
            raise ValueError("--hk-defense requires --arch TopoGDN")
        from defenses.hk_stability import HKConfig, HKStabilityWrapper
        refs_path = args.hk_refs or f"reports/hk_refs/{args.dataset}_seed{args.seed}.pt"
        refs_path_full = refs_path if Path(refs_path).is_absolute() else str(PROJECT_ROOT / refs_path)
        print(f"[hk] loading reference PIs from {refs_path_full}")
        refs = torch.load(refs_path_full, weights_only=False, map_location="cpu")
        ref_p0 = refs["ref_p0"]
        hk_cfg = HKConfig(
            grid_size=ref_p0.shape[-1],
            sigma=args.hk_sigma,
            persistence_clamp=args.hk_persistence_clamp,
            projection_radius=args.hk_projection_radius,
        )
        wrapper = HKStabilityWrapper(model, clean_references=ref_p0, config=hk_cfg)
        n_patched = 0
        for _name, mod in model.named_modules():
            if mod.__class__.__name__ == "TopologyLayer":
                wrapper._install_hook(mod)
                n_patched += 1
        print(f"[hk] patched {n_patched} TopologyLayer instance(s) "
              f"(radius={hk_cfg.projection_radius}, sigma={hk_cfg.sigma})")
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    def model_fwd(X, ei=None):
        if is_topo:
            out = model(X)
            return out[0] if isinstance(out, tuple) else out
        return model(X, ei)

    from scipy.stats import iqr as scipy_iqr
    val_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, _, ei = batch
            xf, yf = x.float(), y.float()
            if defended:
                xf = xf.clamp(0.0, 1.0)
                yf = yf.clamp(0.0, 1.0)
            f = model_fwd(xf, ei.float())
            val_deltas.append((f - yf).abs().cpu().numpy())
    val_deltas = np.concatenate(val_deltas, axis=0)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    edge_index_template = next(iter(test_loader))[3].float()
    EPS = 1e-2

    def surrogate_query(X: torch.Tensor) -> torch.Tensor:
        Xf = X.clamp(0.0, 1.0) if defended else X
        forecast = model_fwd(Xf, edge_index_template)
        target_step = Xf[..., -1]
        delta = (forecast - target_step).abs()
        return (delta - median) / (iqr_v.abs() + EPS)

    test_batches = list(test_loader)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)
    if defended:
        flat_xs = flat_xs.clamp(0.0, 1.0)
    test_labels_per_window = torch.cat([b[2].float() for b in test_batches], dim=0).cpu().numpy().astype(int)
    n_test, n_sensors, W = flat_xs.shape

    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        xf, yf = x.float(), y.float()
        if defended:
            xf = xf.clamp(0.0, 1.0)
            yf = yf.clamp(0.0, 1.0)
        with torch.no_grad():
            f = model_fwd(xf, ei.float())
        s = (f - yf).abs() - median
        s = s / (iqr_v.abs() + EPS)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)
    if args.threshold_abs is not None:
        threshold = float(args.threshold_abs)
        print(f"[threshold] {threshold:.4f} (absolute, F1-optimal)")
    elif args.threshold_percentile >= 100.0:
        threshold = float(val_scores_arr.max())
        print(f"[threshold] {threshold:.4f} (max-of-val)")
    else:
        threshold = float(np.percentile(val_scores_arr, args.threshold_percentile))
        print(f"[threshold] {threshold:.4f} ({args.threshold_percentile} percentile of val)")

    n_max = args.max_windows if args.max_windows else n_test
    n_max = min(n_max, n_test)

    # Pre-compute per-window IDS predictions (no attack baseline)
    BATCH = 256
    ids_clean = np.zeros(n_max, dtype=bool)
    with torch.no_grad():
        for start in range(0, n_max, BATCH):
            end = min(start + BATCH, n_max)
            X = flat_xs[start:end]
            scores = surrogate_query(X)
            max_scores, _ = scores.max(dim=1)
            ids_clean[start:end] = (max_scores > threshold).cpu().numpy()
    print(f"[clean] base detector fired on {ids_clean.sum()}/{n_max} windows ({ids_clean.mean()*100:.1f}%)")

    if attacked:
        ids_attacked = ids_clean.copy()
        detected_idxs = np.where(ids_clean)[0]
        rng = np.random.default_rng(args.seed)
        if len(detected_idxs) > args.max_attacks:
            attack_idxs = rng.choice(detected_idxs, args.max_attacks, replace=False)
        else:
            attack_idxs = detected_idxs
        attack_idxs = sorted(attack_idxs.tolist())
        attacker_name = "PPO" if use_ppo else "BETA"
        print(f"[attack-plan] running {attacker_name} on {len(attack_idxs)} of {len(detected_idxs)} detected windows")

        if use_ppo:
            from torch.distributions import Categorical, Bernoulli
            from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states
            from attack_env.mdp import AdaptiveAttackEnv, MDPConfig
            ppo_path = Path(args.ppo_checkpoint)
            if not ppo_path.is_absolute():
                ppo_path = PROJECT_ROOT / ppo_path
            ckpt = torch.load(ppo_path, weights_only=False)
            pcfg = PolicyConfig(**ckpt["config"])
            policy = AdaptiveAttackPolicy(pcfg)
            policy.load_state_dict(ckpt["policy_state_dict"])
            policy.eval()
            import pandas as pd
            train_df = pd.read_csv(PROJECT_ROOT / f"repos/GDN/data/{args.dataset}/train.csv", index_col=0)
            train_x = torch.from_numpy(train_df.drop(columns=["attack"]).values).float()
            nominal = torch.zeros(n_sensors, 1024)
            for i in range(n_sensors):
                idx = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(args.seed))[:1024]
                nominal[i] = train_x[idx, i]
            mdp_cfg = MDPConfig(epsilon_pgd=0.1, target_score_history_len=pcfg.target_score_history_len,
                                epsilon_reward_floor=0.1,
                                lambda_1_sparsity=0.0, lambda_2_stealth=0.0, lambda_3_physics=0.0,
                                per_window_budget_default=args.budget)
            env = AdaptiveAttackEnv(surrogate_query, nominal, mdp_cfg)
        else:
            beta_cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
            beta_attack = BETAAttack(surrogate_query, lambda: model.learned_graph.detach(),
                                     num_nodes=n_sensors, config=beta_cfg)

        n_flipped = 0
        eps = 0.1
        for n_done, w_idx in enumerate(attack_idxs):
            X = flat_xs[w_idx:w_idx + 1].clone()
            with torch.no_grad():
                scores = surrogate_query(X)[0]
                _, max_idx = scores.max(dim=0)
            target_idx = int(max_idx.item())
            try:
                if use_ppo:
                    torch.manual_seed(args.seed * 1000 + n_done)
                    env.reset(X.clone(), target_idx, budget=args.budget)
                    for _ in range(args.budget):
                        state = env._encode_state()
                        batch = collate_states([state])
                        with torch.no_grad():
                            out = policy(batch)
                        cat_logits = out["cat_logits"][0].clone()
                        cat_logits[target_idx] = -float("inf")
                        full_idx = int(Categorical(logits=cat_logits).sample().item())
                        cat_in_v_minus_u = full_idx if full_idx < target_idx else full_idx - 1
                        sign_sample = Bernoulli(logits=out["sign_logits"][0]).sample()
                        delta_signed = ((sign_sample * 2 - 1) * eps).cpu().numpy()
                        res = env.step((cat_in_v_minus_u, delta_signed))
                        if res.done or res.truncated:
                            break
                    with torch.no_grad():
                        scores_after = surrogate_query(env._current_window)[0]
                        max_after, _ = scores_after.max(dim=0)
                else:
                    X_pert, _ = beta_attack.attack(X.clone(), target_idx, args.budget)
                    with torch.no_grad():
                        scores_after = surrogate_query(X_pert)[0]
                        max_after, _ = scores_after.max(dim=0)
                new_pred = (max_after.item() > threshold)
                if not new_pred and ids_clean[w_idx]:
                    n_flipped += 1
                ids_attacked[w_idx] = new_pred
            except Exception:
                pass
            if (n_done + 1) % 200 == 0:
                print(f"  [attack] {n_done+1}/{len(attack_idxs)}, {n_flipped} flipped")
        ids_final = ids_attacked
        asr = n_flipped / max(1, len(attack_idxs))
        print(f"[asr] flipped {n_flipped}/{len(attack_idxs)} = {asr:.3f} attack-success-rate")
        print(f"[final] post-attack detector fired on {ids_final.sum()}/{n_max} ({ids_final.mean()*100:.1f}%)")
    else:
        ids_final = ids_clean

    # Write IPAL JSONL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = 1451293200
    state_path = str(out_path) + ".ipal.gz"
    with gzip.open(state_path, "wt") as fp:
        for i in range(n_max):
            line = json.dumps({
                "timestamp": int(base_ts + i),
                "malicious": bool(test_labels_per_window[i]),
                "ids": bool(ids_final[i]),
            })
            fp.write(line + "\n")
    print(f"[write] {state_path}")

    # Build attacks.json from labels
    attacks = build_attacks_json(test_labels_per_window[:n_max], base_timestamp=base_ts)
    attacks_path = str(out_path) + ".attacks.json"
    Path(attacks_path).write_text(json.dumps(attacks, indent=2))
    print(f"[write] {attacks_path}  ({len(attacks)} attack ranges)")


if __name__ == "__main__":
    main()
