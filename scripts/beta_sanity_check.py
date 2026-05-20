"""
Single-cell BETA reproduction sanity check on TopoGDN-WADI seed 0.

Per Brain (jrn_01KQNCCCHBPQC36W4QV1B32132): we target the FTA *delta* (clean → B=5),
not the absolute number, since our TopoGDN baseline is already shifted (PA-F1=0.71 vs
BETA 0.90). Acceptance: our FTA delta within ±3pp of BETA's reported delta.

Protocol:
  1. Load TopoGDN seed 0 best checkpoint (Path A' baseline, _oldim256).
  2. Build the test data loader.
  3. Sample N_WINDOWS test windows × K_TARGETS random target sensors per window.
  4. For each (window, target):
     a. Compute clean score via GDN's standardization pipeline.
     b. Binary decision = (score > threshold). Threshold = max validation residual.
     c. Run BETA attack with B=5 to perturb V_bar.
     d. Compute attacked score.
     e. Record whether decision flipped.
  5. Report: clean FTA, attacked FTA, delta, vs BETA's published delta.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPOGDN_REPO = PROJECT_ROOT / "repos/TopoGDN"
sys.path.insert(0, str(PROJECT_ROOT))


def setup_victim(detector: str, seed: int = 0):
    """Use {GDN, TopoGDN, gdn_pyg1x}'s Main class with load_model_path to get a fully wired victim."""
    if detector == "topogdn":
        repo = TOPOGDN_REPO
        ckpt_dir = repo / f"pretrained/wadi_seed{seed}_oldim256"
        train_config_extra = {"use_tcn": True, "use_topo": True, "model": "GDN"}
        save_path = f"wadi_seed{seed}_oldim256"
    elif detector == "gdn":
        repo = PROJECT_ROOT / "repos/GDN"
        ckpt_dir = repo / f"pretrained/wadi_seed{seed}"
        train_config_extra = {}
        save_path = f"wadi_seed{seed}"
    elif detector == "gdn_pyg1x":
        repo = PROJECT_ROOT / "repos/GDN_pyg1x"
        ckpt_dir = repo / f"pretrained/wadi_pyg1x_seed{seed}"
        train_config_extra = {}
        save_path = f"wadi_pyg1x_seed{seed}"
    else:
        raise ValueError(detector)

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidates = list(ckpt_dir.glob("best_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    ckpt_path = "./" + str(candidates[-1].relative_to(repo).as_posix())

    train_config = {
        "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
        "slide_stride": 10, "comment": "beta_sanity", "seed": seed,
        "out_layer_num": 1, "out_layer_inter_dim": 256,
        "decay": 0.0, "val_ratio": 0.1, "topk": 30,
        **train_config_extra,
    }
    env_config = {
        "save_path": save_path,
        "dataset": "wadi",
        "report": "best",
        "device": "cpu",
        "load_model_path": ckpt_path,
    }

    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.model.eval()
    return m


def setup_topogdn(seed: int = 0):
    return setup_victim("topogdn", seed)


def make_forecast_fn(model, detector: str):
    """Returns callable X -> (B, N) forecast that handles the detector-specific forward signature."""
    if detector == "topogdn":
        def f(X):
            forecast, _ = model(X)
            return forecast
        return f
    elif detector in ("gdn", "gdn_pyg1x"):
        # GDN.forward(data, org_edge_index) — needs edge_index. We'll use the dataloader's
        # batch to capture it on first call.
        def f(X, edge_index_cache=[None]):
            if edge_index_cache[0] is None:
                raise RuntimeError("edge_index not yet captured; call set_edge_index first")
            return model(X, edge_index_cache[0])
        return f
    raise ValueError(detector)


def collect_residuals_per_sensor(model, dataloader, detector: str):
    """Run model over loader, collect |forecast - actual| per sensor for stats."""
    deltas: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            x, y, labels, edge_index = batch[0], batch[1], batch[2], batch[3]
            x = x.float()
            y = y.float()
            if detector == "topogdn":
                forecast, _ = model(x)
            else:
                forecast = model(x, edge_index.float())
            delta = (forecast - y).abs().cpu().numpy()  # (B, N)
            deltas.append(delta)
    return np.concatenate(deltas, axis=0)  # (T, N)


def standardize_score(forecast: torch.Tensor, target: torch.Tensor, median: torch.Tensor, iqr: torch.Tensor, eps: float = 1e-2):
    """GDN's per-sensor standardized residual, mirroring evaluate.py:60."""
    delta = (forecast - target).abs()
    return (delta - median) / (iqr.abs() + eps)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["gdn", "topogdn", "gdn_pyg1x"], default="topogdn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-filter", action="store_true",
                        help="Restrict FTA evaluation to ground-truth labeled-anomaly windows.")
    parser.add_argument("--topk-agg", type=int, default=1,
                        help="Top-K sensor aggregation when computing detection event (default: 1 = max).")
    parser.add_argument("--per-sensor-threshold", action="store_true",
                        help="Use per-sensor thresholds (each sensor's threshold = max val on that sensor).")
    parser.add_argument("--n-windows", type=int, default=20, help="number of test windows to sample")
    parser.add_argument("--k-targets", type=int, default=5, help="targets per window")
    parser.add_argument("--budget", type=int, default=5, help="BETA budget B")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"[setup] loading {args.detector.upper()} seed {args.seed}")
    m = setup_victim(args.detector, seed=args.seed)
    model = m.model
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader

    print(f"[stats] computing per-sensor residual median + IQR on validation set")
    from scipy.stats import iqr as scipy_iqr

    # Capture an edge_index from the first batch (constant across the dataset for our use).
    first_batch = next(iter(test_loader))
    edge_index_template = first_batch[3].float()

    val_deltas = collect_residuals_per_sensor(model, val_loader, args.detector)
    median = torch.from_numpy(np.median(val_deltas, axis=0)).float()
    iqr_v = torch.from_numpy(scipy_iqr(val_deltas, axis=0)).float()
    print(f"[stats] median shape={tuple(median.shape)}, mean={median.mean():.4f}; iqr mean={iqr_v.mean():.4f}")

    # Threshold = max validation standardized score. We compute max over (timesteps, sensors)
    # to mirror BETA's "max validation error" convention.
    val_scores = []
    for batch in val_loader:
        x, y, _, ei = batch
        with torch.no_grad():
            if args.detector == "topogdn":
                f, _ = model(x.float())
            else:
                f = model(x.float(), ei.float())
        s = standardize_score(f, y.float(), median, iqr_v)  # (B, N)
        val_scores.append(s.cpu().numpy())
    val_scores_arr = np.concatenate(val_scores, axis=0)  # (T_val, N)

    # Top-K aggregation for the threshold and detection check.
    K = args.topk_agg
    if K == 1:
        val_agg = val_scores_arr.max(axis=1)  # (T_val,)
        agg_label = "max"
    else:
        # top-K-sum across sensors per timestep
        topk_idx = np.argpartition(val_scores_arr, -K, axis=1)[:, -K:]
        val_agg = np.take_along_axis(val_scores_arr, topk_idx, axis=1).sum(axis=1)
        agg_label = f"top{K}_sum"

    if args.per_sensor_threshold:
        # Each sensor has its own threshold = max validation standardized score for that sensor.
        per_sensor_thresholds_np = val_scores_arr.max(axis=0)  # (N,)
        per_sensor_thresholds = torch.from_numpy(per_sensor_thresholds_np).float()
        threshold = None  # not used in per-sensor mode
        print(f"[threshold] per-sensor mode; "
              f"min={per_sensor_thresholds.min():.4f}  median={per_sensor_thresholds.median():.4f}  max={per_sensor_thresholds.max():.4f}")
    else:
        per_sensor_thresholds = None
        threshold = float(val_agg.max())
        print(f"[threshold] aggregation={agg_label}; max validation aggregated score = {threshold:.4f}")

    # Now sample N_WINDOWS test windows uniformly.
    test_batches = list(test_loader)
    total_test_size = sum(b[0].shape[0] for b in test_batches)
    print(f"[test] total test windows = {total_test_size}")

    rng = random.Random(args.seed)
    flat_xs = torch.cat([b[0].float() for b in test_batches], dim=0)  # (T, N, W)
    flat_labels = torch.cat([b[2].float() for b in test_batches], dim=0)  # (T,) per-window 0/1
    print(f"[labels] anomaly windows in test: {int(flat_labels.sum().item())}/{flat_labels.shape[0]} "
          f"({100*flat_labels.float().mean().item():.2f}%)")

    n_sensors = flat_xs.shape[1]

    from attacks.beta import BETAAttack, BETAConfig

    def victim_forward(X: torch.Tensor) -> torch.Tensor:
        if args.detector == "topogdn":
            forecast, _ = model(X)
        else:
            forecast = model(X, edge_index_template)
        target = X[..., -1]
        return standardize_score(forecast, target, median, iqr_v)

    def learned_edge_index_fn():
        return model.learned_graph.detach()

    cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.01, pgd_iters=10, pgd_restarts=5, candidate_k=32)
    attack = BETAAttack(victim_forward, learned_edge_index_fn, num_nodes=n_sensors, config=cfg)

    # BETA's FTA convention (per Brain framing): rate at which DETECTED anomalies
    # remain detected after attack. We sample windows where model already detects an
    # anomaly (max sensor score > threshold), use that argmax sensor as the target,
    # and measure attack success = the score gets pushed below threshold.
    # clean_FTA = 1.0 by construction; attacked_FTA = fraction still detected; delta = 1 - attacked_FTA.

    if args.per_sensor_threshold:
        print(f"[scan] finding test windows where ANY sensor exceeds its per-sensor threshold ...")
    else:
        print(f"[scan] finding test windows where model detects an anomaly (agg={agg_label}, threshold={threshold:.2f}) ...")
    detected_windows: list[tuple[int, int]] = []  # (window_idx, target_sensor_idx)
    BATCH = 256
    with torch.no_grad():
        for start in range(0, flat_xs.shape[0], BATCH):
            X = flat_xs[start:start + BATCH]
            scores = victim_forward(X)  # (B, N)
            if args.per_sensor_threshold:
                # Detection event = ANY sensor exceeds its specific threshold.
                # Target = sensor with the largest exceedance margin (score - threshold).
                margins = scores - per_sensor_thresholds.unsqueeze(0)  # (B, N)
                max_margin, max_idx = margins.max(dim=1)
                detected_mask = max_margin > 0  # (B,)
                for i in range(X.shape[0]):
                    if detected_mask[i].item():
                        detected_windows.append((start + i, int(max_idx[i].item())))
            else:
                if K == 1:
                    agg_scores = scores.max(dim=1).values
                else:
                    topk_vals, _ = torch.topk(scores, k=K, dim=1)
                    agg_scores = topk_vals.sum(dim=1)
                argmax_idx = scores.argmax(dim=1)
                for i in range(X.shape[0]):
                    if agg_scores[i].item() > threshold:
                        detected_windows.append((start + i, int(argmax_idx[i].item())))
    print(f"[scan] detected {len(detected_windows)} windows out of {flat_xs.shape[0]} (rate {len(detected_windows)/flat_xs.shape[0]:.4f})")

    if args.label_filter:
        before = len(detected_windows)
        detected_windows = [
            (w, t) for (w, t) in detected_windows if flat_labels[w].item() > 0.5
        ]
        print(f"[label-filter] kept {len(detected_windows)}/{before} "
              f"detected-AND-true-anomaly windows (drop rate {100*(1-len(detected_windows)/max(before,1)):.2f}%)")

    if len(detected_windows) == 0:
        print("[abort] zero detected windows after filter — cannot measure FTA")
        return

    n_trials = min(args.n_windows * args.k_targets, len(detected_windows))
    sampled = rng.sample(detected_windows, n_trials)

    def aggregate(scores_per_sensor: torch.Tensor) -> torch.Tensor:
        """Apply top-K aggregation matching the detection event (global threshold mode only)."""
        if K == 1:
            return scores_per_sensor.max(dim=-1).values
        topk_vals, _ = torch.topk(scores_per_sensor, k=K, dim=-1)
        return topk_vals.sum(dim=-1)

    def is_detected(scores_per_sensor: torch.Tensor) -> bool:
        """Per-sensor: any sensor s where score[s] > threshold[s].
        Global: aggregate(score) > global threshold."""
        if args.per_sensor_threshold:
            return bool((scores_per_sensor - per_sensor_thresholds).max().item() > 0)
        return bool(aggregate(scores_per_sensor).item() > threshold)

    print(f"[attack] running BETA on {n_trials} (window, target=argmax) pairs ...")
    survived = 0  # post-attack still-detected count
    flipped = 0
    score_deltas = []
    for trial_i, (w_idx, target_idx) in enumerate(sampled):
        X = flat_xs[w_idx:w_idx + 1]
        with torch.no_grad():
            clean_scores = victim_forward(X)[0]  # (N,)
        clean_above = is_detected(clean_scores)
        try:
            X_pert, meta = attack.attack(X.clone(), target_idx, args.budget)
        except Exception as e:
            print(f"  trial {trial_i}: ATTACK FAILED ({type(e).__name__}: {e})")
            continue
        with torch.no_grad():
            attacked_scores = victim_forward(X_pert)[0]
        attacked_above = is_detected(attacked_scores)

        if attacked_above:
            survived += 1
        else:
            flipped += 1

        delta_target = float(attacked_scores[target_idx] - clean_scores[target_idx])
        score_deltas.append({
            "window": w_idx,
            "target": target_idx,
            "clean_target_score": float(clean_scores[target_idx]),
            "attacked_target_score": float(attacked_scores[target_idx]),
            "delta_agg": delta_target,
            "clean_above": clean_above,
            "attacked_above": attacked_above,
            "flipped": clean_above != attacked_above,
        })

        if (trial_i + 1) % 5 == 0:
            sr = survived / (trial_i + 1)
            print(f"  trial {trial_i+1}/{n_trials}  flipped={flipped}  attacked_FTA={sr:.4f}  attack_success={1-sr:.4f}")

    n = len(score_deltas)
    attacked_FTA = survived / n if n else None
    attack_success_rate = (1 - attacked_FTA) if attacked_FTA is not None else None
    # Brain's adjusted target: BETA TopoGDN-WADI FTA delta (clean → B=5) reportedly -26.3pp.
    # Our delta = clean_FTA - attacked_FTA = 1.0 - attacked_FTA = attack_success_rate.

    result = {
        "seed": args.seed,
        "n_trials": n,
        "budget": args.budget,
        "threshold": threshold,
        "detected_window_count": len(detected_windows),
        "test_window_count": int(flat_xs.shape[0]),
        "clean_FTA": 1.0,
        "attacked_FTA": attacked_FTA,
        "delta_FTA": attack_success_rate,  # = clean_FTA - attacked_FTA
        "beta_target_delta_topogdn_wadi": 0.263,  # per Brain spec; ±3pp tolerance
        "in_tolerance": (
            abs(attack_success_rate - 0.263) <= 0.03 if attack_success_rate is not None else None
        ),
        "epsilon": cfg.epsilon,
        "pgd_alpha": cfg.pgd_alpha,
        "pgd_iters": cfg.pgd_iters,
        "pgd_restarts": cfg.pgd_restarts,
        "candidate_k": cfg.candidate_k,
        "score_delta_mean": float(np.mean([d["delta_agg"] for d in score_deltas])) if score_deltas else None,
        "score_delta_abs_mean": float(np.mean([abs(d["delta_agg"]) for d in score_deltas])) if score_deltas else None,
        "topk_agg": K,
        "label_filter": args.label_filter,
        "trial_examples": score_deltas[:10],
    }
    print(json.dumps(result, indent=2, default=str))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"[write] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
