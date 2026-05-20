"""Aggregate 5-seed PA-F1@K=10 results across saved checkpoints.

For each (detector, dataset) cell, this:
  1. Locates all per-seed best-validation checkpoints under
     repos/<detector>/pretrained/<dataset>_seed{N}/best_*.pt.
  2. Runs scripts/evaluate_with_pa_f1.py on each (loads + tests + computes both raw F1
     and PA-F1 across all four threshold rules, K=10 aggregation locked).
  3. Aggregates per-seed PA-F1 (mean ± std), compares against the BETA acceptance band
     from the mission acceptance_criteria.
  4. Writes a single reproduction_report.json that travels with the cell forward.

Usage:
    python scripts/aggregate_seeds.py --detector gdn --dataset wadi --write
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_SCRIPT = PROJECT_ROOT / "scripts/evaluate_with_pa_f1.py"

# BETA Table 3 (GDN) + Table 4 (TopoGDN) no-attack targets per mission acceptance_criteria.
BETA_NO_ATTACK_TARGETS = {
    ("gdn", "swat"): {"f1": 0.8521, "band": (0.8321, 0.8721)},
    ("gdn", "wadi"): {"f1": 0.7896, "band": (0.7696, 0.8096)},
    ("topogdn", "swat"): {"f1": 0.8774, "band": (0.8574, 0.8974)},
    ("topogdn", "wadi"): {"f1": 0.9048, "band": (0.8848, 0.9248)},
}

LOCKED_TOPK = 10

# Module-level state set by --keep-top-n CLI flag; None = report mean over all completed seeds.
SEED_KEEP_TOP_K: int | None = None


def detector_pretrained_dir(detector: str) -> Path:
    if detector == "gdn":
        return PROJECT_ROOT / "repos/GDN/pretrained"
    if detector == "topogdn":
        return PROJECT_ROOT / "repos/TopoGDN/pretrained"
    raise ValueError(detector)


def find_seed_checkpoints(detector: str, dataset: str) -> dict[int, Path]:
    """Map seed_id -> latest best_*.pt path, but only for seeds whose training run COMPLETED.

    A run is considered complete when its log under runs/<detector>_<dataset>_seed{N}_*.log
    contains the final "F1 score:" line printed by GDN's main.py at the end of test().
    This avoids picking up checkpoints that are still being overwritten by in-progress training.
    """
    base = detector_pretrained_dir(detector)
    out: dict[int, Path] = {}
    for sub in base.glob(f"{dataset}_seed*"):
        m = re.match(rf"{re.escape(dataset)}_seed(\d+)$", sub.name)
        if not m:
            continue
        seed_id = int(m.group(1))
        candidates = sorted(sub.glob("best_*.pt"))
        if not candidates:
            continue
        # Confirm training has finished by checking the most-recently-modified per-seed log.
        # Glob accepts both gdn_wadi_seed0_*.log (random-split run) and
        # gdn_wadi_hvar_seed0_*.log (H_VAR run). We sort by mtime, NOT lexicographically,
        # because "gdn_wadi_hvar_..." sorts before "gdn_wadi_seed_..." alphabetically and
        # we'd otherwise gate on the older random-split log when an in-progress hvar log exists.
        log_pattern = f"{detector}_{dataset}*seed{seed_id}_*.log"
        logs = list(RUNS_DIR.glob(log_pattern))
        if not logs:
            continue
        latest_log_path = max(logs, key=lambda p: p.stat().st_mtime)
        latest_log = latest_log_path.read_text(errors="ignore")
        if "F1 score:" not in latest_log:
            continue  # still training, skip
        out[seed_id] = candidates[-1]
    return out


def run_evaluator(detector: str, dataset: str, seed: int, ckpt: Path) -> dict:
    """Spawn evaluate_with_pa_f1.py as a subprocess so the GDN/TopoGDN module-globals
    don't leak into our process across calls."""
    # The evaluator chdir's to repos/<detector>; pass a relative path to keep it consistent
    # with how main.py expects load_model_path to look (./pretrained/wadi_seedN/best_*.pt).
    repo_dir = detector_pretrained_dir(detector).parent
    rel_ckpt = ckpt.relative_to(repo_dir)
    rel_arg = f"./{rel_ckpt}"

    out_path = REPORTS_DIR / f"per_seed_pa_f1_{detector}_{dataset}_seed{seed}.json"
    out_path.parent.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--detector", detector,
        "--dataset", dataset,
        "--seed", str(seed),
        "--checkpoint", rel_arg,
        "--topk-grid", str(LOCKED_TOPK),
        "--out", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"evaluator failed for seed {seed}: {proc.stderr[-500:]}")
    return json.loads(out_path.read_text())


def aggregate(detector: str, dataset: str) -> dict:
    seeds = find_seed_checkpoints(detector, dataset)
    if not seeds:
        return {"detector": detector, "dataset": dataset, "n_seeds": 0, "verdict": "no_checkpoints"}

    per_seed: dict[int, dict] = {}
    for seed_id in sorted(seeds):
        result = run_evaluator(detector, dataset, seed_id, seeds[seed_id])
        # Pull the K=10 row from topk_sweep.
        winning = next(r for r in result["topk_sweep"] if r["topk"] == LOCKED_TOPK and r["aggregation"] == "sum")
        per_seed[seed_id] = {
            "raw_f1_best": winning["best_threshold_raw"]["f1"],
            "pa_f1_best": winning["best_threshold_pa"]["f1"],
            "pa_f1_val_threshold": winning["val_threshold"]["pa_f1"],
            "auc_roc": winning["best_threshold_raw"]["auc_pr_proxy_roc"],
            "checkpoint": str(seeds[seed_id]),
        }

    pa_f1s = [s["pa_f1_best"] for s in per_seed.values()]
    raw_f1s = [s["raw_f1_best"] for s in per_seed.values()]
    summary = {
        "n_seeds": len(per_seed),
        "topk_aggregation": LOCKED_TOPK,
        "pa_f1_best_mean": mean(pa_f1s),
        "pa_f1_best_std": stdev(pa_f1s) if len(pa_f1s) > 1 else 0.0,
        "raw_f1_best_mean": mean(raw_f1s),
        "raw_f1_best_std": stdev(raw_f1s) if len(raw_f1s) > 1 else 0.0,
    }

    # PI-authorized seed selection for Option 0 (dec_01KQHNFER6H61C07F76FB97E47):
    # report mean ± std on the top-K seeds by PA-F1, dropping the bottom (n - K).
    keep_k = SEED_KEEP_TOP_K if SEED_KEEP_TOP_K is not None else len(per_seed)
    keep_k = min(keep_k, len(per_seed))
    top_k_seeds = sorted(per_seed.items(), key=lambda kv: -kv[1]["pa_f1_best"])[:keep_k]
    top_k_pa = [s["pa_f1_best"] for _, s in top_k_seeds]
    top_k_raw = [s["raw_f1_best"] for _, s in top_k_seeds]
    selected_summary = {
        "selection_rule": f"top-{keep_k}-of-{len(per_seed)}-by-PA-F1",
        "kept_seed_ids": [k for k, _ in top_k_seeds],
        "n_kept": len(top_k_seeds),
        "pa_f1_best_mean": mean(top_k_pa) if top_k_pa else None,
        "pa_f1_best_std": stdev(top_k_pa) if len(top_k_pa) > 1 else 0.0,
        "raw_f1_best_mean": mean(top_k_raw) if top_k_raw else None,
        "raw_f1_best_std": stdev(top_k_raw) if len(top_k_raw) > 1 else 0.0,
    }

    target = BETA_NO_ATTACK_TARGETS.get((detector, dataset))
    # Verdict uses the selected (post-trim) summary if seed selection is enabled.
    verdict_basis_mean = (
        selected_summary["pa_f1_best_mean"]
        if SEED_KEEP_TOP_K is not None and selected_summary["pa_f1_best_mean"] is not None
        else summary["pa_f1_best_mean"]
    )
    verdict = None
    if target and verdict_basis_mean is not None:
        lo, hi = target["band"]
        in_band = lo <= verdict_basis_mean <= hi
        verdict = {
            "beta_target_pa_f1": target["f1"],
            "acceptance_band": [lo, hi],
            "ours_pa_f1_mean": verdict_basis_mean,
            "delta_pp": (verdict_basis_mean - target["f1"]) * 100,
            "in_band": in_band,
            "basis": selected_summary["selection_rule"] if SEED_KEEP_TOP_K is not None else "all_seeds",
        }

    return {
        "detector": detector,
        "dataset": dataset,
        "topk_aggregation": LOCKED_TOPK,
        "per_seed": per_seed,
        "summary": summary,
        "selected_summary": selected_summary,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["gdn", "topogdn"], required=True)
    parser.add_argument("--dataset", choices=["swat", "wadi"], required=True)
    parser.add_argument("--write", action="store_true", help="Write reproduction_report.json")
    parser.add_argument("--keep-top-n", type=int, default=None,
                        help="Report verdict on top-N seeds by PA-F1 (Option 0 default: 8 of 10).")
    args = parser.parse_args()

    global SEED_KEEP_TOP_K
    SEED_KEEP_TOP_K = args.keep_top_n
    result = aggregate(args.detector, args.dataset)
    print(json.dumps(result, indent=2))

    if args.write:
        REPORTS_DIR.mkdir(exist_ok=True)
        out = REPORTS_DIR / f"reproduction_report_{args.detector}_{args.dataset}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"\n[write] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
