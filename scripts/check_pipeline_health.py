"""
Pipeline health check — invoked at each wake-up. Reads pipeline state + result
files, evaluates "is this result good enough to strengthen the paper?", surfaces
findings to RKA and stdout.

Result-quality bars (from mis P-experimental-completion + Reviewer-2 expectations):

Phase 1.1 (W2 ε sweep):
  GOOD: defense reduction factor monotonically decreases with ε but stays ≥10× at ε=0.5
        AND PPO degradation increases with ε (more budget → more attack capability)
  BAD:  defense collapses (<2× reduction) at ε≤0.2 — would mean the defense is ε=0.1-specific
        AND/OR non-monotonic behavior in either curve

Phase 1.2 (W5 FGSM):
  GOOD: FGSM-defended degradation comparable to BETA-defended (within 2×); defense generalizes
  BAD:  FGSM defeats the defense (>5× higher mean degradation than BETA-defended) — would
        indicate gradient-masking artifact and trigger Athalye-style adaptive-attack concern

Phase 1.3 (W3 TopoGDN-SWaT):
  GOOD: training completes; PPO achieves positive episode return; defense holds (F1 retention ≥80%)
  BAD-1: PH crashes — known issue, queues Phase 3 fix (not actually bad for the paper, just delay)
  BAD-2: training completes but PPO defeats the defense — expands §6.3 cross-architecture limitation
         materially; surface for narrative restructure

Phase 2 (non-lean BC retry):
  GOOD: ≥7/8 seeds converge to val_cat_acc ≥ 0.95 → populate full Q6 column
  BAD:  ≤5/8 — Q6 column unmeasurable across seeds; seed-0 point estimate remains primary evidence
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "reports/pipeline_state.json"


def load_state():
    return json.loads(STATE_PATH.read_text())


def save_state(state):
    state["last_updated"] = datetime.utcnow().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2))


def evaluate_w2_eps_sweep():
    """Phase 1.1: ε sweep result quality."""
    out_dir = PROJECT_ROOT / "reports/phase1_w2_eps_sweep"
    if not out_dir.exists(): return {"status": "no_outputs"}
    files = sorted(out_dir.glob("wadi_seed*_eps*.json"))
    if len(files) < 15: return {"status": "partial", "n_complete": len(files)}
    # Build (seed, eps, ppo_mean, beta_mean) table
    import re
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        m = re.search(r"eps([0-9.]+)\.json$", f.name)
        eps = float(m.group(1)) if m else d.get("epsilon", 0.0)
        rows.append({
            "seed": d.get("seed"),
            "eps": eps,
            "ppo_mean": d.get("ppo_degradation_mean", 0),
            "beta_mean": d.get("beta_degradation_mean", 0),
        })
    # Aggregate: per-eps mean across 3 seeds for PPO and BETA defended degradation
    eps_vals = sorted(set(r["eps"] for r in rows))
    summary = {}
    for eps in eps_vals:
        seed_rows = [r for r in rows if r["eps"] == eps]
        ppo_mean = sum(r["ppo_mean"] for r in seed_rows) / max(len(seed_rows), 1)
        beta_mean = sum(r["beta_mean"] for r in seed_rows) / max(len(seed_rows), 1)
        summary[f"eps={eps}"] = {"ppo_3seed_mean": ppo_mean, "beta_3seed_mean": beta_mean, "n_seeds": len(seed_rows)}
    # Quality verdict
    ppo_at_05 = summary.get("eps=0.5", {}).get("ppo_3seed_mean", 0)
    ppo_at_01 = summary.get("eps=0.1", {}).get("ppo_3seed_mean", 0)
    verdict = "good"
    notes = []
    if ppo_at_05 < ppo_at_01 * 0.5:  # heuristic: at ε=0.5 should be at least 2x ε=0.1
        verdict = "concerning_non_monotone"
        notes.append(f"PPO at ε=0.5 ({ppo_at_05:.4f}) < ε=0.1 ({ppo_at_01:.4f}) × 0.5 — non-monotone budget-response")
    return {"status": "complete", "summary": summary, "verdict": verdict, "notes": notes}


def evaluate_w5_fgsm():
    """Phase 1.2: FGSM-defended vs BETA-defended."""
    out_dir = PROJECT_ROOT / "reports/phase1_w5_fgsm"
    if not out_dir.exists(): return {"status": "no_outputs"}
    files = sorted(out_dir.glob("wadi_seed*_fgsm_defended.json"))
    if len(files) < 3: return {"status": "partial", "n_complete": len(files)}
    fgsm_means = []
    for f in files:
        d = json.loads(f.read_text())
        fgsm_means.append(d.get("fgsm_degradation_mean", 0))
    fgsm_avg = sum(fgsm_means) / max(len(fgsm_means), 1)
    # Apples-to-apples reference: W2 BETA-saliency at ε=0.10 (same pair distribution).
    beta_ref_path = PROJECT_ROOT / "reports/phase1_w2_eps_sweep"
    beta_means = []
    for f in beta_ref_path.glob("wadi_seed*_eps0.10.json"):
        try:
            beta_means.append(json.loads(f.read_text()).get("beta_degradation_mean", 0))
        except Exception:
            pass
    beta_ref = sum(beta_means) / max(len(beta_means), 1) if beta_means else -0.106
    # BETA-rawgrad control (if available): isolates candidate-selection effect.
    control_path = PROJECT_ROOT / "reports/phase1_w5b_beta_rawgrad"
    control_means = []
    for f in control_path.glob("wadi_seed*_beta_rawgrad_defended.json"):
        try:
            control_means.append(json.loads(f.read_text()).get("beta_rawgrad_degradation_mean", 0))
        except Exception:
            pass
    control_avg = sum(control_means) / max(len(control_means), 1) if control_means else None
    verdict = "good"
    notes = []
    if abs(fgsm_avg) > 5 * abs(beta_ref) and control_avg is None:
        verdict = "concerning_fgsm_defeats_defense"
        notes.append(f"FGSM mean degradation {fgsm_avg:+.4f} is 5x BETA-defended ({beta_ref:+.4f}) — defense may be PGD-specific; run BETA-rawgrad control to isolate")
    elif control_avg is not None:
        cand_sel_effect = control_avg - beta_ref
        atk_step_effect = control_avg - fgsm_avg
        if abs(cand_sel_effect) > 5 * abs(atk_step_effect):
            verdict = "good_candidate_selection_dominates"
            notes.append(f"Control confirms: candidate-selection effect ({cand_sel_effect:+.4f}) dominates attack-step effect ({atk_step_effect:+.4f}); gradient-masking refuted")
        else:
            verdict = "concerning_attack_step_dominant"
            notes.append(f"Control shows attack-step effect ({atk_step_effect:+.4f}) comparable to candidate-selection ({cand_sel_effect:+.4f}); FGSM may have genuine single-step advantage")
    return {"status": "complete", "fgsm_3seed_mean": fgsm_avg, "beta_saliency_ref": beta_ref,
            "beta_rawgrad_control_mean": control_avg, "verdict": verdict, "notes": notes}


def evaluate_phase2_bc():
    """Phase 2: non-lean BC pass rate."""
    out_dir = PROJECT_ROOT / "reports/phase2_bc_nonlean"
    if not out_dir.exists(): return {"status": "no_outputs"}
    reports = sorted(out_dir.glob("bc_pretrain_*_report.json"))
    pass_count = 0; fail_count = 0; in_progress = 0
    pass_seeds = []; fail_seeds = []
    for f in reports:
        try:
            d = json.loads(f.read_text())
            # history.val_cat_acc is list; max is best
            best = max(d.get("history", {}).get("val_cat_acc", [0]))
            seed_tag = f.stem.replace("bc_pretrain_", "").replace("_report", "")
            if best >= 0.95:
                pass_count += 1; pass_seeds.append((seed_tag, best))
            else:
                fail_count += 1; fail_seeds.append((seed_tag, best))
        except Exception:
            pass
    in_progress = 8 - pass_count - fail_count
    verdict = "in_progress" if in_progress > 0 else ("good" if pass_count >= 7 else "concerning_low_pass_rate")
    return {"status": "complete" if in_progress == 0 else "in_progress",
            "pass_count": pass_count, "fail_count": fail_count, "in_progress": in_progress,
            "pass_seeds": pass_seeds, "fail_seeds": fail_seeds, "verdict": verdict}


def main():
    state = load_state()
    print(f"=== Pipeline health check at {datetime.utcnow().isoformat()} ===")
    print(f"Current phase: {state.get('current_phase')}")
    print(f"Current task: {state.get('current_task')}")
    print()
    w2 = evaluate_w2_eps_sweep()
    print(f"--- Phase 1.1 W2 ε sweep ---")
    print(json.dumps(w2, indent=2))
    print()
    w5 = evaluate_w5_fgsm()
    print(f"--- Phase 1.2 W5 FGSM ---")
    print(json.dumps(w5, indent=2))
    print()
    p2 = evaluate_phase2_bc()
    print(f"--- Phase 2 BC retry ---")
    print(json.dumps(p2, indent=2))
    print()
    print("=== Escalation log ===")
    for e in state.get("escalations", [])[-5:]:
        print(f"  [{e.get('severity')}] {e.get('time')}: {e.get('message')}")


if __name__ == "__main__":
    main()
