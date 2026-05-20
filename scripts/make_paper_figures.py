"""
Generate publication figures from eval JSONs for paper Section VIII.

Outputs to paper/figures/:
  - fig_5seed_defended.pdf — 5-seed PPO vs BETA bar chart with error bars
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_5seed_defended():
    """Load PPO defended-trained vs BETA defended for 5 seeds."""
    rows = []
    for s in range(5):
        p = PROJECT_ROOT / f"reports/eval_defended_trained_seed{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append({
            "seed": s,
            "ppo_mean": d["ppo_degradation_mean"],
            "ppo_std": d["ppo_degradation_std"],
            "beta_mean": d["beta_degradation_mean"],
            "beta_std": d["beta_degradation_std"],
        })
    return rows


def fig_5seed_defended():
    """10-seed cross-dataset figure: WADI seeds 0-4 + SWaT seeds 0-4."""
    wadi_rows = load_5seed_defended()
    # Load SWaT 5-seed
    swat_rows = []
    for s in range(5):
        p = PROJECT_ROOT / f"reports/eval_defended_swat_seed{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        swat_rows.append({
            "seed": s,
            "ppo_mean": d["ppo_degradation_mean"],
            "ppo_std": d["ppo_degradation_std"],
            "beta_mean": d["beta_degradation_mean"],
            "beta_std": d["beta_degradation_std"],
        })

    n_wadi = len(wadi_rows)
    n_swat = len(swat_rows)
    n_total = n_wadi + n_swat

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(n_total)
    width = 0.36

    ppo_m = np.array([r["ppo_mean"] for r in wadi_rows + swat_rows])
    ppo_s = np.array([r["ppo_std"] for r in wadi_rows + swat_rows])
    beta_m = np.array([r["beta_mean"] for r in wadi_rows + swat_rows])
    beta_s = np.array([r["beta_std"] for r in wadi_rows + swat_rows])
    sem_ppo = ppo_s / np.sqrt(30)
    sem_beta = beta_s / np.sqrt(30)

    ax.bar(x - width/2, ppo_m, width, yerr=sem_ppo, capsize=3,
           label="Adaptive RL (defended-trained)",
           color="#2E86AB", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, beta_m, width, yerr=sem_beta, capsize=3,
           label="Static BETA (defended eval)",
           color="#E63946", edgecolor="black", linewidth=0.5)

    # Vertical separator between WADI and SWaT
    if n_wadi > 0 and n_swat > 0:
        ax.axvline(n_wadi - 0.5, color="black", linewidth=1.0, linestyle="-", alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Detector seed")
    ax.set_ylabel("Mean target-score degradation")
    ax.set_xticks(x)
    labels = [f"W{r['seed']}" for r in wadi_rows] + [f"S{r['seed']}" for r in swat_rows]
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate WADI / SWaT regions
    if n_wadi > 0:
        ax.text((n_wadi - 1) / 2, ax.get_ylim()[1] * 0.95, "GDN-WADI", ha="center",
                fontsize=8, fontweight="bold", alpha=0.7)
    if n_swat > 0:
        ax.text(n_wadi + (n_swat - 1) / 2, ax.get_ylim()[1] * 0.95, "GDN-SWaT", ha="center",
                fontsize=8, fontweight="bold", alpha=0.7)

    ppo_mean_n = ppo_m.mean()
    beta_mean_n = beta_m.mean()
    n_wins = sum(1 for p, b in zip(ppo_m, beta_m) if p > b)
    ax.text(0.98, 0.98,
            f"{n_total}-seed combined mean:\n  RL: {ppo_mean_n:+.2f}\n  BETA: {beta_mean_n:+.2f}\n  RL wins: {n_wins}/{n_total}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    plt.tight_layout()
    out = FIG_DIR / "fig_5seed_defended.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[fig] wrote {out}")


def fig_beta_undefended_concentration():
    """Show static BETA concentration on dominant target sensor per detector."""
    seeds = list(range(5))
    swat_concentration = [84, 100, 98, 100, 92]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(seeds))
    bars = ax.bar(x, swat_concentration, color="#E63946", edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Detector seed (GDN-SWaT)")
    ax.set_ylabel("Top-target concentration (\\%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.axhline(95, color="black", linewidth=0.5, linestyle="--", alpha=0.5,
               label=f"5-seed mean = 95\\%")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    for b, v in zip(bars, swat_concentration):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v}\\%", ha="center", fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "fig_beta_concentration.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[fig] wrote {out}")


def fig_defense_reduction():
    """Defense reduction factor across detectors (log scale)."""
    seeds_swat = [0, 1, 2, 3, 4]
    swat_reduction = [1500, 1523, 1578, 815, 3332]
    seeds_wadi_label = "WADI 4-seed"
    wadi_reduction = 377_000

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    x_swat = np.arange(len(seeds_swat))
    bars = ax.bar(x_swat, swat_reduction, color="#2E86AB", edgecolor="black", linewidth=0.5,
                  label="GDN-SWaT (per-seed)")
    ax.bar([len(seeds_swat) + 0.5], [wadi_reduction], color="#A23B72", edgecolor="black", linewidth=0.5,
           label="GDN-WADI (4-seed mean)")

    ax.set_yscale("log")
    ax.set_ylabel("BETA reduction factor under defense")
    ax.set_xticks(list(x_swat) + [len(seeds_swat) + 0.5])
    ax.set_xticklabels([f"S{s}" for s in seeds_swat] + [seeds_wadi_label], rotation=20, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.4, which="both")
    ax.legend(loc="upper left", fontsize=8)
    for b, v in zip(bars, swat_reduction):
        ax.text(b.get_x() + b.get_width()/2, v * 1.4, f"{v}\\,$\\times$", ha="center", fontsize=7)
    ax.text(len(seeds_swat) + 0.5, wadi_reduction * 1.4, f"$\\sim${wadi_reduction//1000}k$\\times$",
            ha="center", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "fig_defense_reduction.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    fig_5seed_defended()
    fig_beta_undefended_concentration()
    fig_defense_reduction()
    print("done")
