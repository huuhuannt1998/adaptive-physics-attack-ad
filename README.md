# Effective Perturbation Budget (EPB): code and artifacts

Code, per-experiment result files, and the online supplement for:

> Huan Bui and Chenglong Fu. **Effective Perturbation Budget: Auditing
> Preprocessing-Induced Amplification in ICS Anomaly Detector Robustness.**
> IEEE International Performance, Computing, and Communications Conference
> (IPCCC), 2026.

**Online supplement** (the per-seed tables the paper refers to):
[`supplemental.pdf`](supplemental.pdf)

## Scope note: BETA is our reimplementation

No code artifact for BETA (Xaviar and Ardakanian, arXiv:2509.17987v1) was
available to us, so `attacks/beta.py` is an independent reimplementation
written from the paper text (paper Section V-A). The asymmetric post-PGD
`clip(., 0, 1)` that EPB audits is part of *this* reimplementation. Our
results describe this code and the evaluation convention it follows. They do
not establish that the degradation reported in the BETA paper has the same
cause; that would require BETA's original preprocessing.

All experiments target the gray-box, l_inf-bounded attacker of paper
Section III (epsilon = 0.1, sensor budget b <= 5). They should not be read as
claims about other threat models.

## What is in this repository

- **EPB audit.** Measures the realized detector-input displacement under the
  deployed preprocessing and splits it into an attack-induced part (A) and a
  clean/attacked path-mismatch part (M).
- **Path correction.** Applies the same clip to the clean and attacked paths
  (or a center-preserving projection), with threshold recalibration on the
  transformed validation data. No retraining.
- **Adaptive validation.** A gradient-free PPO attacker, random search, SPSA,
  and a decision-level attacker that minimizes the top-k detection score
  directly.
- **Generality checks.** A non-GNN LSTM detector, the HAI testbed, and three
  train-fit normalizations.

## Repository layout

```
.
├── attacks/          BETA reimplementation, FGSM, gradient-free black-box attackers
├── attack_env/       PPO environment, Transformer policy, query-and-clone surrogate
├── defenses/         Input clipping and threshold-calibration utilities
├── detectors/        LSTM forecaster (non-GNN detector for the generality check)
├── loaders/          WADI / SWaT loaders
├── repos/            Vendored upstream detectors (GDN, GDN_pyg1x, TopoGDN)
├── scripts/          Experiment entry points (see the table below)
├── reports/          Per-experiment JSON outputs; reports/ipal/ holds IPAL streams
└── supplemental.pdf  Online supplement
```

## Setup

Python 3.10, CPU only (all reported runs used a single Apple M4).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

The training shell scripts in `scripts/` set `PROJECT_ROOT` to the original
machine's path on their first lines. Edit it to your checkout before running
them.

## Datasets (not included)

- **WADI and SWaT** are distributed by iTrust (SUTD) under their data-use
  agreement. Place the preprocessed `train.csv` / `test.csv` under both
  `repos/GDN/data/{wadi,swat}/` and `repos/TopoGDN/data/{wadi,swat}/`,
  following the upstream GDN layout (each detector reads its own copy).
- **HAI** is public. Place release `hai-23.05` under `data/hai/hai-23.05/`.

## Reproducing the paper

Every script prints its options with `--help`. Outputs are written to the
`--out` path; the files already in `reports/` are the ones behind the paper.

| Paper result | Script(s) | Output in `reports/` |
|---|---|---|
| Train detectors (5 seeds each) | `scripts/train_gdn_wadi_5seeds.sh`, `scripts/train_topogdn_wadi_5seeds.sh`, and the SWaT variants | checkpoints under `repos/*/pretrained/` (not committed) |
| Sec. V-B: BETA degradation and per-trial EPB | `scripts/eval_beta_swat.py --arch GDN --dataset wadi --seed 1 --out reports/beta_wadi_seed1_epb.json` | `beta_*_seed*.json`, `beta_*_epb.json` |
| Sec. V-C: EPB decomposition into T, A, M | `scripts/epb_decomposition.py --dataset wadi --seed 0 --out reports/epb_decomp_wadi_seed0.json` | `epb_decomp_{wadi,swat}_seed{0-4}.json` |
| Sec. V-C: detector-level F1 under BETA | `scripts/multi_metric_eval.py --detector gdn --dataset wadi --seed 0` | `mm_beta_*_undef.json` |
| Sec. V-E: normalization invariance | `scripts/preprocessing_variant_analysis.py` | `preprocessing_variants.json` |
| Sec. V-E: LSTM detector | `scripts/train_lstm_forecaster.py`, then `scripts/lstm_ood_clamp_audit.py --lstm <ckpt>` | `lstm_audit_{wadi,swat}_seed0.json` |
| Sec. V-E: HAI testbed | `scripts/hai_audit.py --release hai-23.05 --out reports/hai_audit.json` | `hai_audit.json` |
| Table I: correction baselines | `scripts/correction_baselines.py --dataset wadi --seed 0 --out reports/corr_baselines_wadi_seed0.json` | `corr_baselines_{wadi,swat}_seed0.json` |
| Table II: IPAL F1, clean vs. BETA (defended) | `scripts/find_f1_threshold.py --defended`, `scripts/ipal_convert.py` | `ipal/*_defended_f1opt.*` |
| Sec. VI-B: component ablation | `scripts/defense_ablation.py --dataset wadi --seed 0 --variant {full,clip_only,threshold_only} --out ...` | `ablation/` |
| Sec. VI-C: TopoGDN-WADI interpolation sweep | `scripts/topogdn_interp_sweep.py --dataset wadi --seed 0 --out reports/topogdn_interp_wadi_seed0.json` | `topogdn_interp_wadi_seed0.json` |
| Sec. VI-D: labeled calibration split | `scripts/calibration_split_experiment.py --arch GDN --dataset wadi --seed 0` | `calibration_split/` |
| Sec. VI-D: label-free benign / EVT thresholds | `scripts/calibration_eval.py --dataset wadi --seed 0 --out ...` | `calibration_{wadi,swat}_seed0.json` |
| Sec. VI-D: effect of clipping on anomaly recall | `scripts/clipping_side_effects.py --dataset wadi --seed 0 --out ...` | `clip_sideeffects_{wadi,swat}_seed0.json` |
| Sec. VI-D: systems overhead | `scripts/systems_overhead.py --dataset wadi --seed 0 --out ...` | `overhead_{wadi,swat}_seed0.json` |
| Sec. VII: PPO attacker | `scripts/generate_bc_dataset.py`, `scripts/bc_pretrain.py`, `scripts/ppo_train.py --defended`, `scripts/eval_defended_detector.py` (TopoGDN: `*_topogdn.py` variants) | `ppo_train_defended_*`, `eval_defended_*` |
| Sec. VII: random search and SPSA | `scripts/multi_metric_eval.py --defended --attacker {random_search,spsa}`, `scripts/eval_blackbox_attacks.py --defended` | `mm_{random_search,spsa}_*_def.json`, `blackbox_attacker_summary.json` |
| Sec. VII: b-vs-k ablation | `scripts/b_vs_k_ablation.py --dataset wadi --out reports/b_vs_k_wadi.json` | `b_vs_k_wadi.json` |
| Sec. VII: decision-level attacker | `scripts/end_to_end_attack.py --dataset wadi --seed 0 --threshold {f1opt,benign,evt} --out ...` | `e2e_*.json` |
| Table III: matched-window comparison | `scripts/matched_attack_comparison.py --dataset wadi --seed 0 --out reports/matched_wadi_seed0.json` | `matched_{wadi,swat}_seed{0-4}.json` |

Some scripts also accept `--hk-*` options. These belong to an exploratory
TopoGDN extension that is not part of this release and need a module that is
not included. Nothing in the paper uses them.

## Calibration-split result (Sec. VI-D)

| Configuration | Oracle F1 (non-deployable) | val-99.5 F1 | Cal-split F1 (deployable) |
|---|---:|---:|---:|
| GDN-WADI     | 0.380 | 0.217 | **0.379** |
| GDN-SWaT     | 0.657 | 0.631 | **0.658** |
| TopoGDN-WADI | 0.412 | 0.283 | **0.417** |
| TopoGDN-SWaT | 0.420 | 0.218 | **0.418** |

Across all 20 (architecture x dataset x seed) configurations, the labeled
50/50 calibration split stays within 0.012 F1 of the oracle. Per-seed numbers
are in `reports/calibration_split/summary.json`.

## License

Code in this repository is released under the [MIT License](LICENSE).
Vendored third-party code under `repos/` keeps its upstream license (see
`LICENSE` for details).
