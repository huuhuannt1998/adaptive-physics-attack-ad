# adaptive-physics-attack-ad

Adaptive adversarial-attack research code for graph neural network (GNN) based ICS anomaly detectors.

## What's here

- **Static baseline attack.** Reproduction of a gradient-saliency + eigenvector-centrality + PGD attack pipeline on GDN / TopoGDN detectors.
- **Audit.** Quantifies how much of the published headline degradation depends on out-of-range test inputs combined with the attack's clamp-to-$[0,1]$ post-processing step.
- **Defense.** Inference-time input clipping at $[0,1]$ plus three threshold-calibration recipes (`val-99.5`, F1-optimal oracle, labeled calibration-split). The labeled cal-split recipe is the deployable one.
- **Adaptive RL attacker.** Behavior-cloning warm-start followed by PPO training under both undefended and defended distributions. Transformer-encoder policy, hybrid categorical + sign action space.
- **Cross-architecture characterization.** GDN and TopoGDN evaluated on WADI and SWaT; one characterized failure mode (TopoGDN-WADI persistent-homology feature discontinuity).

## Repository layout

```
.
├── attacks/                     # Static-attack implementations (BETA reproduction + variants)
├── attack_env/                  # PPO environment + Transformer policy network
├── defenses/                    # Input clipping + threshold-calibration recipes
├── loaders/                     # IPAL-compatible dataset loaders
├── repos/                       # Vendored upstream detector code
│   ├── GDN/                     # Deng & Hooi 2021, unmodified
│   ├── GDN_pyg1x/               # PyG 1.x compatibility shim
│   └── TopoGDN/                 # With PyG API patch
├── scripts/                     # Entry points (see below)
├── reports/                     # Per-experiment JSON outputs
│   └── calibration_split/       # Cal-split results (20 configs + summary)
└── data/                        # Dataset directory (datasets NOT included)
```

## Datasets

The canonical iTrust testbed datasets WADI (2017) and SWaT (2015) must be obtained from iTrust under their Data Use Agreement. Place pre-processed CSVs under `data/wadi/` and `data/swat/`. The IPAL-compatible loader expects the same structure as upstream GDN / TopoGDN; see `loaders/` for column conventions.

## Reproducing the headline results

Python 3.10 + PyTorch 2.2 (CPU sufficient on consumer hardware).

```bash
# 1. Train GDN on WADI, 5 seeds
bash scripts/train_gdn_wadi_5seeds.sh

# 2. Static attack on undefended detector
python scripts/eval_beta_swat.py --arch GDN --dataset wadi --seed 0

# 3. F1-optimal threshold under input clipping (oracle, non-deployable)
python scripts/find_f1_threshold.py --arch GDN --dataset wadi --seed 0 --defended

# 4. Deployable calibration via labeled cal-split (50/50, K=5)
python scripts/calibration_split_experiment.py --arch GDN --dataset wadi --seed 0
# Output: reports/calibration_split/gdn_wadi_seed0.json

# 5. Defense component ablation
python scripts/defense_ablation.py --arch GDN --dataset wadi

# 6. Adaptive RL attack
python scripts/bc_pretrain.py --dataset wadi --seed 0
python scripts/ppo_train.py --dataset wadi --seed 0 --defended

# 7. Per-window IPAL F1 evaluation under attack
python scripts/eval_defended_detector.py --arch GDN --dataset wadi --seed 0
```

## Calibration-split result

| Configuration | Oracle F1 (non-deployable) | val-99.5 F1 | **Cal-split F1 (deployable)** |
|---|---:|---:|---:|
| GDN-WADI     | 0.380 | 0.217 | **0.379** |
| GDN-SWaT     | 0.657 | 0.631 | **0.658** |
| TopoGDN-WADI | 0.412 | 0.283 | **0.417** |
| TopoGDN-SWaT | 0.420 | 0.218 | **0.418** |

Max $\lvert F1_\text{cal-split} - F1_\text{oracle} \rvert$ across all 20 (arch × dataset × seed) configurations: **0.012**. The labeled cal-split protocol — 50/50 stratified split of the clipped-test set, pick F1-optimal threshold on the calibration partition, report F1/precision/recall/FPR on the held-out partition, K=5 random splits per seed — closes the val→test transfer gap that the val-99.5 heuristic does not.

Per-seed numbers: `reports/calibration_split/summary.json`.

## Status

Work in progress. The paper writing is maintained separately; this repository tracks only the code, configs, and result artifacts.
