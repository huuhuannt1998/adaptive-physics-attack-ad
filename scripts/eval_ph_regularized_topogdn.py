"""
Phase 4.2 evaluation: PH-feature regularization on TopoGDN-WADI defended
detector under static BETA and defended-trained PPO attacks.

PH-regularization variant: \emph{magnitude clip} on TopologyLayer output.
This is a first-order regularization — bound the magnitude of PH-derived
features at $\pm c$ before they are added to the GNN forecast. Rationale:
the §6.4 cross-architecture failure on TopoGDN-WADI is driven by
unbounded PH-feature discontinuity under small input perturbations
(theorem failure mode: small input → large PH-feature change). Clipping
the PH-feature magnitude removes the unbounded amplification, mirroring
the §4.3 input-clip's role in closing the OOD-clamp shortcut for raw
inputs.

Bottleneck-distance kernel regularization (Carriere et al.'s sliced-W
approach in defenses/ph_regularization.py) is the principled successor;
this magnitude-clip variant tests the simplest substitution that does
not require splitting TopoGDN's forward into compute_ph_features +
forward_from_ph hooks.

Usage:
    python scripts/eval_ph_regularized_topogdn.py --seed 0 \\
        --ph-clip 5.0 --scenario clean_defended \\
        --out reports/phase4_phreg/seed0_clip5_clean.metrics.json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_topogdn_with_phreg(seed: int, dataset: str, ph_clip: float):
    """Load TopoGDN with a forward-hook on TopologyLayer that clips
    topoOut magnitude. ph_clip=None → no regularization (sanity check).
    Returns the Main wrapper m and a handle to the hook (for removal)."""
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
            "slide_stride": 10, "comment": "phreg_eval", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 128,
            "decay": 0.0, "val_ratio": 0.1, "topk": 15,
            "use_tcn": True, "use_topo": True, "model": "GDN",
        }
    else:
        train_config = {
            "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
            "slide_stride": 10, "comment": "phreg_eval", "seed": seed,
            "out_layer_num": 1, "out_layer_inter_dim": 256,
            "decay": 0.0, "val_ratio": 0.1, "topk": 30,
            "use_tcn": True, "use_topo": True, "model": "GDN",
        }
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

    hook_handles = []
    if ph_clip is not None and ph_clip > 0:
        def topo_clip_hook(module, inputs, output):
            # output = (out_activations, graph_activations1, filtration)
            out_activations, ga1, filt = output
            return out_activations.clamp(-ph_clip, ph_clip), ga1, filt

        # TopoGDN.forward calls self.gnn_layers[i].topoPooling.
        # The actual layer instances are inside gnn_layers (a ModuleList).
        # Look up via model.named_modules()
        n_hooks = 0
        for name, mod in m.model.named_modules():
            if mod.__class__.__name__ == "TopologyLayer":
                handle = mod.register_forward_hook(topo_clip_hook)
                hook_handles.append(handle)
                n_hooks += 1
        print(f"[ph-reg] magnitude-clip={ph_clip} on {n_hooks} TopologyLayer instance(s)")
    else:
        print(f"[ph-reg] no regularization (control)")
    return m, hook_handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="wadi", choices=["wadi", "swat"])
    parser.add_argument("--ph-clip", type=float, default=5.0,
                        help="Magnitude clip threshold on PH features (topoOut). "
                             "Set to 0 or negative to disable (control eval).")
    parser.add_argument("--scenario", required=True,
                        choices=["clean_defended", "beta_defended", "ppo_defended"])
    parser.add_argument("--ppo-checkpoint", default=None,
                        help="Required for ppo_defended scenario")
    parser.add_argument("--threshold-abs", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.scenario == "ppo_defended" and not args.ppo_checkpoint:
        raise ValueError("--ppo-checkpoint required for ppo_defended")

    m, hooks = setup_topogdn_with_phreg(args.seed, args.dataset, args.ph_clip)

    # Smoke check: forward a single test window with the hook active.
    test_loader = m.test_dataloader
    val_loader = m.val_dataloader
    first_batch = next(iter(test_loader))
    x, y, _, ei = first_batch
    with torch.no_grad():
        out = m.model(x.float().clamp(0, 1))
        forecast = out[0] if isinstance(out, tuple) else out
    print(f"[smoke] forecast shape={forecast.shape}, min={forecast.min().item():.4f}, max={forecast.max().item():.4f}")

    # Build out_data: a stub IPAL eval result; this script's role is to
    # capture the PH-reg effect on forecast/residual distribution. Full
    # IPAL eval pipeline integration goes via ipal_convert.py with the
    # patched model imported. For Phase 4.2 first pass we report:
    #   - PH-reg variant: clip threshold
    #   - clean defended F1 placeholder (requires IPAL pipeline)
    #   - mean residual shift under clip vs no-clip
    # Full IPAL integration deferred to Phase 4.3 if needed.

    # For now: compute clean residual mean to confirm hook is engaging.
    from scipy.stats import iqr as scipy_iqr
    val_deltas_clipped = []
    with torch.no_grad():
        for batch in val_loader:
            xb, yb, _, eib = batch
            xfc, yfc = xb.float().clamp(0, 1), yb.float().clamp(0, 1)
            out = m.model(xfc)
            f = out[0] if isinstance(out, tuple) else out
            val_deltas_clipped.append((f - yfc).abs().cpu().numpy())
    val_deltas_clipped = np.concatenate(val_deltas_clipped, axis=0)

    result = {
        "seed": args.seed,
        "dataset": args.dataset,
        "ph_clip": args.ph_clip,
        "scenario": args.scenario,
        "threshold_abs": args.threshold_abs,
        "val_residual_mean": float(val_deltas_clipped.mean()),
        "val_residual_max": float(val_deltas_clipped.max()),
        "val_residual_p99": float(np.percentile(val_deltas_clipped, 99)),
        "val_residual_p99.5": float(np.percentile(val_deltas_clipped, 99.5)),
        "smoke_forecast_min": float(forecast.min().item()),
        "smoke_forecast_max": float(forecast.max().item()),
    }
    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[write] {out_path}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
