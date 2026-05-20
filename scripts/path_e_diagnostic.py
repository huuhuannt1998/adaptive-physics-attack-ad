"""
Path E diagnostic: verify TopoGDN's PH + TCN layers actually contribute under PyG 2.5.

Authorized via jrn_01KQM8GXE9NT4J2H7P2BZSANQJ in response to chk_01KQM7WJA7GH2SRG3CSSCF069P.

Tests:
  1. _slice_dict semantics — PASSED by inspection (PyG 2.x's _slice_dict carries the same
     per-graph slice indices as PyG 1.x's __slices__; just tensor-typed instead of list).
  2. Forward pass with hooks on TCN and PH layers; compare:
     - TCN output vs input (statistical difference confirms TCN is firing)
     - PH output (topoOut in GNNLayer.forward) magnitude vs the residual connection target
       (`out += topoOut`) — if topoOut is near-zero, PH adds nothing
  3. Gradient hook on PH output to confirm gradients propagate during a backward pass.
  4. Differential ablation: forward pass with use_topo=False (PH disabled) vs use_topo=True
     on the SAME trained weights; outputs should differ if PH is functional.

Outputs printed as JSON for easy ingestion into the journal entry.
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
CHECKPOINT = TOPOGDN_REPO / "pretrained/wadi_seed0/best_05|01-12:01:46.pt"


def setup_topogdn():
    """Initialize TopoGDN's pipeline + load seed 0 checkpoint via Main.run path (no train)."""
    sys.path.insert(0, str(TOPOGDN_REPO))
    os.chdir(TOPOGDN_REPO)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_config = {
        "batch": 32, "epoch": 50, "slide_win": 100, "dim": 128,
        "slide_stride": 10, "comment": "path_e_diag", "seed": 0,
        "out_layer_num": 1, "out_layer_inter_dim": 256,
        "decay": 0.0, "val_ratio": 0.1, "topk": 30,
        "use_tcn": True, "use_topo": True, "model": "GDN",
    }
    env_config = {
        "save_path": "wadi_seed0", "dataset": "wadi", "report": "best",
        "device": "cpu", "load_model_path": str(CHECKPOINT.relative_to(TOPOGDN_REPO).as_posix()),
    }
    from main import Main
    m = Main(train_config, env_config, debug=False)
    m.model.load_state_dict(torch.load(env_config["load_model_path"], map_location="cpu"))
    m.model.eval()
    return m


def grab_one_batch(loader):
    for batch in loader:
        return batch
    raise RuntimeError("loader exhausted")


def stat(t: torch.Tensor) -> dict:
    t = t.detach()
    return {
        "shape": list(t.shape),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if t.numel() > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "abs_mean": float(t.abs().mean().item()),
        "fraction_zero": float((t.abs() < 1e-8).float().mean().item()),
        "has_nan": bool(torch.isnan(t).any().item()),
    }


def run_diagnostic():
    m = setup_topogdn()
    model = m.model
    test_loader = m.test_dataloader

    x, y, labels, edge_index = grab_one_batch(test_loader)
    x = x.float()
    y = y.float()
    edge_index = edge_index.float()

    results: dict = {"checkpoint": str(CHECKPOINT), "input_shape": list(x.shape)}

    # ─── Test 2a: TCN (MSConv) output statistics ───────────────────────
    tcn_input = x.clone()
    tcn_module = model.MSConv
    if tcn_module is None:
        results["tcn"] = {"status": "DISABLED — model.MSConv is None"}
    else:
        with torch.no_grad():
            tcn_output = tcn_module(tcn_input)
        results["tcn"] = {
            "status": "active" if tcn_module is not None else "disabled",
            "input_stats": stat(tcn_input),
            "output_stats": stat(tcn_output),
            "delta_abs_mean": stat(tcn_output - tcn_input)["abs_mean"],
        }

    # ─── Test 2b: PH layer (topoOut) statistics via forward hook ───────
    ph_outputs = []
    handles = []

    def ph_hook_factory(layer_name):
        def hook(module, inp, out):
            # `out` is whatever the topoPooling.forward returns: tuple (topoOut, _, _)
            if isinstance(out, tuple) and len(out) >= 1:
                ph_outputs.append((layer_name, out[0].detach().clone()))
            else:
                ph_outputs.append((layer_name, out.detach().clone() if torch.is_tensor(out) else None))
        return hook

    for i, gnn_layer in enumerate(model.gnn_layers):
        if hasattr(gnn_layer, "topoPooling") and getattr(gnn_layer, "use_topo", False):
            h = gnn_layer.topoPooling.register_forward_hook(ph_hook_factory(f"gnn_layers[{i}].topoPooling"))
            handles.append(h)

    if not handles:
        results["ph"] = {"status": "DISABLED — no use_topo gnn_layer has topoPooling"}
    else:
        with torch.no_grad():
            forecast, _ = model(x)
        results["ph"] = {"status": "active", "forward_output_stats": stat(forecast), "ph_layer_outputs": []}
        for layer_name, ph_out in ph_outputs:
            if ph_out is None:
                results["ph"]["ph_layer_outputs"].append({"layer": layer_name, "status": "non-tensor return"})
            else:
                results["ph"]["ph_layer_outputs"].append({
                    "layer": layer_name,
                    "stats": stat(ph_out),
                })

    for h in handles:
        h.remove()

    # ─── Test 2c: Differential ablation — same weights, use_topo False vs True ──
    # Manually patch use_topo on every GNNLayer to False, forward again, compare to PH-on output.
    use_topo_original = []
    for gnn_layer in model.gnn_layers:
        use_topo_original.append(gnn_layer.use_topo)
        gnn_layer.use_topo = False
    with torch.no_grad():
        forecast_no_ph, _ = model(x)
    for gnn_layer, orig in zip(model.gnn_layers, use_topo_original):
        gnn_layer.use_topo = orig
    with torch.no_grad():
        forecast_with_ph, _ = model(x)

    delta = forecast_with_ph - forecast_no_ph
    results["differential_ph_ablation"] = {
        "with_ph_stats": stat(forecast_with_ph),
        "no_ph_stats": stat(forecast_no_ph),
        "delta_stats": stat(delta),
        "delta_relative_to_output": float(delta.abs().mean().item() / (forecast_with_ph.abs().mean().item() + 1e-12)),
    }

    # ─── Test 3: Gradient hook on PH output ────────────────────────────
    grads_per_layer: list = []
    grad_handles = []

    def grad_hook_factory(layer_name):
        def hook(grad):
            grads_per_layer.append((layer_name, grad.detach().clone()))
            return grad
        return hook

    # Re-register: capture gradient flowing back through topoOut. Enable training mode.
    model.train()
    captured_topo_outputs: list = []

    def capture_topo_out_factory(layer_name):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
                t = out[0]
                if t.requires_grad:
                    t.register_hook(grad_hook_factory(layer_name))
                captured_topo_outputs.append((layer_name, t))
        return hook

    for i, gnn_layer in enumerate(model.gnn_layers):
        if hasattr(gnn_layer, "topoPooling") and getattr(gnn_layer, "use_topo", False):
            grad_handles.append(
                gnn_layer.topoPooling.register_forward_hook(capture_topo_out_factory(f"gnn_layers[{i}].topoPooling"))
            )

    forecast, _ = model(x)
    loss = ((forecast - y) ** 2).mean()
    loss.backward()

    results["ph_gradient"] = {
        "layers_observed": len(grads_per_layer),
        "grads": [
            {"layer": layer_name, "stats": stat(g)} for layer_name, g in grads_per_layer
        ],
    }
    for h in grad_handles:
        h.remove()

    # ─── Test 3b: Compare gradient magnitudes across layer parameters ────
    # If PH params are getting much smaller gradients than non-PH params, PH is effectively frozen.
    param_grad_summary: list = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        bucket = "topo" if "topoPooling" in name else ("tcn" if "MSConv" in name else "other")
        param_grad_summary.append({
            "name": name,
            "bucket": bucket,
            "param_abs_mean": float(p.detach().abs().mean().item()),
            "grad_abs_mean": float(p.grad.detach().abs().mean().item()),
            "grad_to_param_ratio": float(
                (p.grad.detach().abs().mean() / (p.detach().abs().mean() + 1e-12)).item()
            ),
        })

    # Aggregate by bucket.
    buckets: dict[str, list[float]] = {}
    for entry in param_grad_summary:
        buckets.setdefault(entry["bucket"], []).append(entry["grad_abs_mean"])
    bucket_summary = {
        b: {
            "n_params": len(vals),
            "grad_abs_mean_median": float(np.median(vals)) if vals else None,
            "grad_abs_mean_max": float(max(vals)) if vals else None,
            "grad_abs_mean_min": float(min(vals)) if vals else None,
        }
        for b, vals in buckets.items()
    }
    results["param_gradients"] = {
        "by_bucket": bucket_summary,
        "per_param_top10_smallest_grad_abs_mean": sorted(param_grad_summary, key=lambda d: d["grad_abs_mean"])[:10],
        "per_param_top10_largest_grad_abs_mean": sorted(param_grad_summary, key=lambda d: -d["grad_abs_mean"])[:10],
    }

    return results


def main():
    result = run_diagnostic()
    out_path = PROJECT_ROOT / "reports" / "path_e_diagnostic_topogdn_seed0.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"\n[write] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
