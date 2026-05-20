"""
P1.T4: Behavior cloning pretraining.

Trains the AdaptiveAttackPolicy on (state, BETA-action) pairs collected by
generate_bc_dataset.py. Loss = CE(cat_logits, target_cat) + α * MSE(delta_mu, target_delta).

Target per mission spec: BC alone achieves ≥80% of BETA's mean target-score
degradation on a held-out validation slice. If less, escalate before launching PPO.

Brain checkpoint trigger: surface at completion with BC-alone degradation %.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attack_env.policy import AdaptiveAttackPolicy, PolicyConfig, collate_states  # noqa: E402


class BCDataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _collate_pairs(batch: list[dict]) -> dict:
    states = [b["state"] for b in batch]
    actions = [b["action"] for b in batch]
    state_batch = collate_states(states)
    cat_action = torch.tensor([a["categorical"] for a in actions], dtype=torch.long)
    delta_action = torch.stack([a["delta"] for a in actions], dim=0)
    return {
        "state": state_batch,
        "action_cat": cat_action,
        "action_delta": delta_action,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reports/bc_dataset_gdn_wadi_seed0.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-sign", type=float, default=1.0,
                        help="weight on per-channel sign BCE relative to categorical CE")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out-checkpoint", default="reports/bc_policy_gdn_wadi_seed0.pt")
    parser.add_argument("--out-report", default="reports/bc_pretrain_report.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = PROJECT_ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    pairs: list[dict] = torch.load(dataset_path, weights_only=False)
    print(f"[load] {len(pairs)} BC pairs from {dataset_path}")

    # Determine config dimensions from the first pair.
    first_state = pairs[0]["state"]
    first_action = pairs[0]["action"]
    n_sensors = first_state["current_window"].shape[1]
    window_size = first_state["current_window"].shape[2]
    H = first_state["target_score_history"].shape[0]
    W = first_action["delta"].shape[0]
    print(f"[shape] N={n_sensors}, W={window_size}, H={H}, perturbation_dim={W}")

    cfg = PolicyConfig(
        n_sensors=n_sensors, window_size=window_size, target_score_history_len=H,
        d_model=128, n_layers=2, n_heads=4, dropout=0.2, perturbation_dim=W,
        epsilon_pgd=0.1,
    )
    policy = AdaptiveAttackPolicy(cfg)
    print(f"[policy] {policy.num_parameters():,} params")

    # Train/val split.
    rng = random.Random(args.seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_set = Subset(BCDataset(pairs), train_indices)
    val_set = Subset(BCDataset(pairs), val_indices)
    print(f"[split] train={len(train_set)}, val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_pairs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_pairs)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_cat_acc": [], "val_sign_acc": []}
    best_val_cat_acc = -1.0
    best_state_dict = None
    best_epoch = -1

    for epoch in range(args.epochs):
        policy.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = policy(batch["state"])
            loss_cat = F.cross_entropy(out["cat_logits"], batch["action_cat"])
            # F2 (jrn_01KQPR5QSJBEHD1Q9H1WAXTF0C): per-channel sign BCE in place of Gaussian MSE.
            sign_target = (batch["action_delta"] > 0).float()  # (B, W)
            loss_sign = F.binary_cross_entropy_with_logits(out["sign_logits"], sign_target)
            loss = loss_cat + args.alpha_sign * loss_sign
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        policy.eval()
        val_losses = []
        cat_correct = 0
        cat_total = 0
        sign_correct = 0
        sign_total = 0
        with torch.no_grad():
            for batch in val_loader:
                out = policy(batch["state"])
                loss_cat = F.cross_entropy(out["cat_logits"], batch["action_cat"])
                sign_target = (batch["action_delta"] > 0).float()
                loss_sign = F.binary_cross_entropy_with_logits(out["sign_logits"], sign_target)
                val_losses.append((loss_cat + args.alpha_sign * loss_sign).item())
                pred_cat = out["cat_logits"].argmax(dim=-1)
                cat_correct += (pred_cat == batch["action_cat"]).sum().item()
                cat_total += batch["action_cat"].numel()
                pred_sign = (out["sign_logits"] > 0).float()
                sign_correct += (pred_sign == sign_target).sum().item()
                sign_total += sign_target.numel()

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_cat_acc = cat_correct / max(cat_total, 1)
        val_sign_acc = sign_correct / max(sign_total, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cat_acc"].append(val_cat_acc)
        history["val_sign_acc"].append(val_sign_acc)

        if val_cat_acc > best_val_cat_acc:
            best_val_cat_acc = val_cat_acc
            best_state_dict = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_cat_acc={val_cat_acc:.3f}  val_sign_acc={val_sign_acc:.3f}")

    out_ckpt = PROJECT_ROOT / args.out_checkpoint if not Path(args.out_checkpoint).is_absolute() else Path(args.out_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Save best-val checkpoint (D3 gate is on cat-head accuracy; best-val warm-start matters for PPO).
    save_state = best_state_dict if best_state_dict is not None else policy.state_dict()
    torch.save({
        "policy_state_dict": save_state,
        "config": vars(cfg),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_cat_acc": best_val_cat_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_cat_acc": history["val_cat_acc"][-1],
        "final_val_sign_acc": history["val_sign_acc"][-1],
    }, out_ckpt)
    print(f"[write] BC policy checkpoint (best-val from epoch {best_epoch}, val_cat_acc={best_val_cat_acc:.3f}) -> {out_ckpt}")

    out_report = PROJECT_ROOT / args.out_report if not Path(args.out_report).is_absolute() else Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps({
        "n_pairs": len(pairs),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "epochs": args.epochs,
        "history": history,
        "policy_params": policy.num_parameters(),
        "config": vars(cfg),
    }, indent=2, default=str))
    print(f"[write] BC report -> {out_report}")


if __name__ == "__main__":
    main()
