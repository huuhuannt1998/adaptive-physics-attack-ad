"""
P1.T3: PPO actor-critic with Transformer encoder for the adaptive attack policy.

Per Backbrief approval (jrn_01KQPNGW00RMTNZYD6XDZA8AZ9) and mission spec
(mis_01KQPMGFEYSC2JG9JDP5WAZ4MA):
  - Transformer encoder: d_model=128, 2 layers, 4 attention heads, ~150K params total
  - Hybrid action: categorical (n_sensors classes including no-op) + Gaussian (W-dim δ)
  - Dual head sharing the encoder; separate value head for PPO critic

State input from MDP env (`attack_env/mdp.py`):
  - current_window: (1, N, W) per-sensor history
  - target_score_history: (H,) recent target-sensor scores
  - target_idx: int (one-hot encoded)
  - budget_remaining: int (scalar feature)
  - step: int (scalar feature)
  - clean_target_score: float (scalar feature)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    n_sensors: int = 127
    window_size: int = 100
    target_score_history_len: int = 5
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    perturbation_dim: int = 100  # = window_size
    epsilon_pgd: float = 0.1
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 0.5


def _per_sensor_window_embed(window: torch.Tensor, d_model: int, conv: nn.Module) -> torch.Tensor:
    """Embed a (B, N, W) window into (B, N, d_model) per-sensor tokens via 1D conv."""
    B, N, W = window.shape
    flat = window.reshape(B * N, 1, W)  # (B*N, 1, W)
    embedded = conv(flat)  # (B*N, d_model, W')
    pooled = embedded.mean(dim=-1)  # (B*N, d_model)
    return pooled.reshape(B, N, d_model)


class AdaptiveAttackPolicy(nn.Module):
    """Transformer-encoder actor-critic policy for the adaptive attack MDP.

    Forward output:
      - cat_logits: (B, n_sensors) — categorical action logits (last index = no-op)
      - delta_mu: (B, W) — Gaussian mean for δ_t conditioned on (state, sampled categorical)
      - delta_log_std: (B, W) — Gaussian log-std (state-independent during BC; full network during PPO)
      - value: (B,) — V(s) for PPO critic
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Per-sensor window encoder: 1D conv front-end → d_model.
        self.window_conv = nn.Sequential(
            nn.Conv1d(1, config.d_model // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(config.d_model // 2, config.d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        # Sensor-position embedding (learned, identifies which sensor each token represents).
        self.sensor_pos_emb = nn.Embedding(config.n_sensors, config.d_model)

        # Compact state features (target_idx one-hot, target_score_history, budget, step, clean_score).
        # Project to d_model, prepend as a "summary" token so attention can read it.
        self.summary_proj = nn.Linear(
            config.n_sensors  # one-hot target
            + config.target_score_history_len  # score history
            + 3,  # budget, step, clean_score
            config.d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 2,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Heads.
        # F2 (jrn_01KQPR5QSJBEHD1Q9H1WAXTF0C): drop Gaussian; use per-channel sign logits.
        # Action perturbation = sign(sigmoid_logit > 0.5) × epsilon_pgd. During BC, target is
        # sign(BETA_delta > 0); loss is per-channel BCEWithLogits. During PPO sampling, sample
        # Bernoulli per channel from sigmoid(logit). Magnitude is FIXED at epsilon_pgd.
        self.cat_head = nn.Linear(config.d_model, config.n_sensors)
        self.sign_head = nn.Linear(config.d_model, config.perturbation_dim)
        self.value_head = nn.Linear(config.d_model, 1)

    def encode(self, states: dict) -> torch.Tensor:
        """Returns (B, d_model) — pooled summary token after Transformer encoding."""
        cw = states["current_window"]  # (B, N, W)
        if cw.dim() == 3 and cw.shape[0] != self.config.window_size:
            # already (B, N, W)
            pass
        elif cw.dim() == 4:
            cw = cw.squeeze(1)  # (1, 1, N, W) → (1, N, W) etc.

        B = cw.shape[0]
        N = cw.shape[1]
        device = cw.device

        # Per-sensor tokens.
        sensor_tokens = _per_sensor_window_embed(cw, self.config.d_model, self.window_conv)  # (B, N, d_model)
        sensor_pos = self.sensor_pos_emb(torch.arange(N, device=device).unsqueeze(0).expand(B, N))
        sensor_tokens = sensor_tokens + sensor_pos

        # Summary token.
        target_one_hot = F.one_hot(states["target_idx"], num_classes=self.config.n_sensors).float()  # (B, N)
        score_hist = states["target_score_history"]  # (B, H)
        budget = states["budget_remaining"].float().unsqueeze(-1)  # (B, 1)
        step = states["step"].float().unsqueeze(-1)  # (B, 1)
        clean = states["clean_target_score"].float().unsqueeze(-1)  # (B, 1)
        summary_in = torch.cat([target_one_hot, score_hist, budget, step, clean], dim=-1)  # (B, ...)
        summary_token = self.summary_proj(summary_in).unsqueeze(1)  # (B, 1, d_model)

        seq = torch.cat([summary_token, sensor_tokens], dim=1)  # (B, N+1, d_model)
        encoded = self.encoder(seq)
        return encoded[:, 0, :]  # summary-position output

    def forward(self, states: dict) -> dict:
        h = self.encode(states)  # (B, d_model)
        cat_logits = self.cat_head(h)  # (B, N)
        sign_logits = self.sign_head(h)  # (B, W) — per-channel sign logits
        # Deterministic action: positive logit → +ε, negative → -ε. tanh provides smooth approx
        # for forward inference path (so delta_mu is differentiable for training).
        delta_mu = torch.tanh(sign_logits) * self.config.epsilon_pgd  # (B, W), bounded to ±ε
        value = self.value_head(h).squeeze(-1)
        return {
            "cat_logits": cat_logits,
            "sign_logits": sign_logits,
            "delta_mu": delta_mu,
            "value": value,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def collate_states(state_list: list[dict]) -> dict:
    """Stack a list of state dicts into a batched dict."""
    out = {}
    out["current_window"] = torch.stack(
        [s["current_window"].squeeze(0) if s["current_window"].dim() == 3 and s["current_window"].shape[0] == 1
         else s["current_window"] for s in state_list], dim=0
    )
    out["target_score_history"] = torch.stack([s["target_score_history"] for s in state_list], dim=0)
    out["target_idx"] = torch.tensor([s["target_idx"] for s in state_list], dtype=torch.long)
    out["budget_remaining"] = torch.tensor([s["budget_remaining"] for s in state_list], dtype=torch.long)
    out["step"] = torch.tensor([s["step"] for s in state_list], dtype=torch.long)
    out["clean_target_score"] = torch.tensor([s["clean_target_score"] for s in state_list], dtype=torch.float32)
    return out
