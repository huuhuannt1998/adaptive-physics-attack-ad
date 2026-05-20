"""
Surrogate training pipeline for grey-box adversarial attack.

Phase 0 spec (jrn_01KQGR0QSAXDXKWX31F3JAS49K, ratifying R-EXEC-6): the surrogate query
budget N is UNBOUNDED, equal to the number of training windows. The point of Phase 0
is end-to-end pipeline validation, not surrogate sample-efficiency. Phase 1 will reintroduce
a constrained N as part of the threat-model story.

The surrogate replicates the locked grey-box query interface (`(B, W, N) -> (B, N)`) so
it is a drop-in differentiable substitute the Phase 1 PPO policy can backprop through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SurrogateTrainConfig:
    """Defaults sized for the Phase 0 sanity check, not for sample-efficient Phase 1 training."""

    epochs: int = 10
    lr: float = 1e-3
    batch_size: Optional[int] = None  # use loader's native batch when None
    query_budget: Optional[int] = None  # None == unbounded (Phase 0 default)
    log_every: int = 50
    weight_decay: float = 0.0


class MLPSurrogate(nn.Module):
    """Cheap baseline surrogate: per-sensor MLP over the flattened window.

    Same I/O contract as `GreyBoxVictim.query`: takes (B, W, N), returns (B, N).
    Phase 1 will swap this for a GDN-shaped (or TopoGDN-shaped) surrogate via the
    `surrogate_factory` parameter to `train_surrogate`. This baseline exists so the
    pipeline runs end-to-end and so we can sanity-check that the surrogate actually
    fits the victim's residual signal at Phase 0.
    """

    def __init__(self, n_sensors: int, window: int, hidden: int = 256):
        super().__init__()
        self.n_sensors = n_sensors
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(window * n_sensors, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.n_sensors:
            raise ValueError(
                f"expected (B, W, N) with N={self.n_sensors}, got {tuple(x.shape)}"
            )
        return self.net(x.flatten(start_dim=1))


def _victim_query_callable(victim) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapter so `train_surrogate` works with any object exposing a `query(X) -> (B, N)` method."""
    if not hasattr(victim, "query"):
        raise TypeError(f"victim {victim!r} has no `query` method")
    return victim.query


def train_surrogate(
    victim,
    train_loader: Iterable,
    surrogate: nn.Module,
    config: SurrogateTrainConfig = SurrogateTrainConfig(),
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    """Train surrogate to mimic victim.query on windows from train_loader.

    Args:
        victim: object exposing `query(X: (B, W, N)) -> (B, N)`. Standard usage: a
            `GreyBoxVictim`, but any object with the same shape contract works.
        train_loader: iterable yielding either (X,) or (X, *) tuples where X is (B, W, N).
            Phase 1 will replace this with an attacker-side query buffer; for Phase 0 we
            iterate the entire victim training set.
        surrogate: nn.Module mapping (B, W, N) -> (B, N).
        config: hyperparameters; defaults are Phase 0-sized.
        device: training device.

    Returns:
        (trained surrogate, history dict {step: loss}).
    """
    device = torch.device(device)
    surrogate = surrogate.to(device).train()
    query_fn = _victim_query_callable(victim)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: dict[int, float] = {}
    step = 0
    queries_used = 0

    for epoch in range(config.epochs):
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            if config.query_budget is not None and queries_used + x.shape[0] > config.query_budget:
                # Phase 1 will hit this branch; Phase 0 default leaves query_budget=None.
                allowed = config.query_budget - queries_used
                if allowed <= 0:
                    return surrogate.eval(), history
                x = x[:allowed]

            with torch.no_grad():
                target = query_fn(x).detach()  # (B, N)

            pred = surrogate(x)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            queries_used += x.shape[0]
            if step % config.log_every == 0:
                history[step] = float(loss.item())
            step += 1

            if config.query_budget is not None and queries_used >= config.query_budget:
                return surrogate.eval(), history

    return surrogate.eval(), history
