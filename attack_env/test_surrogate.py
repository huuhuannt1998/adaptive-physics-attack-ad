"""Smoke tests for the surrogate training pipeline."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from attack_env.grey_box import GreyBoxVictim, StandardizationStats
from attack_env.surrogate import MLPSurrogate, SurrogateTrainConfig, train_surrogate


def _identity_forecast(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :]


def _make_victim_and_loader(n_sensors=4, window=10, n_samples=64):
    stats = StandardizationStats(median=torch.zeros(n_sensors), iqr=torch.ones(n_sensors))
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    victim = GreyBoxVictim(_identity_forecast, stats, edge_index)

    torch.manual_seed(0)
    X = torch.randn(n_samples, window, n_sensors)
    loader = DataLoader(TensorDataset(X), batch_size=16, shuffle=True)
    return victim, loader, n_sensors, window


def test_train_surrogate_runs_and_loss_decreases():
    victim, loader, n_sensors, window = _make_victim_and_loader()
    surrogate = MLPSurrogate(n_sensors=n_sensors, window=window, hidden=64)
    cfg = SurrogateTrainConfig(epochs=8, lr=1e-2, log_every=1)

    trained, history = train_surrogate(victim, loader, surrogate, cfg)

    assert trained.training is False  # returned in eval mode
    losses = list(history.values())
    assert len(losses) >= 4
    # Loss should decrease meaningfully across epochs.
    assert losses[-1] < losses[0] * 0.5


def test_query_budget_caps_samples_used():
    victim, loader, n_sensors, window = _make_victim_and_loader(n_samples=128)
    surrogate = MLPSurrogate(n_sensors=n_sensors, window=window, hidden=32)
    cfg = SurrogateTrainConfig(epochs=10, lr=1e-3, query_budget=20, log_every=1)

    trained, history = train_surrogate(victim, loader, surrogate, cfg)
    # Pipeline should exit early; we cannot observe queries_used directly but the loss
    # history length must be tiny relative to unbounded.
    assert len(history) <= 3, f"expected very few steps under budget=20, got {len(history)}"


def test_surrogate_io_matches_victim_contract():
    """Surrogate must accept (B, W, N) and return (B, N) — same contract as victim.query."""
    n_sensors, window = 5, 100
    surrogate = MLPSurrogate(n_sensors=n_sensors, window=window)
    X = torch.randn(3, window, n_sensors)
    out = surrogate(X)
    assert out.shape == (3, n_sensors)


def test_surrogate_is_differentiable():
    """Phase 1 needs gradient flow through the surrogate. Confirm it backprops."""
    n_sensors, window = 5, 20
    surrogate = MLPSurrogate(n_sensors=n_sensors, window=window)
    X = torch.randn(2, window, n_sensors, requires_grad=True)
    out = surrogate(X).sum()
    out.backward()
    assert X.grad is not None
    assert X.grad.abs().sum() > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
