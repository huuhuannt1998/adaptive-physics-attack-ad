"""Smoke tests for P1.T3 policy network."""
from __future__ import annotations

import pytest
import torch

from attack_env.policy import (
    AdaptiveAttackPolicy,
    PolicyConfig,
    collate_states,
)


def _make_state(B=2, N=8, W=20, H=3):
    return {
        "current_window": torch.randn(B, N, W),
        "target_score_history": torch.randn(B, H),
        "target_idx": torch.randint(0, N, (B,)),
        "budget_remaining": torch.randint(0, 5, (B,)),
        "step": torch.randint(0, 5, (B,)),
        "clean_target_score": torch.randn(B),
    }


def test_policy_forward_output_shapes():
    cfg = PolicyConfig(n_sensors=8, window_size=20, target_score_history_len=3,
                       d_model=64, n_layers=1, n_heads=2, perturbation_dim=20)
    policy = AdaptiveAttackPolicy(cfg)
    state = _make_state(B=2, N=8, W=20, H=3)
    out = policy(state)
    assert out["cat_logits"].shape == (2, 8)
    assert out["sign_logits"].shape == (2, 20)
    assert out["delta_mu"].shape == (2, 20)
    assert out["value"].shape == (2,)


def test_delta_mu_bounded_by_epsilon():
    cfg = PolicyConfig(n_sensors=8, window_size=20, target_score_history_len=3,
                       d_model=64, n_layers=1, n_heads=2, perturbation_dim=20,
                       epsilon_pgd=0.1)
    policy = AdaptiveAttackPolicy(cfg)
    state = _make_state(B=4, N=8, W=20, H=3)
    out = policy(state)
    assert out["delta_mu"].abs().max() <= 0.1 + 1e-6


def test_param_count_is_compact_target_150K():
    """Mission target: ~150K params total at full N=127, W=100, d_model=128."""
    cfg = PolicyConfig(n_sensors=127, window_size=100, target_score_history_len=5,
                       d_model=128, n_layers=2, n_heads=4, perturbation_dim=100)
    policy = AdaptiveAttackPolicy(cfg)
    n_params = policy.num_parameters()
    print(f"  param count = {n_params}")
    # Target ~150K; allow generous tolerance.
    assert 100_000 <= n_params <= 600_000, f"got {n_params}"


def test_collate_states_batches_correctly():
    states = [
        {
            "current_window": torch.randn(1, 8, 20),
            "target_score_history": torch.randn(3),
            "target_idx": 3,
            "budget_remaining": 5,
            "step": 0,
            "clean_target_score": 0.5,
        }
        for _ in range(4)
    ]
    batched = collate_states(states)
    assert batched["current_window"].shape == (4, 8, 20)
    assert batched["target_score_history"].shape == (4, 3)
    assert batched["target_idx"].shape == (4,) and batched["target_idx"].dtype == torch.long


def test_policy_is_differentiable():
    cfg = PolicyConfig(n_sensors=8, window_size=20, target_score_history_len=3,
                       d_model=64, n_layers=1, n_heads=2, perturbation_dim=20)
    policy = AdaptiveAttackPolicy(cfg)
    state = _make_state(B=2)
    state["current_window"].requires_grad_(True)
    out = policy(state)
    loss = out["cat_logits"].sum() + out["delta_mu"].sum() + out["value"].sum()
    loss.backward()
    assert state["current_window"].grad is not None
    # Policy params have gradients
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
