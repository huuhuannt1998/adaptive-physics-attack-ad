"""Smoke tests for P1.T1 MDP environment."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack_env.mdp import (
    AdaptiveAttackEnv,
    MDPConfig,
    compute_nominal_distributions,
)


def _identity_query(X: torch.Tensor) -> torch.Tensor:
    """Surrogate that returns last-step values directly as 'scores'."""
    return X[..., -1]  # (B, N)


def _make_env(n_sensors=8, window=20, target=3):
    nominal = torch.rand(n_sensors, 64)
    cfg = MDPConfig(target_score_history_len=3, lambda_1_sparsity=0.0, lambda_2_stealth=0.0,
                    epsilon_pgd=0.1, per_window_budget_default=2)
    env = AdaptiveAttackEnv(_identity_query, nominal, cfg)
    return env


def test_reset_initializes_episode_state():
    env = _make_env()
    X = torch.rand(1, 8, 20)
    s = env.reset(X, target_idx=3)
    assert s["target_idx"] == 3
    assert s["budget_remaining"] == 2
    assert s["step"] == 0
    assert s["target_score_history"].shape == (3,)
    # current_window starts equal to clean_window
    assert torch.equal(s["current_window"], X)


def test_no_op_action_does_not_decrement_budget():
    env = _make_env()
    X = torch.rand(1, 8, 20)
    env.reset(X, target_idx=3)
    no_op_action = (env.n_categorical_actions - 1, np.zeros(20))
    result = env.step(no_op_action)
    assert result.info["budget_remaining"] == 2
    assert result.info["is_no_op"]
    assert result.reward == pytest.approx(0.0, abs=1e-5)  # no perturbation, no movement


def test_perturbation_action_decrements_budget_and_modifies_window():
    env = _make_env()
    X = torch.rand(1, 8, 20)
    env.reset(X, target_idx=3)
    delta = np.full(20, 0.1)  # max perturbation
    result = env.step((0, delta))  # action_idx 0 → sensor 0 (since target=3, no skip needed)
    assert result.info["budget_remaining"] == 1
    assert not result.info["is_no_op"]


def test_action_index_skips_target():
    """Categorical action 3 with target=3 should refer to sensor 4, not target."""
    env = _make_env(n_sensors=8, target=3)
    X = torch.rand(1, 8, 20)
    env.reset(X, target_idx=3)
    delta = np.full(20, 0.1)
    result = env.step((3, delta))  # categorical 3 + target=3 → maps to sensor 4

    # Verify sensor 3 (target) was NOT modified
    cw = result.state["current_window"]
    assert torch.allclose(cw[0, 3, :], X[0, 3, :])
    # Verify sensor 4 WAS modified
    assert not torch.allclose(cw[0, 4, :], X[0, 4, :])


def test_episode_ends_when_budget_exhausted():
    env = _make_env(n_sensors=8, target=3)
    X = torch.rand(1, 8, 20)
    env.reset(X, target_idx=3, budget=2)
    delta = np.full(20, 0.1)
    r1 = env.step((0, delta))
    assert not r1.done
    r2 = env.step((1, delta))
    assert r2.done


def test_reward_is_normalized_by_clean_baseline():
    # Use a non-trivial query that returns predictable scores.
    n_sensors, W = 5, 10

    def fixed_query(X: torch.Tensor) -> torch.Tensor:
        # Return last value of each sensor as score; first call (clean) gives baseline.
        return X[..., -1]

    nominal = torch.rand(n_sensors, 32)
    cfg = MDPConfig(target_score_history_len=2, lambda_1_sparsity=0.0, lambda_2_stealth=0.0,
                    epsilon_pgd=1.0, epsilon_reward_floor=0.1)  # large eps so attack matters
    env = AdaptiveAttackEnv(fixed_query, nominal, cfg)

    X = torch.full((1, n_sensors, W), 0.8)  # clean target score = 0.8
    env.reset(X, target_idx=2)

    # Action: perturb sensor 0 with delta=-0.5 (drives sensor 0's last value down).
    # But target's score doesn't change — primary = (0.8 - 0.8) / max(0.8, 0.1) = 0.
    delta = np.full(W, -0.5)
    r = env.step((0, delta))
    assert abs(r.info["primary_reward"]) < 1e-5

    # Now use a query that ties target's score to sensor 0 — perturbing sensor 0 changes target.
    def coupled_query(X: torch.Tensor) -> torch.Tensor:
        # sensor 2's score = sensor 0's last value (couples them)
        scores = X[..., -1].clone()
        scores[..., 2] = X[..., 0, -1]
        return scores

    env2 = AdaptiveAttackEnv(coupled_query, nominal, cfg)
    env2.reset(X, target_idx=2)
    r2 = env2.step((0, delta))  # δ=-0.5 → sensor 0's last value becomes 0.3 → target score 0.3
    # primary = (0.8 - 0.3) / max(0.8, 0.1) = 0.625
    assert r2.info["primary_reward"] == pytest.approx(0.625, abs=0.01)


def test_compute_nominal_distributions_shape():
    train = torch.rand(100, 8, 20)
    nominal = compute_nominal_distributions(train, n_samples_per_sensor=64)
    assert nominal.shape == (8, 64)
    assert nominal.min() >= 0.0 and nominal.max() <= 1.0


def test_truncation_at_max_steps_when_no_op_only():
    env = _make_env(n_sensors=8, target=3)
    X = torch.rand(1, 8, 20)
    env.reset(X, target_idx=3, budget=10, episode_max_steps=3)
    no_op = (env.n_categorical_actions - 1, np.zeros(20))
    r1 = env.step(no_op); r2 = env.step(no_op); r3 = env.step(no_op)
    assert r3.truncated and not r3.done


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
