"""
P1.T1: Gym-compatible MDP environment for adaptive RL attack on GNN-AD victims.

Authorized via Backbrief jrn_01KQPNGW00RMTNZYD6XDZA8AZ9.
Test-split protocol: jrn_01KQPNN9D9GQ113CJ0Y3R83PAZ.

State        s_t = (target_score_history[5], context_window[W,N], remaining_budget, step)
Action       a_t = (categorical sensor index in V\\{u} ∪ {no-op}, Gaussian δ_t in ℓ∞ ε-ball)
Episode      = one anomaly time-range from training-split (70% of 16 WADI ranges)
Reward       r_t = (f_ζ(X_clean)_u - f_ζ(X̃_t)_u) / max(f_ζ(X_clean)_u, ε_floor=0.1)
                    - λ_1 · 1[i_t ≠ no-op]
                    - λ_2 · D_KS(δ_t, π_nominal_i)
                    [- 0 · ||g(X̃_t)||_2  : Plan B physics, λ_3=0 in Phase 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import torch


@dataclass
class MDPConfig:
    """Hyperparameters per Backbrief approval."""

    epsilon_pgd: float = 0.1  # ℓ∞ perturbation ball, BETA-stated
    target_score_history_len: int = 5
    epsilon_reward_floor: float = 0.1  # Brain-locked ε floor for reward normalization
    lambda_1_sparsity: float = 0.01
    lambda_2_stealth: float = 0.5
    lambda_3_physics: float = 0.0  # Plan B, OUT for Phase 1
    per_window_budget_default: int = 5
    target_sensor: int | None = None  # if None, picked from anomaly range argmax
    seed: int = 0


@dataclass
class StepResult:
    """One env step output."""
    state: dict
    reward: float
    done: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class AdaptiveAttackEnv:
    """Gym-compatible attack environment for Phase 1 adaptive policy training.

    Wraps a surrogate `query(X) -> (B, N)` callable (per grey-box interface in
    `attack_env/grey_box.py`). The MDP is single-window-per-episode: agent gets
    a clean window from the policy-training split, picks up to B influencer-sensor
    perturbations within ε ℓ∞ ball, and gets rewarded for suppressing the target's
    standardized score.
    """

    def __init__(
        self,
        surrogate_query: Callable[[torch.Tensor], torch.Tensor],
        nominal_distributions: torch.Tensor,  # (N, K) per-channel sample bins for KS distance
        config: MDPConfig = MDPConfig(),
    ):
        """
        Args:
            surrogate_query: callable X -> (B, N). Same contract as GreyBoxVictim.query.
            nominal_distributions: shape (N, K) — K sample values per sensor representing
                training-set nominal distribution. Used to compute D_KS(δ_t, π_nominal_i)
                approximately by comparing the perturbation δ_t's empirical CDF to the
                stored nominal samples for the chosen sensor i_t.
            config: hyperparameters.
        """
        self.surrogate_query = surrogate_query
        self.nominal = nominal_distributions
        self.config = config
        self.n_sensors = nominal_distributions.shape[0]

        # Episode state (set in reset())
        self._clean_window: torch.Tensor | None = None  # (1, N, W) clean input
        self._current_window: torch.Tensor | None = None  # (1, N, W) modified by attack so far
        self._target_idx: int | None = None
        self._clean_target_score: float | None = None
        self._budget_remaining: int = 0
        self._step_idx: int = 0
        self._episode_max_steps: int = 0
        self._target_score_hist: list[float] = []

    # ──────────────────────────────────────────────────────────────────
    # Action space helpers
    # ──────────────────────────────────────────────────────────────────
    @property
    def n_categorical_actions(self) -> int:
        """N-1 sensors (target excluded) + 1 no-op = N total."""
        return self.n_sensors

    @property
    def perturbation_dim(self) -> int:
        """δ_t lives in ℝ^W where W is the window length; clipped to ε ℓ∞ ball."""
        if self._clean_window is None:
            raise RuntimeError("env not reset")
        return self._clean_window.shape[-1]

    # ──────────────────────────────────────────────────────────────────
    # Reset / Step
    # ──────────────────────────────────────────────────────────────────
    def reset(
        self,
        clean_window: torch.Tensor,
        target_idx: int,
        budget: int | None = None,
        episode_max_steps: int | None = None,
    ) -> dict:
        """Start a new episode on the given clean window with the specified target sensor.

        Args:
            clean_window: (1, N, W) clean input window; the LAST timestep is the prediction
                target; previous timesteps are the input context.
            target_idx: sensor index u to attack. Must be in [0, N).
            budget: per-window budget B (defaults to config.per_window_budget_default).
            episode_max_steps: hard cap on steps per episode regardless of budget. If None,
                defaults to 2× budget (allows mostly-no-op exploration).
        """
        if clean_window.dim() != 3 or clean_window.shape[0] != 1:
            raise ValueError(f"clean_window must be (1, N, W), got {tuple(clean_window.shape)}")
        if not (0 <= target_idx < self.n_sensors):
            raise ValueError(f"target_idx {target_idx} out of [0, {self.n_sensors})")

        self._clean_window = clean_window.detach().clone()
        self._current_window = clean_window.detach().clone()
        self._target_idx = int(target_idx)
        self._budget_remaining = int(budget) if budget is not None else self.config.per_window_budget_default
        self._episode_max_steps = int(episode_max_steps) if episode_max_steps is not None else 2 * self._budget_remaining
        self._step_idx = 0

        with torch.no_grad():
            scores = self.surrogate_query(self._clean_window)  # (1, N)
        self._clean_target_score = float(scores[0, self._target_idx].item())
        self._target_score_hist = [self._clean_target_score] * self.config.target_score_history_len

        return self._encode_state()

    def step(self, action: tuple[int, np.ndarray]) -> StepResult:
        """Apply action, return next StepResult.

        Args:
            action: (sensor_index_or_no_op, delta_vector)
                - sensor_index_or_no_op: int in [0, n_categorical_actions). The last index
                  is the no-op convention; sensor indices ≠ target are the attacker choices.
                  Convention: action_idx == self.n_sensors - 1 is no-op
                              action_idx < self.n_sensors - 1 maps to sensor index after
                              skipping target_idx.
                - delta_vector: shape (W,) ℓ∞-clipped to ±epsilon_pgd.
        """
        if self._clean_window is None:
            raise RuntimeError("env not reset")
        sensor_idx_action, delta = action
        sensor_idx_action = int(sensor_idx_action)
        is_no_op = (sensor_idx_action == self.n_categorical_actions - 1)

        sparsity_penalty = 0.0
        stealth_penalty = 0.0

        if not is_no_op and self._budget_remaining > 0:
            # Map the action's categorical index to the actual sensor index, skipping target.
            attacked_sensor = sensor_idx_action
            if attacked_sensor >= self._target_idx:
                attacked_sensor = attacked_sensor + 1  # skip target

            delta_t = torch.as_tensor(delta, dtype=self._current_window.dtype)
            delta_t = delta_t.clamp(-self.config.epsilon_pgd, self.config.epsilon_pgd)
            self._current_window[0, attacked_sensor, :] = (
                self._clean_window[0, attacked_sensor, :] + delta_t
            ).clamp(0.0, 1.0)
            self._budget_remaining -= 1

            sparsity_penalty = self.config.lambda_1_sparsity
            if self.config.lambda_2_stealth > 0:
                stealth_penalty = self.config.lambda_2_stealth * self._ks_distance(delta_t, attacked_sensor)

        with torch.no_grad():
            attacked_scores = self.surrogate_query(self._current_window)
        attacked_target_score = float(attacked_scores[0, self._target_idx].item())
        self._target_score_hist = self._target_score_hist[1:] + [attacked_target_score]

        # Brain-locked normalized degradation reward with ε floor.
        denom = max(abs(self._clean_target_score), self.config.epsilon_reward_floor)
        primary = (self._clean_target_score - attacked_target_score) / denom
        reward = primary - sparsity_penalty - stealth_penalty

        self._step_idx += 1
        budget_exhausted = self._budget_remaining <= 0
        reached_max = self._step_idx >= self._episode_max_steps
        done = budget_exhausted
        truncated = (not done) and reached_max

        return StepResult(
            state=self._encode_state(),
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
            info={
                "is_no_op": is_no_op,
                "sensor_idx_action": sensor_idx_action,
                "primary_reward": float(primary),
                "sparsity_penalty": float(sparsity_penalty),
                "stealth_penalty": float(stealth_penalty),
                "current_target_score": attacked_target_score,
                "clean_target_score": self._clean_target_score,
                "budget_remaining": self._budget_remaining,
                "step": self._step_idx,
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # State encoding
    # ──────────────────────────────────────────────────────────────────
    def _encode_state(self) -> dict:
        """State dict consumed by the policy network."""
        return {
            "current_window": self._current_window.clone(),  # (1, N, W)
            "target_score_history": torch.tensor(self._target_score_hist, dtype=torch.float32),  # (H,)
            "target_idx": int(self._target_idx),
            "budget_remaining": int(self._budget_remaining),
            "step": int(self._step_idx),
            "clean_target_score": float(self._clean_target_score),
        }

    # ──────────────────────────────────────────────────────────────────
    # KS distance (approximate)
    # ──────────────────────────────────────────────────────────────────
    def _ks_distance(self, delta_t: torch.Tensor, sensor_idx: int) -> float:
        """Approximate D_KS between δ_t's empirical distribution and the precomputed
        nominal distribution for the chosen sensor.

        delta_t: (W,) — the perturbation values at this time-window position.
        Normalized to [0,1] feature space already (by clamp), so KS is computed
        on δ_t values vs the stored nominal samples (same scale).
        """
        delta_np = delta_t.detach().cpu().numpy()
        delta_sorted = np.sort(delta_np)
        nominal_sorted = np.sort(self.nominal[sensor_idx].cpu().numpy())
        n_d = len(delta_sorted)
        n_n = len(nominal_sorted)
        # Two-sample KS: max |F_d(x) - F_n(x)| over the merged support
        merged = np.concatenate([delta_sorted, nominal_sorted])
        merged.sort()
        # Empirical CDFs at each merged value
        f_d = np.searchsorted(delta_sorted, merged, side="right") / n_d
        f_n = np.searchsorted(nominal_sorted, merged, side="right") / n_n
        return float(np.max(np.abs(f_d - f_n)))


def compute_nominal_distributions(train_windows: torch.Tensor, n_samples_per_sensor: int = 1024) -> torch.Tensor:
    """Pre-compute per-sensor nominal value distributions from training data.

    Args:
        train_windows: (T, N, W) training-set windows, all values in [0,1].
        n_samples_per_sensor: K, samples to keep per sensor for KS comparison.

    Returns:
        (N, K) tensor of per-sensor nominal value samples.
    """
    if train_windows.dim() != 3:
        raise ValueError(f"train_windows must be (T, N, W), got {tuple(train_windows.shape)}")
    T, N, W = train_windows.shape
    flat = train_windows.permute(1, 0, 2).reshape(N, -1)  # (N, T*W)
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(N):
        idx = torch.randperm(flat.shape[1], generator=g)[:n_samples_per_sensor]
        samples.append(flat[i, idx])
    return torch.stack(samples, dim=0)  # (N, K)
