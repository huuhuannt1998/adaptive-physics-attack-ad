"""
Grey-box query interface for trained GNN-based anomaly detectors.

Threat model (jrn_01KQG3WF8PV5S8GSGGVBGJ9H0P): the attacker has query access to the
deployed victim and trains a surrogate. The attacker knows or learns the graph A.
Detector parameters and per-sensor standardization stats are private.

Brain-locked interface (jrn_01KQGR0QSAXDXKWX31F3JAS49K, ratifying R-EXEC-4):
    query(X: torch.Tensor) -> torch.Tensor
        X      shape (batch, window=100, N)   N=51 SWaT, N=127 WADI
        return shape (batch, N)               per-window per-sensor STANDARDIZED errors,
                                              before max-aggregation.

Contract enforced here:
    - No binary decisions exposed.
    - No threshold exposed.
    - No raw (pre-standardization) forecasting errors exposed.
    - Standardization uses per-sensor median + IQR computed once at training time on
      training-set forecasting residuals; never recomputed at query time.
    - Smoothing (4-step rolling mean used in GDN's published `evaluate.py`) is INTENTIONALLY
      omitted from `query()` to keep the interface stateless across calls. Time-axis
      smoothing belongs in the attack policy, not in the victim wrapper.
    - Graph A is exposed via `graph()` — threat model grants this knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


_EPSILON_IQR = 1e-2  # mirrors GDN evaluate.py:58


@dataclass
class StandardizationStats:
    """Per-sensor median and IQR of |forecast - actual| residuals, computed at training time."""

    median: torch.Tensor  # shape (N,)
    iqr: torch.Tensor  # shape (N,)

    def __post_init__(self):
        assert self.median.shape == self.iqr.shape, "median and iqr must align"
        assert self.median.dim() == 1, "stats must be per-sensor 1-D tensors"


class GreyBoxVictim:
    """Wraps a trained 1-step-forecast detector with the locked grey-box query interface.

    Construct via `GreyBoxVictim.from_checkpoint(...)` or by passing a forecast callable +
    standardization stats + edge_index directly.
    """

    def __init__(
        self,
        forecast_fn: Callable[[torch.Tensor], torch.Tensor],
        stats: StandardizationStats,
        edge_index: torch.Tensor,
        device: str | torch.device = "cpu",
    ):
        # Private state. Underscore prefix marks "do not access from attacker code."
        self._forecast_fn = forecast_fn
        self._stats = stats
        self._edge_index = edge_index
        self._device = torch.device(device)
        self._n_sensors: int = stats.median.numel()

        # Move stats to the same device as the model.
        self._mid = stats.median.to(self._device)
        self._iqr_abs = torch.abs(stats.iqr.to(self._device))

    @property
    def n_sensors(self) -> int:
        return self._n_sensors

    def graph(self) -> torch.Tensor:
        """Threat-model-permitted: attacker knows the learned graph A."""
        return self._edge_index.clone()

    @torch.no_grad()
    def query(self, X: torch.Tensor) -> torch.Tensor:
        """Standardized per-sensor anomaly scores for a batch of windows.

        Args:
            X: shape (batch, window, N). The last time-step is the prediction target;
               steps [0..window-2] are the input context.

        Returns:
            shape (batch, N) standardized errors. No smoothing applied.
        """
        if X.dim() != 3:
            raise ValueError(f"expected (batch, window, N), got {tuple(X.shape)}")
        if X.shape[-1] != self._n_sensors:
            raise ValueError(f"expected N={self._n_sensors}, got N={X.shape[-1]}")

        X = X.to(self._device).float()
        x_input = X[:, :-1, :]  # (B, W-1, N)
        x_target = X[:, -1, :]  # (B, N)

        forecast = self._forecast_fn(x_input)  # expected shape (B, N)
        if forecast.shape != x_target.shape:
            raise RuntimeError(
                f"forecast shape mismatch: expected {tuple(x_target.shape)}, got {tuple(forecast.shape)}"
            )

        delta = torch.abs(forecast - x_target)  # (B, N)
        err_std = (delta - self._mid) / (self._iqr_abs + _EPSILON_IQR)
        return err_std

    def __repr__(self) -> str:
        # Intentionally hides the wrapped model identity from external callers.
        return f"GreyBoxVictim(n_sensors={self._n_sensors}, device={self._device})"


def compute_training_stats(
    forecast_fn: Callable[[torch.Tensor], torch.Tensor],
    train_loader,
    n_sensors: int,
    device: str | torch.device = "cpu",
) -> StandardizationStats:
    """Compute per-sensor median and IQR over training-set forecasting residuals.

    Iterate the training loader once with the model in eval mode; collect |forecast - actual|
    residuals per sensor; compute median + IQR via numpy/scipy after gathering.

    The training loader is expected to yield (x_input, x_target, ...) tuples where
    x_input has shape (B, W-1, N) and x_target has shape (B, N).
    """
    from scipy.stats import iqr as scipy_iqr

    device = torch.device(device)
    deltas_per_sensor: list[np.ndarray] = []
    with torch.no_grad():
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x_input, x_target = batch[0].to(device).float(), batch[1].to(device).float()
            else:
                raise TypeError("train_loader must yield (x_input, x_target, ...) tuples")
            forecast = forecast_fn(x_input)
            delta = torch.abs(forecast - x_target).detach().cpu().numpy()  # (B, N)
            deltas_per_sensor.append(delta)

    all_deltas = np.concatenate(deltas_per_sensor, axis=0)  # (T, N)
    if all_deltas.shape[1] != n_sensors:
        raise ValueError(f"residual matrix shape {all_deltas.shape} disagrees with n_sensors={n_sensors}")

    median = np.median(all_deltas, axis=0)  # (N,)
    iqr_vals = scipy_iqr(all_deltas, axis=0)  # (N,)
    return StandardizationStats(
        median=torch.from_numpy(median).float(),
        iqr=torch.from_numpy(iqr_vals).float(),
    )
