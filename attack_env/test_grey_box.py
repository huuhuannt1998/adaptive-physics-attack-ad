"""Smoke tests for the grey-box query interface contract."""
from __future__ import annotations

import pytest
import torch

from attack_env.grey_box import GreyBoxVictim, StandardizationStats, _EPSILON_IQR


def _identity_forecast(x_input: torch.Tensor) -> torch.Tensor:
    """Forecast = last-step-of-context: (B, W-1, N) -> (B, N)."""
    return x_input[:, -1, :]


def _make_victim(n_sensors: int = 5, device: str = "cpu") -> GreyBoxVictim:
    stats = StandardizationStats(
        median=torch.zeros(n_sensors),
        iqr=torch.ones(n_sensors),
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return GreyBoxVictim(_identity_forecast, stats, edge_index, device=device)


def test_query_output_shape():
    v = _make_victim(n_sensors=5)
    X = torch.randn(7, 100, 5)
    out = v.query(X)
    assert out.shape == (7, 5)


def test_query_known_value():
    """With identity forecast, median=0, iqr=1: standardized error == |X[-1] - X[-2]| / (1 + ε)."""
    v = _make_victim(n_sensors=3)
    # Hand-craft X so we know expected output
    X = torch.tensor([[[0.0, 0.0, 0.0],
                       [1.0, 2.0, 3.0],
                       [1.5, 2.0, 3.5]]])  # (1, 3, 3)
    out = v.query(X)
    expected = torch.tensor([[0.5, 0.0, 0.5]]) / (1.0 + _EPSILON_IQR)
    assert torch.allclose(out, expected, atol=1e-6)


def test_query_rejects_wrong_n():
    v = _make_victim(n_sensors=5)
    with pytest.raises(ValueError, match="N=5"):
        v.query(torch.randn(2, 100, 7))


def test_query_rejects_wrong_dim():
    v = _make_victim(n_sensors=5)
    with pytest.raises(ValueError, match="batch, window, N"):
        v.query(torch.randn(100, 5))


def test_no_threshold_or_binary_exposed():
    v = _make_victim(n_sensors=5)
    # public surface check
    public = [m for m in dir(v) if not m.startswith("_")]
    assert set(public) >= {"query", "graph", "n_sensors"}
    forbidden = {"threshold", "decide", "predict_label", "is_anomaly", "raw_error"}
    assert not (set(public) & forbidden), f"forbidden methods exposed: {set(public) & forbidden}"


def test_graph_returns_clone():
    v = _make_victim(n_sensors=5)
    g = v.graph()
    g[0, 0] = 999  # mutate caller's copy
    assert v.graph()[0, 0] != 999  # internal state untouched


def test_query_is_stateless():
    """Calling query twice on the same X must produce identical outputs (no smoothing memory)."""
    v = _make_victim(n_sensors=5)
    X = torch.randn(3, 100, 5)
    a = v.query(X)
    b = v.query(X)
    assert torch.equal(a, b)


def test_query_no_grad_leak():
    """query must run under no_grad — output must not require grad."""
    v = _make_victim(n_sensors=5)
    X = torch.randn(2, 100, 5, requires_grad=True)
    out = v.query(X)
    assert not out.requires_grad


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
