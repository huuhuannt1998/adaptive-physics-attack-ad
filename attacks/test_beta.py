"""Smoke tests for BETA reimplementation modules."""
from __future__ import annotations

import pytest
import torch

from attacks.beta import (
    BETAAttack,
    BETAConfig,
    eigenvector_centrality,
    pgd_attack,
    prune_to_budget,
    select_candidate_nodes,
)


def _toy_victim(n_sensors=10, window=20):
    """Toy victim: linear projection of last-step input to per-sensor "anomaly score"."""
    torch.manual_seed(0)
    W = torch.randn(n_sensors, n_sensors) * 0.5

    def forward(X):
        # X: (B, N, W) → score = W @ X[..., -1]
        last_step = X[..., -1]  # (B, N)
        return last_step @ W.T  # (B, N)

    return forward


def test_select_candidate_nodes_excludes_target_and_returns_topk():
    forward = _toy_victim(n_sensors=10, window=20)
    X = torch.randn(2, 10, 20)
    cand = select_candidate_nodes(forward, X, target_idx=3, k=5)
    assert cand.shape == (5,)
    assert 3 not in cand.tolist(), "target index should be excluded"
    assert len(set(cand.tolist())) == 5, "candidates should be unique"


def test_eigenvector_centrality_on_star():
    """Star graph: center node should dominate centrality."""
    edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
    centrality = eigenvector_centrality(edge_index, num_nodes=5)
    assert centrality.argmax().item() == 0
    assert centrality[0] > centrality[1:].max()


def test_prune_to_budget_returns_top_centrality_subset():
    edge_index = torch.tensor([[0, 0, 0, 0, 1], [1, 2, 3, 4, 2]], dtype=torch.long)
    candidates = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    pruned = prune_to_budget(candidates, edge_index, num_nodes=5, budget=2)
    assert pruned.shape == (2,)
    # Nodes 1 and 2 have higher degree (1 has 2, 2 has 2, 3 and 4 have 1 each).
    assert set(pruned.tolist()) <= {1, 2, 3, 4}


def test_pgd_attack_shifts_target_score():
    forward = _toy_victim(n_sensors=10, window=20)
    X = torch.rand(1, 10, 20)
    V_bar = torch.tensor([0, 1, 2], dtype=torch.long)
    clean_pred = forward(X)[:, 3]
    cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.02, pgd_iters=10, pgd_restarts=3)
    X_pert, achieved_loss = pgd_attack(forward, X, V_bar, target_idx=3, clean_prediction=clean_pred, config=cfg)
    # Perturbation should be ≤ ε in the V_bar slice and == 0 elsewhere.
    delta = (X_pert - X).abs()
    other_nodes = [i for i in range(10) if i not in V_bar.tolist()]
    assert delta[:, other_nodes, :].max().item() < 1e-5, "non-V_bar nodes should be untouched (modulo clamp)"
    assert delta[:, V_bar, :].max().item() <= cfg.epsilon + 1e-5
    # Score should have moved in flip direction by at least some amount.
    final_score = forward(X_pert)[:, 3].item()
    direction = -torch.sign(clean_pred).item()
    if direction != 0:
        assert direction * (final_score - clean_pred.item()) > 0, "PGD should move score toward flip direction"


def test_full_beta_attack_pipeline():
    """End-to-end: select → prune → pgd. Verifies orchestration without crashing."""
    forward = _toy_victim(n_sensors=10, window=20)
    fake_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    cfg = BETAConfig(epsilon=0.1, pgd_alpha=0.02, pgd_iters=5, pgd_restarts=2, candidate_k=6)
    attack = BETAAttack(
        victim_forward=forward,
        learned_edge_index_fn=lambda: fake_edge_index,
        num_nodes=10,
        config=cfg,
    )
    X = torch.rand(1, 10, 20)
    X_pert, meta = attack.attack(X, target_idx=5, budget=3)
    assert X_pert.shape == X.shape
    assert len(meta["V_bar"]) <= 3
    assert 5 not in meta["V_bar"], "target should not appear in V_bar"
    assert "score_delta" in meta


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
