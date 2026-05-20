"""
BETA reimplementation: budgeted, static, grey-box adversarial attack on GNN-based
anomaly detection. Faithful to the pipeline described in Xaviar & Ardakanian 2025
(arXiv 2509.17987), modulo the GAFExplainer ambiguity which we resolved to
Ying-2019-GNNExplainer-via-gradient-saliency in jrn_01KQNCMWFWQ2DPHGBKPKYY9YV7.

Pipeline:
    1. select_candidate_nodes(victim, X, target_idx, k)
       - Saliency-based: |∂score_u / ∂X|, mean over time-axis, take top-k nodes.
       - Implements BETA's "GAFExplainer-based candidate selection" with the
         interpretation that BETA cited Ying-2019 GNNExplainer in the bibliography
         and saliency is the operational equivalent (and what GNNExplainer reduces
         to in the limit of zero mask-training epochs).
    2. prune_to_budget(candidates, learned_graph, B)
       - Eigenvector centrality on the victim's learned graph A; rerank candidates;
         take top-B as the influencer set V_bar.
    3. pgd_attack(victim, X, V_bar, target_idx, eps, alpha, iters, restarts)
       - PGD on x[..., V_bar, :] with α=0.01, 10 iterations, 5 random restarts,
         ε=0.1 in [0,1] feature space (BETA's stated convention).
       - Loss: maximize anomaly-score-flip at target u — i.e., move toward the
         OPPOSITE side of the decision boundary from the clean prediction.

The operational metric BETA uses (FTA) is not implemented here; that's a
post-attack evaluation built atop this attack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BETAConfig:
    """Hyperparameters from BETA paper text + lit_01KQG4D8J9H9PMKP5W13SXMC46 methodology_notes."""

    epsilon: float = 0.1  # ℓ∞ ball, in [0,1] Min-Max-normalized feature space
    pgd_alpha: float = 0.01
    pgd_iters: int = 10
    pgd_restarts: int = 5
    candidate_k: int = 32  # saliency top-k before centrality prune (BETA-paper-implicit)


def select_candidate_nodes(
    victim_forward,
    X: torch.Tensor,
    target_idx: int,
    k: int,
) -> torch.Tensor:
    """Saliency-based candidate selection for BETA's "explainer-picks-influencer-nodes" step.

    Args:
        victim_forward: callable taking X -> (B, N) anomaly scores (one per sensor).
            For GDN/TopoGDN, this is `lambda x: model(x)[0]` — the model returns
            (forecast, learned_graph), we take the forecast.
        X: input window of shape (B, N, W) where N = number of sensors, W = window len.
        target_idx: the sensor u whose decision we want to flip.
        k: number of top-saliency nodes to return (pre-prune candidate set size).

    Returns:
        LongTensor of shape (k,) with sensor indices, sorted by descending saliency.
        Excludes target_idx from results since BETA's threat model safeguards u.
    """
    if X.dim() != 3:
        raise ValueError(f"X must be (B, N, W), got {tuple(X.shape)}")
    B, N, W = X.shape

    X = X.detach().clone().requires_grad_(True)
    scores = victim_forward(X)  # (B, N)
    if scores.shape != (B, N):
        raise RuntimeError(f"victim_forward must return (B, N); got {tuple(scores.shape)}")

    target_score = scores[:, target_idx].sum()
    target_score.backward()

    if X.grad is None:
        raise RuntimeError("victim_forward did not propagate gradients to X")

    # Per-sensor saliency = mean |gradient| over time-axis and batch.
    importance = X.grad.detach().abs().mean(dim=(0, 2))  # (N,)
    importance[target_idx] = -1.0  # safeguarded; never select target
    top = torch.topk(importance, k=min(k, N - 1)).indices
    return top.detach()


def eigenvector_centrality(edge_index: torch.Tensor, num_nodes: int, num_iter: int = 200, eps: float = 1e-8) -> torch.Tensor:
    """Canonical eigenvector centrality via power iteration on the (un-normalized,
    symmetrized) adjacency derived from edge_index. Returns the leading eigenvector
    (absolute values).

    BETA paper specifies eigenvector centrality (Bonacich 1972), not PageRank.
    For a star K_{1,4}, this gives the center higher score than the leaves.
    """
    if edge_index.numel() == 0:
        return torch.ones(num_nodes) / num_nodes

    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    src, dst = edge_index[0].long(), edge_index[1].long()
    A[src, dst] = 1.0
    A = (A + A.T).clamp(max=1.0)  # symmetrize, undirected; clamp so multi-edges don't double-count

    v = torch.ones(num_nodes, dtype=torch.float32) / num_nodes
    prev_v = torch.zeros_like(v)
    for _ in range(num_iter):
        v_new = A @ v
        norm = v_new.norm() + eps
        v_new = v_new / norm
        if torch.allclose(v_new, prev_v, atol=1e-7):
            v = v_new
            break
        prev_v = v
        v = v_new
    return v.abs()


def prune_to_budget(
    candidates: torch.Tensor,
    learned_edge_index: torch.Tensor,
    num_nodes: int,
    budget: int,
) -> torch.Tensor:
    """Eigenvector-centrality reranking of candidates; return top-budget."""
    centrality = eigenvector_centrality(learned_edge_index, num_nodes)  # (N,)
    cand = candidates.detach().long()
    cand_centrality = centrality[cand]
    keep = min(budget, len(cand))
    top = torch.topk(cand_centrality, k=keep).indices
    return cand[top].detach()


def pgd_attack(
    victim_forward,
    X: torch.Tensor,
    V_bar: torch.Tensor,
    target_idx: int,
    clean_prediction: torch.Tensor,  # binary or scalar score; tells us which way to flip
    config: BETAConfig = BETAConfig(),
) -> tuple[torch.Tensor, float]:
    """Projected Gradient Descent on x[:, V_bar, :] within ℓ∞ ε-ball.

    Optimization objective: maximize the magnitude of the change in target's anomaly
    score in the flip-friendly direction. We use a sign-flip loss: if clean_prediction
    is "anomaly" (high score), we minimize the score; if "normal" (low score), we
    maximize it. This is operationalized by signing the loss based on the clean
    prediction's sign.

    Returns (X_perturbed, best_loss_score).
    """
    if X.dim() != 3:
        raise ValueError(f"X must be (B, N, W), got {tuple(X.shape)}")

    V_bar = V_bar.long()
    best_X = None
    best_loss_score = -float("inf")

    # Decide direction: if clean target score is "high" (anomaly), we drive it down.
    # If "low" (normal), we drive it up. Sign convention: loss_sign = +1 to drive UP.
    direction = -torch.sign(clean_prediction).item() if torch.is_tensor(clean_prediction) else -float(np.sign(clean_prediction))
    if direction == 0:
        direction = 1.0  # tie-break: drive up

    for restart in range(config.pgd_restarts):
        # Random init within ℓ∞ ε-ball, only at V_bar features.
        delta = torch.zeros_like(X)
        delta[:, V_bar, :] = (torch.rand_like(X[:, V_bar, :]) * 2 - 1) * config.epsilon
        delta = delta.detach()

        for it in range(config.pgd_iters):
            delta = delta.detach().requires_grad_(True)
            x_perturbed = (X + delta).clamp(0.0, 1.0)
            score = victim_forward(x_perturbed)[:, target_idx].sum()
            loss = direction * score
            loss.backward()

            with torch.no_grad():
                grad = delta.grad
                step = config.pgd_alpha * grad.sign()
                # Mask: only update V_bar features.
                mask = torch.zeros_like(delta)
                mask[:, V_bar, :] = 1.0
                step = step * mask
                delta = (delta + step).clamp(-config.epsilon, config.epsilon).detach()

        x_final = (X + delta).clamp(0.0, 1.0).detach()
        with torch.no_grad():
            final_score = victim_forward(x_final)[:, target_idx].sum().item()

        achieved = direction * final_score
        if achieved > best_loss_score:
            best_loss_score = achieved
            best_X = x_final

    return best_X, best_loss_score


class BETAAttack:
    """Orchestrates select → prune → PGD pipeline for a single (window, target_sensor) pair."""

    def __init__(
        self,
        victim_forward,
        learned_edge_index_fn,
        num_nodes: int,
        config: BETAConfig = BETAConfig(),
    ):
        """
        Args:
            victim_forward: callable X -> (B, N) anomaly scores.
            learned_edge_index_fn: callable () -> LongTensor (2, E) — fetches the victim's
                learned graph A. For GDN/TopoGDN this reads model.learned_graph after a
                forward pass; we expose it as a callable so caller controls the lifecycle.
            num_nodes: N (51 SWaT, 127 WADI).
            config: hyperparameters.
        """
        self.victim_forward = victim_forward
        self.learned_edge_index_fn = learned_edge_index_fn
        self.num_nodes = num_nodes
        self.config = config

    def attack(self, X: torch.Tensor, target_idx: int, budget: int) -> tuple[torch.Tensor, dict]:
        """Run the full BETA pipeline. Returns (X_perturbed, metadata)."""
        with torch.no_grad():
            clean_score = self.victim_forward(X)[:, target_idx].mean().item()

        candidates = select_candidate_nodes(self.victim_forward, X, target_idx, k=self.config.candidate_k)

        # Trigger a forward pass to populate model.learned_graph if it's lazily set.
        with torch.no_grad():
            _ = self.victim_forward(X)
        learned_edge_index = self.learned_edge_index_fn()

        V_bar = prune_to_budget(candidates, learned_edge_index, self.num_nodes, budget)

        clean_pred_tensor = torch.tensor([clean_score])
        X_perturbed, achieved = pgd_attack(
            self.victim_forward, X, V_bar, target_idx, clean_pred_tensor, self.config
        )

        with torch.no_grad():
            final_score = self.victim_forward(X_perturbed)[:, target_idx].mean().item()

        return X_perturbed, {
            "candidates": candidates.tolist(),
            "V_bar": V_bar.tolist(),
            "clean_score": clean_score,
            "final_score": final_score,
            "achieved_loss": achieved,
            "score_delta": final_score - clean_score,
        }
