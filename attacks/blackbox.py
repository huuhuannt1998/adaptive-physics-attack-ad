"""
Gradient-free, query-based black-box attackers for the adaptive-attack
evaluation (paper Section VII).

These complement the gradient-free PPO attacker and contrast with BETA's
white-box-surrogate PGD. All three operate under the same gray-box
scalar-query threat model; the black-box attackers here use ONLY the
scalar score interface (no gradients), and we count every victim query
so attack strength can be reported as a function of query budget.

Two standard baselines:
  - RandomSearchAttack: sample random perturbations in the eps-ball on a
    query-selected budget set; keep the best (lowest target score).
  - SPSAAttack: Simultaneous Perturbation Stochastic Approximation --
    estimate the gradient of the target score via two-sided random
    finite differences, take a projected sign/step update.

Both match the BETAAttack interface:
    attack(X, target_idx, budget) -> (X_perturbed, info)
where info includes 'queries' (total victim_forward calls) and
'score_delta' (final - clean target score; negative = successful
evasion).

Objective (evasion): minimize the target sensor's anomaly score so a
detected window flips to "normal". Degradation reported downstream is
clean_score - final_score (positive = successful).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlackBoxConfig:
    epsilon: float = 0.1          # ell_inf ball radius in [0,1] feature space
    # Random search
    rs_iters: int = 200           # number of random proposals
    # SPSA
    spsa_iters: int = 50          # SPSA optimization steps
    spsa_c: float = 0.01          # perturbation magnitude for gradient estimate
    spsa_lr: float = 0.02         # step size
    spsa_avg: int = 4             # gradient-estimate averages per step
    seed: int = 0


def _query_saliency_budget_set(victim_forward, X, target_idx, budget, eps,
                               candidate_idxs):
    """Gradient-free budget-set selection. For each candidate sensor, bump
    it by +eps across the window, query the target score, and rank sensors
    by how much they reduce it. Return the top-`budget` and the query count.
    """
    base = victim_forward(X)[0, target_idx].item()
    n_q = 1
    deltas = []
    for j in candidate_idxs:
        Xj = X.clone()
        Xj[:, j, :] = (Xj[:, j, :] + eps).clamp(0.0, 1.0)
        s = victim_forward(Xj)[0, target_idx].item()
        n_q += 1
        deltas.append((base - s, int(j)))  # larger reduction = better
    deltas.sort(reverse=True)
    chosen = [j for _, j in deltas[:budget]]
    return torch.tensor(chosen, dtype=torch.long), n_q


class _BaseBlackBox:
    def __init__(self, victim_forward, num_nodes, config: BlackBoxConfig = BlackBoxConfig()):
        self.victim_forward = victim_forward
        self.num_nodes = num_nodes
        self.config = config

    def _candidates(self, target_idx):
        # All sensors except the target are eligible for the budget set.
        return [j for j in range(self.num_nodes) if j != target_idx]


class RandomSearchAttack(_BaseBlackBox):
    """Random search in the eps-ball on a query-selected budget set."""

    def attack(self, X: torch.Tensor, target_idx: int, budget: int):
        cfg = self.config
        g = torch.Generator().manual_seed(cfg.seed + target_idx)
        with torch.no_grad():
            clean = self.victim_forward(X)[0, target_idx].item()
            V_bar, n_q = _query_saliency_budget_set(
                self.victim_forward, X, target_idx, budget, cfg.epsilon,
                self._candidates(target_idx))

            best_X = X.clone()
            best_score = clean
            for _ in range(cfg.rs_iters):
                delta = torch.zeros_like(X)
                rnd = (torch.rand(X[:, V_bar, :].shape, generator=g) * 2 - 1) * cfg.epsilon
                delta[:, V_bar, :] = rnd
                Xp = (X + delta).clamp(0.0, 1.0)
                s = self.victim_forward(Xp)[0, target_idx].item()
                n_q += 1
                if s < best_score:
                    best_score = s
                    best_X = Xp
        return best_X, {
            "queries": n_q,
            "V_bar": V_bar.tolist(),
            "clean_score": clean,
            "final_score": best_score,
            "score_delta": best_score - clean,
        }


class SPSAAttack(_BaseBlackBox):
    """SPSA: two-sided random finite-difference gradient estimate of the
    target score w.r.t. the budget-set perturbation, with a projected
    sign-step update."""

    def attack(self, X: torch.Tensor, target_idx: int, budget: int):
        cfg = self.config
        g = torch.Generator().manual_seed(cfg.seed + 1000 + target_idx)
        with torch.no_grad():
            clean = self.victim_forward(X)[0, target_idx].item()
            V_bar, n_q = _query_saliency_budget_set(
                self.victim_forward, X, target_idx, budget, cfg.epsilon,
                self._candidates(target_idx))

            shape = X[:, V_bar, :].shape
            delta = torch.zeros(shape)  # perturbation on the budget set only
            best_X, best_score = X.clone(), clean

            for _ in range(cfg.spsa_iters):
                grad_est = torch.zeros(shape)
                for _ in range(cfg.spsa_avg):
                    # Rademacher perturbation direction.
                    bump = (torch.randint(0, 2, shape, generator=g).float() * 2 - 1)
                    d_plus = (delta + cfg.spsa_c * bump).clamp(-cfg.epsilon, cfg.epsilon)
                    d_minus = (delta - cfg.spsa_c * bump).clamp(-cfg.epsilon, cfg.epsilon)
                    Xp, Xm = X.clone(), X.clone()
                    Xp[:, V_bar, :] = (X[:, V_bar, :] + d_plus).clamp(0.0, 1.0)
                    Xm[:, V_bar, :] = (X[:, V_bar, :] + d_minus).clamp(0.0, 1.0)
                    sp = self.victim_forward(Xp)[0, target_idx].item()
                    sm = self.victim_forward(Xm)[0, target_idx].item()
                    n_q += 2
                    grad_est += (sp - sm) / (2.0 * cfg.spsa_c) * bump
                grad_est /= cfg.spsa_avg
                # Descend the target score (evasion); project to eps-ball.
                delta = (delta - cfg.spsa_lr * grad_est.sign() * cfg.epsilon).clamp(
                    -cfg.epsilon, cfg.epsilon)
                Xcur = X.clone()
                Xcur[:, V_bar, :] = (X[:, V_bar, :] + delta).clamp(0.0, 1.0)
                s = self.victim_forward(Xcur)[0, target_idx].item()
                n_q += 1
                if s < best_score:
                    best_score, best_X = s, Xcur
        return best_X, {
            "queries": n_q,
            "V_bar": V_bar.tolist(),
            "clean_score": clean,
            "final_score": best_score,
            "score_delta": best_score - clean,
        }
