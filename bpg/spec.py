"""Problem specification: budgets, horizon, tolerance, and the stage (price, barrier, regularizers).

Vocabulary follows the paper "The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization":

  budgets b_k > 0, cost discounts gamma_k in (0, 1], utility discount beta in (0, 1);
  reserves z_{0,k} = b_k, z_{t+1,k} = (z_{t,k} - g_k) / gamma_k;
  failure at the first action index tau with min_k z_{tau+1,k} < 0 (equality is not failure);
  finite-horizon risk p_T = P(tau <= T), covering T + 1 actions;
  geometric risk q_beta = E[beta^tau 1{tau < infinity}];
  price of failure risk m: the clipped reward r_m = max{-m, u - B(z_{t+1})} is -m in every
  period from failure onward, so m / (1 - beta) is the cost of a unit of geometric risk;
  stopped utility U, interior loss I_m, clipped-reward value V_m = U - I_m - m q_beta / (1 - beta).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np


def _tuple(x, K=None) -> Tuple[float, ...]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if K is not None and arr.size == 1 and K > 1:
        arr = np.repeat(arr, K)
    return tuple(float(v) for v in arr)


@dataclass(frozen=True)
class BudgetSpec:
    """The resource-budget problem as the paper poses it.

    budgets   b_k > 0, one per resource.
    beta      utility discount in (0, 1). Under a physical horizon it also weights the periods.
    gammas    cost discounts gamma_k in (0, 1]; 1 means the budget is a plain cumulative allowance.
    ubar      known upper bound on the one-step utility, ubar >= 0 (needed by every certificate).
    u_minus   magnitude of a lower bound, -u_minus <= u (0 if the utility is nonnegative); the
              performance results and clipping need m >= u_minus, otherwise the clipping acts on
              the utility itself and the frozen-multiplier form is the one to use.
    horizon   None for an infinite physical horizon (absorbing implementation, lump -m/(1-beta) at
              failure); an integer H for a task that ends after H actions (tail only to H).
    """
    budgets: Tuple[float, ...]
    beta: float
    ubar: float
    gammas: Optional[Tuple[float, ...]] = None
    u_minus: float = 0.0
    horizon: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "budgets", _tuple(self.budgets))
        K = len(self.budgets)
        g = tuple(1.0 for _ in range(K)) if self.gammas is None else _tuple(self.gammas, K)
        object.__setattr__(self, "gammas", g)
        if any(b <= 0 for b in self.budgets):
            raise ValueError("budgets must be positive")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if any(not 0.0 < x <= 1.0 for x in self.gammas):
            raise ValueError("cost discounts must lie in (0, 1]")
        if self.ubar < 0 or self.u_minus < 0:
            raise ValueError("ubar and u_minus are magnitudes, hence nonnegative")
        if self.horizon is not None and int(self.horizon) < 1:
            raise ValueError("horizon must be a positive integer or None")

    @property
    def K(self) -> int:
        return len(self.budgets)

    @property
    def budgets_array(self) -> np.ndarray:
        return np.asarray(self.budgets, dtype=float)

    @property
    def gammas_array(self) -> np.ndarray:
        return np.asarray(self.gammas, dtype=float)

    def with_(self, **changes) -> "BudgetSpec":
        return replace(self, **changes)


@dataclass(frozen=True)
class Requirement:
    """What the deployment must satisfy, stated before any learning mechanism is chosen.

    T       last action index of the test; the requirement p_T <= alpha covers T + 1 actions.
    alpha   tolerated failure probability.
    zeta    error probability of the statistical validation (a confidence, not a risk level;
            the paper insists that zeta and alpha are not interchangeable).
    kind    'finite' for p_T <= alpha; 'geometric' for q_beta <= alpha (a random service lifetime
            with survival beta per period); 'eventual' for P(tau < infinity) <= alpha, which the
            paper certifies only with additional cost-tail structure.
    family  number of simultaneous statements (candidates, horizons, tests) sharing zeta.
    """
    T: int
    alpha: float
    zeta: float = 0.05
    kind: str = "finite"
    family: int = 1

    def __post_init__(self):
        if self.kind not in ("finite", "geometric", "eventual"):
            raise ValueError("kind must be 'finite', 'geometric' or 'eventual'")
        if not 0.0 <= self.alpha <= 1.0 or not 0.0 < self.zeta < 1.0:
            raise ValueError("alpha in [0, 1], zeta in (0, 1)")
        if int(self.T) < 0 or int(self.family) < 1:
            raise ValueError("T >= 0 and family >= 1")

    @property
    def error_per_statement(self) -> float:
        return self.zeta / self.family


@dataclass(frozen=True)
class Stage:
    """The fixed stage (m, theta, eta, kappa) of the learning problem.

    m             price of failure risk (clipping level); m = 0 is the unpriced learner.
    theta         scale of the interior barrier B(z) = theta sum_k log(1 + b_k / z_k); 0 removes it
                  (failure-only price).
    eta           weight of the action relative entropy to the action reference.
    kappa         strength of the parameter regularizer (weight decay / divergence to the prior).
    clip_utility  True: the paper's clipped reward max{-m, u - B}. False: the frozen-multiplier
                  form, pre-failure reward u - B unclipped, only the failure priced; use it when the
                  utility takes values below -m (for example a crash penalty).
    penalty       'price' (fixed price, the discounted total of m charged at failure) or
                  'multiplier' (a flat penalty nu at failure, updated by dual ascent; m is then only
                  the clipping level of a barrier reward and is unused when theta = 0).
    multiplier    initial value nu of the flat penalty in 'multiplier' mode.
    """
    m: float
    theta: float = 0.0
    eta: float = 0.0
    kappa: float = 0.0
    clip_utility: bool = True
    penalty: str = "price"
    multiplier: float = 0.0

    def __post_init__(self):
        if self.m < 0 or self.theta < 0 or self.eta < 0 or self.kappa < 0 or self.multiplier < 0:
            raise ValueError("m, theta, eta, kappa and multiplier are nonnegative")
        if self.penalty not in ("price", "multiplier"):
            raise ValueError("penalty must be 'price' or 'multiplier'")

    @property
    def failure_only(self) -> bool:
        return self.theta == 0.0

    def with_(self, **changes) -> "Stage":
        return replace(self, **changes)


# ---------------------------------------------------------------------------------------------
# Conversions between the price and the multiplier, and the failure tails
# ---------------------------------------------------------------------------------------------
def discounted_total(m: float, beta: float) -> float:
    """m / (1 - beta): the cost of a unit of geometric risk at price m (the absorbing lump sum)."""
    return float(m) / (1.0 - float(beta))


def price_from_multiplier(nu: float, beta: float) -> float:
    """A multiplier method that converges to a flat failure penalty nu returns the fixed-price optimum
    for m = (1 - beta) nu (the fixed-price proposition and the text after it), in the absorbing form and without an interior barrier."""
    return (1.0 - float(beta)) * float(nu)


def multiplier_from_price(m: float, beta: float) -> float:
    return discounted_total(m, beta)


def failure_tail(m: float, beta: float, t: int, horizon: Optional[int] = None) -> float:
    """Discounted total of the per-period price charged at a failure at action index t:
    m / (1 - beta) for an infinite horizon, m (1 - beta^(H - t)) / (1 - beta) under a physical
    horizon H (the failing action t and the periods t+1, ..., H-1)."""
    if horizon is None:
        return discounted_total(m, beta)
    H = int(horizon)
    if t >= H:
        return 0.0
    return float(m) * (1.0 - float(beta) ** (H - t)) / (1.0 - float(beta))


def price_from_multiplier_physical(nu: float, beta: float, horizon: int, typical_failure_time: float) -> float:
    """Under a physical horizon the flat penalty nu equals the tail of the price m at the typical
    failure time t: nu = m (1 - beta^(H - t)) / (1 - beta). Inverts that relation."""
    frac = 1.0 - float(beta) ** (int(horizon) - float(typical_failure_time))
    return float(nu) * (1.0 - float(beta)) / frac


def discounted_sum_to(beta: float, H: int) -> float:
    """S_H = (1 - beta^H) / (1 - beta)."""
    return (1.0 - float(beta) ** int(H)) / (1.0 - float(beta))


__all__ = ["BudgetSpec", "Requirement", "Stage", "discounted_total", "price_from_multiplier",
           "multiplier_from_price", "failure_tail", "price_from_multiplier_physical", "discounted_sum_to"]
