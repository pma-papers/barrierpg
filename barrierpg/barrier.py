"""Interior barriers B: R^K -> [0, inf], finite on the positive orthant and infinite elsewhere."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


class LogBarrier:
    """B(z) = theta sum_k log(1 + b_k / z_k) on the positive orthant, +inf elsewhere (the paper's
    example barrier); theta = 0 gives the failure-only price (B = 0 on the orthant, +inf outside)."""

    def __init__(self, theta: float, budgets: Sequence[float]):
        self.theta = float(theta)
        self.budgets = np.asarray(budgets, dtype=float).reshape(-1)
        if self.theta < 0 or np.any(self.budgets <= 0):
            raise ValueError("theta >= 0 and positive budgets")

    def __call__(self, z) -> float:
        z = np.asarray(z, dtype=float).reshape(-1)
        if np.any(z <= 0.0):          # infinite outside the open positive orthant, also when theta = 0:
            return math.inf           # at zero reserve the reward is clipped to -m (as the paper's clipped reward specifies)
        if self.theta == 0.0:
            return 0.0
        return float(self.theta * np.sum(np.log1p(self.budgets / z)))

    def lower_level(self, eps: Sequence[float]) -> float:
        """underline B(eps) = inf{B(z): z_k <= eps_k for some k} = theta min_k log(1 + b_k / eps_k),
        the quantity of the near-exhaustion certificate (certificates.near_exhaustion_bound)."""
        e = np.asarray(eps, dtype=float).reshape(-1)
        if self.theta == 0.0:
            return 0.0
        return float(self.theta * np.min(np.log1p(self.budgets / e)))


def interior_term(B_value: float, u: float, m: float, clip_utility: bool = True) -> float:
    """The pre-failure reduction of the reward: r_m = u - min{B, u + m} when clipping, u - B otherwise.
    Summed with discounts before failure it is the interior loss I_m."""
    if not clip_utility:
        return float(B_value)
    return float(min(B_value, u + m))


__all__ = ["LogBarrier", "interior_term"]
