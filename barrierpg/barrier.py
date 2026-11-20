"""Reserve penalties for the post-transition reserve in BudgetWrapper."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class _ReservePenalty:
    """Common validation for separable functions of reserve fractions."""

    def __init__(self, theta: float, budgets: Sequence[float]):
        self.theta = float(theta)
        self.budgets = np.asarray(budgets, dtype=float).reshape(-1).copy()
        if (not math.isfinite(self.theta) or self.theta < 0 or not self.budgets.size
                or not np.isfinite(self.budgets).all() or np.any(self.budgets <= 0)):
            raise ValueError("expected a finite nonnegative theta and positive finite budgets")

    def _reserve(self, z) -> np.ndarray:
        reserve = np.asarray(z, dtype=float).reshape(-1)
        if reserve.shape != self.budgets.shape or np.any(np.isnan(reserve)):
            raise ValueError("expected one reserve per budget and no NaNs")
        return reserve

    def lower_level(self, eps: Sequence[float]) -> float:
        """Infimum of the penalty when at least one reserve is at most eps."""
        reserve = self._reserve(eps)
        if not np.isfinite(reserve).all() or np.any(reserve <= 0):
            raise ValueError("reserve thresholds must be positive and finite")
        return float(np.min(self._terms(reserve)))


class LogBarrier(_ReservePenalty):
    """B(z) = theta sum_k log(1 + b_k / z_k) on the positive orthant, +inf elsewhere (the paper's
    example barrier). A failure-only stage in BudgetWrapper bypasses the barrier entirely before failure."""

    def __call__(self, z) -> float:
        z = self._reserve(z)
        if np.any(z <= 0):
            return math.inf
        return float(np.sum(self._terms(z)))

    def _terms(self, z: np.ndarray) -> np.ndarray:
        if self.theta == 0.0:
            return np.zeros_like(z)
        return self.theta * np.logaddexp(0, np.log(self.budgets) - np.log(z))


class InverseBarrier(_ReservePenalty):
    """Reciprocal barrier theta * sum(b_k / z_k), infinite at nonpositive reserve."""

    def __call__(self, z) -> float:
        z = self._reserve(z)
        if np.any(z <= 0):
            return math.inf
        return float(np.sum(self._terms(z)))

    def _terms(self, z: np.ndarray) -> np.ndarray:
        if self.theta == 0.0:
            return np.zeros_like(z)
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(math.log(self.theta) + np.log(self.budgets) - np.log(z))


class ReserveHinge(_ReservePenalty):
    """Finite penalty theta * sum(max(0, fraction - z_k / b_k)**power).

    Use power=1 for a linear slope below the target reserve and power=2 for
    a smooth onset. A target fraction of 1 makes the linear term proportional
    to used budget when reserves stay between zero and their starting values.
    """

    def __init__(self, theta: float, budgets: Sequence[float], *, fraction: float, power: int = 1):
        super().__init__(theta, budgets)
        if not 0 < fraction <= 1 or power not in (1, 2):
            raise ValueError("fraction must be in (0, 1] and power must be 1 or 2")
        self.fraction, self.power = float(fraction), power

    def __call__(self, z) -> float:
        return float(np.sum(self._terms(self._reserve(z))))

    def _terms(self, z: np.ndarray) -> np.ndarray:
        if self.theta == 0.0:
            return np.zeros_like(z)
        with np.errstate(over="ignore"):
            shortfall = np.clip(self.fraction - z / self.budgets, 0, None)
        return self.theta * shortfall ** self.power


def interior_term(B_value: float, u: float, m: float, clip_utility: bool = True) -> float:
    """The pre-failure reduction of the reward: r_m = u - min{B, u + m} when clipping, u - B otherwise.
    Summed with discounts before failure it is the interior loss I_m."""
    if not clip_utility:
        return float(B_value)
    return float(min(B_value, u + m))


__all__ = ["LogBarrier", "InverseBarrier", "ReserveHinge", "interior_term"]
