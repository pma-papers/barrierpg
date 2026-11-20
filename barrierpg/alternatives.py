"""Expected-cost policy updates and upper-tail risk for finite scenario models."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class CpoStep:
    step: np.ndarray
    status: Literal["unconstrained", "active", "recovery", "infeasible"]
    predicted_cost_gap: float


def cpo_step(utility_gradient, cost_gradient, fisher, cost_gap: float, kl_limit: float) -> CpoStep:
    """Solve the dense single-cost CPO subproblem.

    Maximize the linearized utility subject to cost_gap + cost_gradient @ step <= 0
    and step @ fisher @ step / 2 <= kl_limit. The gap is current expected cost
    minus its limit. If the linear constraint is infeasible, the recovery step
    minimizes the predicted cost within the trust region.
    """
    reward = np.asarray(utility_gradient, dtype=float)
    cost = np.asarray(cost_gradient, dtype=float)
    metric = np.asarray(fisher, dtype=float)
    n = reward.size
    if (reward.shape != (n,) or n == 0 or cost.shape != (n,) or metric.shape != (n, n)
            or not all(np.isfinite(x).all() for x in (reward, cost, metric))
            or not isfinite(cost_gap) or not isfinite(kl_limit) or kl_limit <= 0):
        raise ValueError("expected finite gradients, a square Fisher matrix and a positive KL limit")
    if not np.allclose(metric, metric.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Fisher matrix must be symmetric")
    try:
        factor = np.linalg.cholesky((metric + metric.T) / 2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Fisher matrix must be positive definite") from exc

    # In the transformed coordinates the quadratic trust region is a ball.
    gain = np.linalg.solve(factor, reward)
    normal = np.linalg.solve(factor, cost)
    radius = sqrt(kl_limit) * sqrt(2)
    gain_norm = float(np.linalg.norm(gain))
    direction = radius * (gain / gain_norm) if gain_norm else np.zeros_like(gain)
    normal_norm = float(np.linalg.norm(normal))

    if cost_gap + normal @ direction <= 0:
        status = "unconstrained"
    elif normal_norm == 0:
        direction = np.zeros_like(gain)
        status = "infeasible"
    else:
        unit_normal = normal / normal_norm
        offset = -cost_gap / normal_norm
        if offset < -radius:
            direction = -radius * unit_normal
            status = "recovery"
        else:
            tangent = gain - (gain @ unit_normal) * unit_normal
            tangent_norm = float(np.linalg.norm(tangent))
            direction = offset * unit_normal
            if tangent_norm > np.finfo(float).eps * gain_norm * n:
                remaining = sqrt(max(0.0, (radius - abs(offset)) * (radius + abs(offset))))
                direction += remaining * (tangent / tangent_norm)
            status = "active"

    step = np.linalg.solve(factor.T, direction)
    return CpoStep(step, status, float(cost_gap + cost @ step))


@dataclass(frozen=True)
class CpoUpdate:
    step: np.ndarray
    status: Literal["accepted", "recovery", "rejected"]


def cpo_update(utility_gradient, cost_gradient, fisher, cost_gap: float, kl_limit: float,
               evaluate_step: Callable[[np.ndarray], tuple[float, float, float]], *,
               backtracks: int = 12, shrink: float = 0.5) -> CpoUpdate:
    """Backtrack the reference step using a caller's utility gain, cost gap and mean KL.

    The callback receives an increment relative to the current parameters. It
    must measure KL from the old to the new policy. A recovery update can still
    violate the cost limit. The reliability of each check depends on the caller's
    estimates.
    """
    if backtracks < 0 or not 0 < shrink < 1:
        raise ValueError("expected nonnegative backtracks and a shrink factor in (0, 1)")
    proposal = cpo_step(utility_gradient, cost_gradient, fisher, cost_gap, kl_limit)
    recovery = proposal.status == "recovery"
    trial = proposal.step.copy()
    if proposal.status != "infeasible" and np.any(trial):
        for _ in range(backtracks + 1):
            gain, gap, kl = evaluate_step(trial)
            if all(map(isfinite, (gain, gap, kl))) and 0 <= kl <= kl_limit:
                if recovery and gap < cost_gap:
                    return CpoUpdate(trial, "recovery")
                if not recovery and gap <= 0 and (gain >= 0 or cost_gap > 0):
                    return CpoUpdate(trial, "accepted")
            trial = trial * shrink
    return CpoUpdate(np.zeros_like(trial), "rejected")


def omnisafe_cpo(env_id: str, *, cost_limit: float, custom_cfgs: dict | None = None):
    """Construct OmniSafe's CPO Agent for a registered cost environment.

    The caller defines the episode cost and starts training via agent.learn().
    cost_limit uses the units of OmniSafe's Metrics/EpCost for that environment.
    """
    if not isfinite(cost_limit):
        raise ValueError("cost_limit must be finite")
    try:
        from omnisafe import Agent
    except ModuleNotFoundError as exc:
        if exc.name != "omnisafe":
            raise
        raise ImportError("install OmniSafe in a compatible environment to train CPO") from exc
    cfg = dict(custom_cfgs or {})
    cfg["algo_cfgs"] = dict(cfg.get("algo_cfgs", {}), cost_limit=float(cost_limit))
    return Agent("CPO", env_id, custom_cfgs=cfg)


def _scenario_weights(size: int, probabilities) -> np.ndarray:
    if probabilities is None:
        return np.full(size, 1 / size)
    weights = np.asarray(probabilities, dtype=float)
    if (weights.shape != (size,) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-10)):
        raise ValueError("scenario probabilities must be nonnegative and sum to one")
    return weights / weights.sum()


def upper_cvar(costs, tail_mass: float, probabilities=None) -> float:
    """Average cost in the worst tail_mass fraction of a finite distribution.

    tail_mass lies in (0, 1]. Probability atoms at the tail boundary are split
    if needed. The confidence level convention is 1 - tail_mass.
    """
    values = np.asarray(costs, dtype=float)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or not 0 < tail_mass <= 1):
        raise ValueError("expected a nonempty finite cost vector and tail_mass in (0, 1]")
    weights = _scenario_weights(values.size, probabilities)
    if tail_mass == 1:
        return float(weights @ values)
    order = np.argsort(values)[::-1]
    sorted_weights = weights[order]
    prior_mass = np.concatenate(([0.0], np.cumsum(sorted_weights[:-1])))
    included = np.minimum(sorted_weights, np.maximum(tail_mass - prior_mass, 0.0))
    return float((included / tail_mass) @ values[order])


@dataclass(frozen=True)
class ScenarioSelection:
    index: int | None
    risk: np.ndarray


def select_scenario_policy(expected_utility, costs_by_policy, *, cost_limit: float,
                           formulation: Literal["mean", "chance", "cvar", "pathwise"],
                           probabilities=None, budget: float | None = None,
                           tail_mass: float | None = None) -> ScenarioSelection:
    """Select the highest-utility feasible candidate from a finite policy list.

    Rows of costs_by_policy are candidates and columns are common scenarios.
    Cost is one scalar per trajectory, such as total consumption. For chance
    constraints, budget sets the strict crossing threshold and cost_limit is a
    probability. Pathwise feasibility checks every positive-probability scenario.
    """
    utility = np.asarray(expected_utility, dtype=float)
    costs = np.asarray(costs_by_policy, dtype=float)
    if (costs.ndim != 2 or costs.size == 0 or utility.shape != (costs.shape[0],)
            or not np.isfinite(costs).all() or not np.isfinite(utility).all() or not isfinite(cost_limit)):
        raise ValueError("expected finite policy utilities, scenario costs and a cost limit")
    weights = _scenario_weights(costs.shape[1], probabilities)
    if formulation == "mean":
        risk = costs @ weights
    elif formulation == "chance":
        if budget is None or not isfinite(budget) or not 0 <= cost_limit <= 1:
            raise ValueError("chance constraints need a finite budget and a probability limit")
        risk = (costs > budget) @ weights
    elif formulation == "cvar":
        if tail_mass is None:
            raise ValueError("CVaR constraints need a tail_mass")
        risk = np.array([upper_cvar(row, tail_mass, weights) for row in costs])
    elif formulation == "pathwise":
        risk = np.max(costs[:, weights > 0], axis=1)
    else:
        raise ValueError("unknown scenario formulation")
    feasible = np.flatnonzero(risk <= cost_limit)
    index = int(feasible[np.argmax(utility[feasible])]) if feasible.size else None
    return ScenarioSelection(index, risk)
