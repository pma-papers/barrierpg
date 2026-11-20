"""Backward induction for finite resource-budget Markov decision processes."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from numbers import Integral
from typing import Sequence

import numpy as np

from .barrier import LogBarrier, interior_term
from .spec import Stage, failure_charge


@dataclass(frozen=True)
class Transition:
    """One outcome of a state-action pair.

    A terminal outcome has `next_state=None`. Consumption is a nonnegative
    integer in units of the tabular reserve. Utility is earned if that outcome
    leaves the reserve nonnegative, including on a successful terminal step.
    """

    probability: float
    next_state: int | None
    utility: float
    consumption: int = 0

    def __post_init__(self):
        if not isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("transition probability must lie in [0, 1]")
        if self.next_state is not None and (
            not isinstance(self.next_state, Integral) or self.next_state < 0
        ):
            raise ValueError("next_state must be a nonnegative state index or None")
        if not isfinite(self.utility):
            raise ValueError("transition utility must be finite")
        if not isinstance(self.consumption, Integral) or self.consumption < 0:
            raise ValueError("transition consumption must be a nonnegative integer")


@dataclass(frozen=True)
class BudgetDPSolution:
    """Optimal action indices and the model-based outcomes of the selected policy.

    Arrays are indexed by `[action_time, state, reserve]`. Values and risks
    are conditional on starting at that entry with no prior failure. The policy
    has one row per decision time. The other arrays include the horizon with
    zero continuation value and risk. Infeasible pathwise entries have action
    -1, value -inf, and NaN utility and risk.
    """

    policy: np.ndarray
    value: np.ndarray
    stopped_utility: np.ndarray
    failure_probability: np.ndarray
    geometric_risk: np.ndarray


def solve_budget_dp(
    transitions: Sequence[Sequence[Sequence[Transition]]],
    *,
    horizon: int,
    budget: int,
    beta: float = 1.0,
    stage: Stage | None = None,
    pathwise: bool = False,
) -> BudgetDPSolution:
    """Solve a known finite-horizon model by Bellman backward induction.

    Supply `transitions[state][action]` as a sequence of outcomes whose
    probabilities sum to one. The model is time homogeneous and the policy
    may depend on time and reserve. There is one undiscounted integer budget.
    `beta` is the utility discount. Failure occurs on a strict budget
    crossing and stops the process before that action's utility is earned.

    In priced mode, `stage` determines the failure charge and optional log
    barrier. The default pays utility before failure with no failure charge.
    A fixed lump charge uses the stage's multiplier mode. In pathwise mode,
    an action is feasible only if every positive-probability outcome remains
    feasible through the horizon. With no stage, this maximizes expected
    discounted utility over the feasible policies. With a stage, its interior
    loss can also enter the objective. Ties select the first action.
    """
    if not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(budget, Integral) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("beta must lie in (0, 1]")
    if len(transitions) == 0:
        raise ValueError("the model must have at least one state")
    for actions in transitions:
        if len(actions) == 0:
            raise ValueError("each state must have at least one action")
        for outcomes in actions:
            if not isclose(fsum(o.probability for o in outcomes), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("the probabilities of each state-action pair must sum to one")
            if any(o.next_state is not None and o.next_state >= len(transitions) for o in outcomes):
                raise ValueError("an outcome refers to a state outside the model")

    stage = Stage(m=0.0) if stage is None else stage
    if not all(isfinite(x) for x in (stage.m, stage.theta, stage.multiplier)):
        raise ValueError("stage price, barrier strength and multiplier must be finite")
    if stage.eta or stage.kappa:
        raise ValueError("the tabular solver uses no action or parameter regularization")

    barrier_cost = np.zeros(budget + 1)
    if stage.theta:
        barrier = LogBarrier(stage.theta, [budget])
        barrier_cost[:] = [barrier([reserve]) for reserve in range(budget + 1)]

    shape = (horizon + 1, len(transitions), budget + 1)
    value = np.zeros(shape)
    utility = np.zeros(shape)
    failure = np.zeros(shape)
    geometric = np.zeros(shape)
    policy = np.full((horizon, len(transitions), budget + 1), -1, dtype=int)

    for t in reversed(range(horizon)):
        if stage.penalty == "multiplier":
            charge = stage.multiplier
        elif beta == 1:
            charge = stage.m * (horizon - t)
        else:
            charge = failure_charge(stage.m, beta, t, horizon)

        for state, actions in enumerate(transitions):
            for reserve in range(budget + 1):
                index = (t, state, reserve)
                best = -np.inf
                utility[index] = failure[index] = geometric[index] = np.nan
                for action, outcomes in enumerate(actions):
                    score = expected_utility = risk = discounted_risk = 0.0
                    for outcome in outcomes:
                        p = outcome.probability
                        if p == 0:
                            continue
                        remaining = reserve - int(outcome.consumption)
                        if remaining < 0:
                            if pathwise:
                                break
                            score -= p * charge
                            risk += p
                            discounted_risk += p
                            continue

                        u = outcome.utility
                        penalty = barrier_cost[remaining]
                        # Match BudgetWrapper's reward floor at exact depletion.
                        if stage.theta and remaining == 0:
                            reward = -stage.m
                        else:
                            reward = u - interior_term(penalty, u, stage.m, stage.clip_utility) if stage.theta else u

                        if outcome.next_state is not None:
                            successor = (t + 1, int(outcome.next_state), remaining)
                            if pathwise and value[successor] == -np.inf:
                                break
                            reward += beta * value[successor]
                            u += beta * utility[successor]
                            risk += p * failure[successor]
                            discounted_risk += p * beta * geometric[successor]
                        score += p * reward
                        expected_utility += p * u
                    else:
                        if score > best:
                            best = score
                            policy[index] = action
                            utility[index] = expected_utility
                            failure[index] = risk
                            geometric[index] = discounted_risk
                value[index] = best

    return BudgetDPSolution(policy, value, utility, failure, geometric)
