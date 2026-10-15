"""A point robot navigating through hazards with a damage allowance (Safety-Gymnasium).

The task is SafetyPointGoal1 of the Safety-Gymnasium suite. A point robot with a forward force and a turning rate
earns the distance by which it closes in on a goal, and a bonus each time it reaches one; a new goal then appears
elsewhere, so an episode of 1000 steps visits as many goals as the robot reaches. Eight hazards are scattered over
the floor; each step the robot spends inside a hazard costs one unit. The suite's customary limit is 25 units per
episode.

Which requirement does the task mean? The suite's own formulation limits the expected cost per episode to 25, a
constrained MDP. Read as a damage allowance, the limit is a budget: the robot can take 25 units of damage in a
mission, and exhausting it ends the mission. The two requirements differ: an expected cost of 25 allows exhausting
a budget of 25 in half of the episodes. This module poses the second, a tolerated probability of exhausting the
allowance within the episode, which is the class of problems the clipped barrier is for.

Mapping to the budget problem. The budget is the damage allowance; the step cost is the suite's cost; the horizon is
the episode, H = 1000, with utility discount beta = 0.999. The suite's reward is bounded by about one per step (the
goal bonus) and is rarely negative, so ubar = 1.1 and u_minus = 0.1 are taken as its bounds.

Needs Safety-Gymnasium (Python 3.8 to 3.10, MuJoCo 2.3), which pins older Gymnasium versions: use a separate
environment for this problem.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..spec import BudgetSpec, Stage
from ..wrapper import BudgetWrapper


@dataclass(frozen=True)
class Allowance:
    damage: float = 25.0
    horizon: int = 1000
    beta: float = 0.999
    ubar: float = 1.1
    u_minus: float = 0.1
    task: str = "SafetyPointGoal1-v0"

    def spec(self) -> BudgetSpec:
        return BudgetSpec(budgets=(self.damage,), beta=self.beta, ubar=self.ubar, gammas=(1.0,), u_minus=self.u_minus,
                          horizon=self.horizon)


def hazard_cost(obs, action, next_obs, reward, info):
    return [float(info["cost"])]


def make_navigation_env(stage: Stage, allowance: Allowance = Allowance(), *, true_budget_scale: float = 1.0) -> BudgetWrapper:
    """The robot under the given stage. The wrapper appends the remaining allowance as a fraction and the elapsed fraction
    of the episode, which the task's own observation lacks."""
    import safety_gymnasium
    from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium        # moves the step cost into info['cost']
    base = SafetyGymnasium2Gymnasium(safety_gymnasium.make(allowance.task))
    return BudgetWrapper(base, hazard_cost, allowance.spec(), stage, observe="fraction", true_budget_scale=true_budget_scale,
                         observe_time=True)


__all__ = ["Allowance", "hazard_cost", "make_navigation_env"]
