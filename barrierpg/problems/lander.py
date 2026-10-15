"""A lunar lander with a fuel tank: imposing a budget on a controller that is already in use.

The lander is Gymnasium's LunarLanderContinuous. Its task rewards a soft landing on the pad (+100), punishes a crash
(-100), shapes the approach, and charges the fuel its engines burn. The fuel now comes from a tank: each frame
consumes 0.30 per unit of main-engine power and 0.03 per unit of side-engine power, and a lander that empties its
tank before touching down has failed. The tank of 24 units is smaller than a controller trained without a tank
needs: such a controller burns about 26 to 28 units in a median landing, so it runs dry in most landings unless it
learns to save fuel.

The setting is common in practice. A controller is in use; a new limit arrives (a smaller tank, a battery, a quota).
Retraining from scratch with the tank does not even discover landing, because random exploration empties the tank
within about a hundred frames. The budget is therefore imposed by continuing to train the controller in use, and that
controller is the natural prior: the policy one would accept if training gained nothing.

Mapping to the budget problem. The budget is the tank; the step cost is the fuel burned; the horizon is the episode,
H = 1000 frames, with utility discount beta = 0.999. The task's crash penalty lies below any sensible price, so the
stage should use clip_utility=False (only the failure is priced). info at the last frame says whether the lander
'landed' or 'crashed'. Needs Gymnasium with Box2D (pip install "gymnasium[box2d]").
"""
from __future__ import annotations

from dataclasses import dataclass

from ..spec import BudgetSpec, Stage
from ..wrapper import BudgetWrapper

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None


def fuel_use(action) -> float:
    """Fuel burned in one frame: the task's own engine-power rule (0.30 per unit of main, 0.03 per unit of side)."""
    a0, a1 = float(action[0]), float(action[1])
    main = (min(max(a0, 0.0), 1.0) + 1.0) * 0.5 if a0 > 0.0 else 0.0
    side = min(max(abs(a1), 0.5), 1.0) if abs(a1) > 0.5 else 0.0
    return 0.30 * main + 0.03 * side


@dataclass(frozen=True)
class Tank:
    capacity: float = 24.0      # a controller trained without a tank burns about 26 to 28 units in a median landing
    horizon: int = 1000
    beta: float = 0.999
    ubar: float = 100.0          # the landing bonus bounds the task's reward per frame
    u_minus: float = 100.0       # the crash penalty

    def spec(self) -> BudgetSpec:
        return BudgetSpec(budgets=(self.capacity,), beta=self.beta, ubar=self.ubar, gammas=(1.0,), u_minus=self.u_minus,
                          horizon=self.horizon)


if gym is not None:
    class LanderOutcome(gym.Wrapper):
        """Adds info['landed'] and info['crashed'] at the last frame, read from the simulator's state."""

        def __init__(self, **make_kwargs):
            super().__init__(gym.make("LunarLanderContinuous-v3", **make_kwargs))

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            if terminated:
                core = self.env.unwrapped
                info["crashed"] = bool(core.game_over or abs(obs[0]) >= 1.0)
                info["landed"] = bool(not info["crashed"] and not core.lander.awake)
            return obs, reward, terminated, truncated, info


def fuel_cost(obs, action, next_obs, reward, info):
    return [fuel_use(action)]


def make_lander_env(stage: Stage, tank: Tank = Tank(), *, true_budget_scale: float = 1.0) -> BudgetWrapper:
    """The lander under the given stage; the wrapper appends the remaining fuel as a fraction of the tank."""
    if gym is None:
        raise ImportError("the lander needs gymnasium with Box2D: pip install 'gymnasium[box2d]'")
    return BudgetWrapper(LanderOutcome(), fuel_cost, tank.spec(), stage, observe="fraction", true_budget_scale=true_budget_scale)


__all__ = ["Tank", "fuel_use", "fuel_cost", "make_lander_env"] + (["LanderOutcome"] if gym is not None else [])
