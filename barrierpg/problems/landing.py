"""The landing problem of the paper's lander study: a learned lander paid for each landing.

The problem is built on the simulator of Gymnasium's LunarLanderContinuous-v3, with its dynamics and its ends of an
episode. It is not the simulator's own task: the simulator's reward, its penalty for a crash and its reward for coming
to rest do not enter it. A revenue of 36 is paid in the frame in which the lander comes to rest with its centre over the
pad, and each unit of fuel costs 1. The fuel of a frame is the quantity that the simulator's reward charges, 0.30 per
unit of main-engine power and 0.03 per unit of side-engine power.

There are two budgets, as in the paper's model of the study.
  1. The tank of 24 units. Fuel is accumulated against it until the first frame at which a leg touches the ground. From
     then on an empty tank sets both engines off, and the episode continues.
  2. A unit budget that the loss of the lander exceeds. The lander is lost when the simulator ends the episode because
     of a crash or because the lander has left the frame.
An episode ends at the first failure, and the failing frame earns no utility. The observation is the simulator's state
and the remaining fraction of the tank. The discount is beta = 0.999 and the physical horizon is H = 1000 frames; the
episode ends at H as a terminal state of the problem (the replication's runs end it as a truncation, which matters only
to a learner that bootstraps its value there).

The price is charged from the failing frame to the physical horizon, with the reward before failure unclipped, that is
Stage(m, clip_utility=False). The paper gives a price by its charge k, the charge of a failure at frame 100 in units of
the revenue, k = m (1 - beta^900) / (36 (1 - beta)). At the charge k a lost lander costs as much as k landings earn.
Landing.price(k) and Landing.charge(m) convert; the paper's charges are 0, 0.25, 0.5, 1, 2, 4, 8 and 16.

During training the paper adds (beta Phi(s') - Phi(s)) / 12 to the reward of a frame, where Phi is the potential of the
simulator's own reward, which falls with the distance to the pad, with the speed and with the tilt of the lander, and
Phi = 0 at the end of an episode. Its discounted sum over an episode is -Phi(s_0) / 12 on every path, so it lowers the
value of every policy by the same amount and leaves the problem unchanged. make_landing_env adds it with aid=True;
evaluation reads the utility from info, or uses aid=False.

The paper's learner and its three training stages are described in its Online Resource and run by fuel-tank/lunar.py of
the replication repository; tests/test_sb3.py checks this module against that file step by step. Needs Gymnasium with
Box2D (pip install "gymnasium[box2d]").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spec import BudgetSpec, Stage
from ..wrapper import BudgetWrapper

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

FUEL_MAIN, FUEL_SIDE = 0.30, 0.03     # fuel per frame per unit of main- and side-engine power
LOSS = 2.0                            # what the loss of the lander charges to the second budget; it exceeds the unit budget
PAD_HALF_WIDTH = 0.2                  # the pad spans the horizontal coordinates -0.2 to 0.2 of the observation


def fuel_use(action) -> float:
    """Fuel of one frame: the simulator's engine-power rule (0.30 per unit of main, 0.03 per unit of side power)."""
    a0, a1 = float(action[0]), float(action[1])
    main = (min(max(a0, 0.0), 1.0) + 1.0) * 0.5 if a0 > 0.0 else 0.0
    side = min(max(abs(a1), 0.5), 1.0) if abs(a1) > 0.5 else 0.0
    return FUEL_MAIN * main + FUEL_SIDE * side


def potential(state) -> float:
    """The potential of the simulator's own reward, from the first eight coordinates of the observation. It is computed
    in the observation's own precision (float32 from the simulator), as in the paper's runs."""
    s = np.asarray(state)
    return float(-100.0 * np.hypot(s[0], s[1]) - 100.0 * np.hypot(s[2], s[3]) - 100.0 * abs(s[4]) + 10.0 * s[6] + 10.0 * s[7])


@dataclass(frozen=True)
class Landing:
    """The landing problem's constants (the paper's values by default)."""
    revenue: float = 36.0
    fuel_price: float = 1.0
    tank: float = 24.0
    horizon: int = 1000
    beta: float = 0.999
    aid_scale: float = 1.0 / 12.0
    charge_frame: int = 100              # the charge k is the charge of a failure at this frame, in units of the revenue

    def spec(self) -> BudgetSpec:
        """The tank and the unit budget of the lander; the utility of a frame lies between minus the fuel of a frame at
        full thrust (u_minus = 0.33 at the default fuel price) and the revenue."""
        return BudgetSpec(budgets=(self.tank, 1.0), beta=self.beta, ubar=self.revenue,
                          u_minus=self.fuel_price * (FUEL_MAIN + FUEL_SIDE), horizon=self.horizon)

    def _charge_factor(self) -> float:
        return (1.0 - self.beta ** (self.horizon - self.charge_frame)) / ((1.0 - self.beta) * self.revenue)

    def price(self, charge: float) -> float:
        """The price m of the charge k: m = k * revenue * (1 - beta) / (1 - beta^(H - 100))."""
        return float(charge) / self._charge_factor()

    def charge(self, m: float) -> float:
        """The charge k of the price m: the charge of a failure at frame 100 in units of the revenue."""
        return float(m) * self._charge_factor()

    def stage(self, charge: float) -> Stage:
        """The paper's stage at the charge k: the price of k, the reward before failure unclipped, no barrier."""
        return Stage(m=self.price(charge), clip_utility=False)


if gym is not None:
    class LandingSimulator(gym.Env):
        """The simulator with the tank and the revenue. The reward is the utility of the frame (the revenue at
        completion less the fuel); the observation is the simulator's state with the remaining fraction of the tank.

        info per frame: fuel (drawn from the tank), tank_draw (the fuel accumulated against the first budget, zero
        after touchdown), lost, landed, crashed, completed (came to rest over the pad), exhausted (the tank ran out
        before touchdown), touched, empty."""
        metadata = {"render_modes": []}

        def __init__(self, landing: Landing = Landing(), **make_kwargs):
            super().__init__()
            self.landing = landing
            self.inner = gym.make("LunarLanderContinuous-v3", **make_kwargs)
            self.action_space = self.inner.action_space
            low = np.append(self.inner.observation_space.low, 0.0).astype(np.float32)
            high = np.append(self.inner.observation_space.high, 1.0).astype(np.float32)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        def _obs(self, o):
            return np.append(o, max(self.reserve, 0.0) / self.landing.tank).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            o, info = self.inner.reset(seed=seed, options=options)
            self.reserve, self.touched = float(self.landing.tank), False
            return self._obs(o), info

        def step(self, action):
            after_touchdown = self.touched                      # a leg touched the ground in an earlier frame
            if after_touchdown and self.reserve <= 0.0:
                action = np.array([-1.0, 0.0], dtype=np.float32)     # the tank is empty: both engines are off
            o, _, terminated, truncated, info = self.inner.step(action)
            g = fuel_use(action)
            if after_touchdown:
                g = min(g, max(self.reserve, 0.0))              # the last frame of thrust uses what is left
            self.reserve -= g
            exhausted = self.reserve < 0.0 and not after_touchdown
            self.touched = self.touched or bool(o[6] > 0.5 or o[7] > 0.5)
            core = self.inner.unwrapped
            crashed = bool(terminated and not exhausted and core.game_over)
            landed = bool(terminated and not exhausted and not core.game_over and not core.lander.awake)
            lost = bool(terminated and not exhausted and not landed)
            completed = bool(landed and abs(float(o[0])) <= PAD_HALF_WIDTH)
            utility = -self.landing.fuel_price * g + (self.landing.revenue if completed else 0.0)
            info = dict(info)
            info.update(fuel=g, tank_draw=0.0 if after_touchdown else g, lost=lost, landed=landed, crashed=crashed,
                        completed=completed, exhausted=bool(exhausted), touched=self.touched, empty=bool(self.reserve <= 0.0))
            return self._obs(o), float(utility), bool(terminated), bool(truncated), info

    class ProgressAid(gym.Wrapper):
        """Adds scale * (beta Phi(s') - Phi(s)) to the reward, with Phi the simulator's potential and Phi = 0 at the end of
        an episode. The discounted sum over an episode is -scale Phi(s_0) on every path, so the problem is unchanged."""

        def __init__(self, env, beta: float, scale: float):
            super().__init__(env)
            self.beta, self.scale = float(beta), float(scale)

        def reset(self, **kwargs):
            o, info = self.env.reset(**kwargs)
            self.prev_phi = potential(o[:8])
            return o, info

        def step(self, action):
            o, r, terminated, truncated, info = self.env.step(action)
            phi = 0.0 if (terminated or truncated) else potential(o[:8])
            aid = self.scale * (self.beta * phi - self.prev_phi)
            self.prev_phi = phi
            info = dict(info)
            info["aid"] = aid
            return o, float(r) + aid, terminated, truncated, info


def landing_cost(obs, action, next_obs, reward, info):
    """The two step costs: the fuel accumulated against the tank (until touchdown) and the loss of the lander."""
    return [info["tank_draw"], LOSS if info["lost"] else 0.0]


def make_landing_env(stage: Stage, landing: Landing = Landing(), *, aid: bool = True, **make_kwargs):
    """The landing problem under the given stage (the paper's: landing.stage(k)). With aid=True the reward also carries
    the training aid of the paper; the utility in info and the evaluation records do not."""
    if gym is None:
        raise ImportError("the landing problem needs gymnasium with Box2D: pip install 'gymnasium[box2d]'")
    env = BudgetWrapper(LandingSimulator(landing, **make_kwargs), landing_cost, landing.spec(), stage, observe="none")
    return ProgressAid(env, landing.beta, landing.aid_scale) if aid else env


__all__ = ["Landing", "fuel_use", "potential", "landing_cost", "make_landing_env", "LOSS"] + \
    (["LandingSimulator", "ProgressAid"] if gym is not None else [])
