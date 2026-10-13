"""An asset operated under a wear allowance (the running example, Sections 3.1 and 6.1 of the paper), numpy only.

Price index s' = 0.8 s + 0.2 w, w ~ Unif[-1, 1], s_0 = 0; operating rate a in [0, 1];
utility u = (1 + s/2) a / (3/2) in [0, 1]; wear g = 0.1 a xi with xi gamma-distributed, mean one,
shape 1/2; allowance b = 1, cost discount one, beta = 0.95. Maintenance access arrives after each
period with probability 1 - beta, so the geometric risk is exactly the probability of a breakdown
before access. Standing still never fails and earns nothing; every positive rate can fail.

WearCore is the raw simulator (reward = utility, info carries the wear); wrap it with
BudgetWrapper(cost=wear_cost, ...) to obtain the paper's objectives. Policies: idle, full rate,
the incumbent rule a = min{1, z/(3b)} with Gaussian dispersion in the pre-clipping variable,
and the exact benchmark's reference numbers for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..spec import BudgetSpec, Stage
from ..wrapper import BudgetWrapper

try:
    import gymnasium as gym
    _Box = gym.spaces.Box
except ImportError:  # pragma: no cover
    gym = None
    _Box = None


@dataclass(frozen=True)
class WearModel:
    beta: float = 0.95
    budget: float = 1.0
    wear_rate: float = 0.10
    wear_shape: float = 0.5
    rho: float = 0.8
    price_swing: float = 0.5

    def utility(self, s, a):
        return (1.0 + self.price_swing * s) * a / (1.0 + self.price_swing)

    def spec(self, ubar: float = 1.0) -> BudgetSpec:
        return BudgetSpec(budgets=(self.budget,), beta=self.beta, ubar=ubar, gammas=(1.0,), u_minus=0.0, horizon=None)


class WearCore:
    """Raw simulator. Observation (s, 1): the price index and a constant; action a in [0, 1] (clipped).
    Reward is the utility; info['wear'] is the step cost. Episodes never end by themselves; the
    BudgetWrapper ends them at failure or by geometric stopping."""
    metadata = {"render_modes": []}

    def __init__(self, model: WearModel = WearModel()):
        self.model = model
        self.np_random = np.random.default_rng()
        if _Box is not None:
            self.observation_space = _Box(np.array([-1.0, 0.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
            self.action_space = _Box(0.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.s = 0.0
        return np.array([self.s, 1.0], dtype=np.float32), {}

    def step(self, action):
        M = self.model
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, 1.0))
        xi = self.np_random.gamma(M.wear_shape, 1.0 / M.wear_shape)
        u = float(M.utility(self.s, a))
        g = M.wear_rate * a * xi
        self.s = M.rho * self.s + (1.0 - M.rho) * self.np_random.uniform(-1.0, 1.0)
        return np.array([self.s, 1.0], dtype=np.float32), u, False, False, dict(wear=g, rate=a)


def wear_cost(obs, action, next_obs, reward, info):
    return [info["wear"]]


def make_wear_env(stage: Stage, model: WearModel = WearModel(), *, geometric_stop: bool = True, true_budget_scale: float = 1.0,
                  ubar: float = 1.0) -> BudgetWrapper:
    """The wear-allowance problem under the given stage. Observation: (s, 1, z/b)."""
    return BudgetWrapper(WearCore(model), wear_cost, model.spec(ubar), stage, observe="fraction",
                         geometric_stop=geometric_stop, true_budget_scale=true_budget_scale)


# --- policies (obs = (s, 1, z/b)) -------------------------------------------------------------
def idle_policy(obs):
    return np.zeros(1, dtype=np.float32)


def full_rate_policy(obs):
    return np.ones(1, dtype=np.float32)


class IncumbentRule:
    """a = min{1, z / (span b)} with Gaussian dispersion std in the pre-clipping variable x in [-1, 1],
    rate = (x + 1)/2 (the campaign's incumbent: span 3, std 0.1)."""

    def __init__(self, span: float = 3.0, std: float = 0.1, seed: int = 0):
        self.span, self.std = float(span), float(std)
        self.rng = np.random.default_rng(seed)

    def mean_x(self, z_frac):
        rate = np.minimum(1.0, np.maximum(z_frac, 0.0) / self.span)
        return 2.0 * rate - 1.0

    def __call__(self, obs):
        x = self.mean_x(float(obs[-1])) + self.std * self.rng.standard_normal()
        return np.array([(np.clip(x, -1.0, 1.0) + 1.0) / 2.0], dtype=np.float32)

    def deterministic(self, obs):
        return np.array([(np.clip(self.mean_x(float(obs[-1])), -1.0, 1.0) + 1.0) / 2.0], dtype=np.float32)

    def log_density(self, x, z_frac):
        mu = self.mean_x(z_frac)
        return -0.5 * ((x - mu) / self.std) ** 2 - np.log(self.std) - 0.5 * np.log(2 * np.pi)

    def kl_from_gaussian(self, mean_x, std_x, z_frac):
        """KL(N(mean_x, std_x^2) || rule at z) in the pre-clipping variable (closed form)."""
        mu = self.mean_x(z_frac)
        return np.log(self.std / std_x) + (std_x ** 2 + (mean_x - mu) ** 2) / (2 * self.std ** 2) - 0.5


# Exact benchmark (dynamic programming on a grid, numerics/wear-allowance/benchmark.json of the manuscript): the
# failure-only optimum's (breakdown probability q_beta, stopped utility U) by price, and the exact
# barrier optima for theta = 0.1. Grid values; the paper re-evaluates the policies by Monte Carlo.
EXACT_FRONTIER = {0.0: (0.59187, 5.4343), 0.01: (0.55413, 5.429), 0.02: (0.43856, 5.3937), 0.03: (0.33083, 5.3402),
                  0.05: (0.20696, 5.2443), 0.1: (0.10602, 5.1051), 0.2: (0.05737, 4.9697), 0.3: (0.04056, 4.8878),
                  0.5: (0.02639, 4.7788), 1.0: (0.01453, 4.6131), 2.0: (0.00718, 4.4044), 3.0: (0.00452, 4.2751),
                  5.0: (0.00253, 4.123), 10.0: (0.00115, 3.9332), 20.0: (0.00053, 3.7629), 30.0: (0.00033, 3.6685)}
EXACT_BARRIER = {0.3: (0.02417, 4.2441), 1.0: (0.00648, 3.9883), 3.0: (0.00175, 3.7352), 10.0: (0.00046, 3.5069)}
INCUMBENT = dict(q_beta=0.0108, U=2.73)   # the incumbent rule (span 3, std 0.1), Table 1 of the manuscript

__all__ = ["WearModel", "WearCore", "wear_cost", "make_wear_env", "idle_policy", "full_rate_policy", "IncumbentRule",
           "EXACT_FRONTIER", "EXACT_BARRIER", "INCUMBENT"]
