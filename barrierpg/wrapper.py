"""Environment wrapper: budgets, bounded barrier reward, failure tail, multiplier mode.

Works as a gymnasium.Wrapper when gymnasium is installed; otherwise as a plain wrapper around any
object with reset() -> (obs, info) and step(action) -> (obs, reward, terminated, truncated, info).

Per step the wrapper computes the K costs g_k from a user function, updates the reserves
z_{t+1,k} = (z_{t,k} - g_k) / gamma_k, and replaces the reward:
  before failure   r = max{-m, u - B(z_{t+1})}      (clip_utility=True, the paper's clipped reward)
                   r = u - B(z_{t+1})               (clip_utility=False, frozen-multiplier form)
  at failure       r = -m / (1 - beta)              (absorbing implementation, horizon None)
                   r = -m (1 - beta^(H - t)) / (1 - beta)   (physical horizon H)
                   r = -nu                          (penalty='multiplier', nu set by dual ascent)
and terminates the episode at failure. With geometric_stop=True the episode also ends after each
period with probability 1 - beta, which implements the utility discount exactly when the learner
uses algorithmic discount one (the convention of the wear-allowance study).

info carries per step: failure, utility, interior_loss, cost, reserve, reserve_fraction, t; and at
the end of an episode a 'budget_episode' record (the key 'episode' is reserved by Stable-Baselines3's Monitor) (U, I, tau, failed, length, min_reserve_fraction,
clipped_return) that barrierpg.evaluate and the training monitor consume.

true_budget_scale < 1 judges failure against a reduced true allowance while the observation shows
the nominal reserve (the misjudged-allowance test of the wear study).

observe_time=True also appends the elapsed fraction of the horizon, t/H. With a fixed horizon the policy needs it to
pace the budget: whether the remaining budget suffices depends on how much of the episode is left.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .barrier import LogBarrier, interior_term
from .spec import BudgetSpec, Stage, failure_tail

try:  # optional dependency
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None


class _PlainWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


_Base = gym.Wrapper if gym is not None else _PlainWrapper


if gym is not None:
    class _Adapter(gym.Env):
        """Presents a plain reset/step object (no gymnasium.Env base) to gym.Wrapper."""
        metadata = {"render_modes": []}

        def __init__(self, env):
            self.inner = env
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

        def reset(self, *, seed=None, options=None):
            return self.inner.reset(seed=seed, options=options)

        def step(self, action):
            return self.inner.step(action)

CostFn = Callable[[object, object, object, float, dict], Sequence[float]]
UtilityFn = Callable[[object, object, object, float, dict], float]


class BudgetWrapper(_Base):
    """See the module docstring.

    cost(obs, action, next_obs, reward, info) -> K nonnegative step costs g_k.
    utility(obs, action, next_obs, reward, info) -> one-step utility u (default: the reward).
    spec: BudgetSpec; stage: Stage; observe: 'fraction' (z/b clipped to [0, 1]), 'reserve' or 'none'.
    geometric_stop: end the episode after each period with probability 1 - beta (see above).
    true_budget_scale: judge failure against scale * b while observing the nominal reserve.
    observe_time: also observe t/H, the elapsed fraction of the horizon (needs spec.horizon).
    """

    def __init__(self, env, cost: CostFn, spec: BudgetSpec, stage: Stage, *, utility: Optional[UtilityFn] = None,
                 observe: str = "fraction", geometric_stop: bool = False, true_budget_scale: float = 1.0,
                 barrier: Optional[Callable] = None, observe_time: bool = False):
        if gym is not None and not isinstance(env, gym.Env):
            env = _Adapter(env)
        super().__init__(env)
        self.cost_fn, self.utility_fn = cost, utility
        self.budget_spec, self.stage = spec, stage      # gym.Wrapper reserves the name spec
        self.b = spec.budgets_array
        self.gam = spec.gammas_array
        self.K = spec.K
        self.beta, self.m = float(spec.beta), float(stage.m)
        self.horizon = spec.horizon
        self.penalty, self.multiplier = stage.penalty, float(stage.multiplier)
        self.clip_utility = bool(stage.clip_utility)
        self.observe, self.geometric_stop = observe, bool(geometric_stop)
        self.true_scale = float(true_budget_scale)
        self.observe_time = bool(observe_time)
        if self.observe_time and spec.horizon is None:
            raise ValueError("observe_time needs a horizon in the BudgetSpec")
        self.B = LogBarrier(stage.theta, self.b) if barrier is None else barrier
        self._rng = np.random.default_rng()
        if gym is not None and observe != "none" and hasattr(env, "observation_space") \
                and isinstance(env.observation_space, gym.spaces.Box):
            low = np.append(env.observation_space.low, np.zeros(self.K + int(self.observe_time))).astype(np.float32)
            hi = np.ones(self.K) if observe == "fraction" else self.b
            high = np.append(env.observation_space.high, np.append(hi, np.ones(int(self.observe_time)))).astype(np.float32)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    # --- controls that callbacks use ------------------------------------------------------
    def set_multiplier(self, value: float):
        self.multiplier = float(value)

    def set_price(self, value: float):
        self.m = float(value)

    # --- helpers --------------------------------------------------------------------------
    def _obs(self, o):
        if self.observe == "none":
            return o
        z = np.maximum(self.reserve, 0.0)
        extra = np.minimum(z / self.b, 1.0) if self.observe == "fraction" else z
        if self.observe_time:
            extra = np.append(extra, min(self.t / self.horizon, 1.0))
        return np.append(np.asarray(o, dtype=np.float32).reshape(-1), extra).astype(np.float32)

    def failure_reward(self) -> float:
        if self.penalty == "multiplier":
            return -self.multiplier
        return -failure_tail(self.m, self.beta, self.t, self.horizon)

    # --- gym API --------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is not None:
            # an independent stream for the geometric clock: seeding it with the environment's own seed would correlate
            # maintenance access with the environment's shocks (the two generators would replay the same numbers)
            self._rng = np.random.default_rng([int(seed), 0x6C6F636B])
        o, info = self.env.reset(seed=seed, options=options, **kwargs) if gym is not None else self.env.reset(seed=seed, **kwargs)
        if hasattr(self.cost_fn, "reset"):
            self.cost_fn.reset()
        self.t = 0
        self.cum = np.zeros(self.K)                       # sum_j gamma^j g_j
        self.reserve = self.b.copy()                      # nominal reserve z_t (observed)
        self.min_true_fraction = 1.0
        self.U = self.I = self.clipped = 0.0
        self.tau = None
        self.last_obs = o
        return self._obs(o), info

    def step(self, action):
        o, r, terminated, truncated, info = self.env.step(action)
        g = np.asarray(self.cost_fn(self.last_obs, action, o, r, info), dtype=float).reshape(-1)
        if g.shape != (self.K,) or np.any(g < 0):
            raise ValueError(f"cost must return {self.K} nonnegative numbers")
        u = float(r) if self.utility_fn is None else float(self.utility_fn(self.last_obs, action, o, r, info))
        t = self.t
        self.cum = self.cum + self.gam ** t * g
        nominal_next = (self.reserve - g) / self.gam
        true_next = self.gam ** (-(t + 1)) * (self.true_scale * self.b - self.cum)
        failed = bool(np.any(true_next < 0.0))
        self.min_true_fraction = float(min(self.min_true_fraction, np.min(true_next / self.b)))
        w = 1.0 if self.geometric_stop else self.beta ** t
        info = dict(info)
        if failed:
            reward = self.failure_reward()
            utility = interior = 0.0
            self.tau = t
            terminated = True
        else:
            Bz = self.B(nominal_next if self.true_scale == 1.0 else true_next)
            if self.stage.theta == 0.0 and (self.penalty == "multiplier" or not self.clip_utility):
                interior = 0.0            # failure-only price, unclipped (frozen-multiplier) form: only the failure is priced
            else:
                interior = interior_term(Bz, u, self.m, self.clip_utility)
                if not np.isfinite(interior):
                    # a reserve of exactly zero (possible with whole-unit costs) under the unclipped form: the barrier is
                    # infinite there, and the step is charged as the clipped reward charges it, down to -m
                    interior = u + self.m
            reward = u - interior
            utility = u
            self.U += w * u
            self.I += w * interior
        self.clipped += w * reward
        self.reserve = nominal_next
        self.t = t + 1
        self.last_obs = o
        if not failed and self.horizon is not None and self.t >= self.horizon:
            terminated = True
        if not failed and self.geometric_stop and not (terminated or truncated) and self._rng.random() > self.beta:
            terminated = True
        info.update(failure=failed, utility=utility, interior_loss=interior, cost=g.copy(),
                    reserve=self.reserve.copy(), reserve_fraction=(np.maximum(self.reserve, 0.0) / self.b).copy(), t=t)
        if terminated or truncated:
            info["episode_failed"] = failed
            info["budget_episode"] = dict(U=self.U, I=self.I, tau=self.tau, failed=failed, length=self.t,
                                   min_reserve_fraction=self.min_true_fraction, clipped_return=self.clipped,
                                   geometric_stop=self.geometric_stop)
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info


class RunningMaxCost:
    """Nonnegative step costs from a resource use that can be negative (for example net emissions in a week).

    signed_use(obs, action, next_obs, reward, info) -> K signed uses d_t. The transform charges the increment of the
    running maximum of the cumulative use, g_t = max(C_t, M_{t-1}) - M_{t-1} with C_t = d_0 + ... + d_t, so that
    sum_j g_j = M_t >= C_t and the first time the running maximum exceeds the budget is the first time the cumulative
    use does. Use it with cost discount one. The observed reserve is then b - M_t, which is below the true remaining
    allowance b - C_t after a net saving; if the requirement concerns the end of the horizon only (a year-end miss), check
    that the first crossing coincides with the end-of-horizon event on evaluated trajectories (the emissions study did)."""

    def __init__(self, signed_use: Callable, K: int = 1):
        self.signed_use, self.K = signed_use, int(K)
        self.reset()

    def reset(self):
        self.C = np.zeros(self.K)
        self.M = np.zeros(self.K)

    def __call__(self, obs, action, next_obs, reward, info):
        d = np.asarray(self.signed_use(obs, action, next_obs, reward, info), dtype=float).reshape(-1)
        self.C = self.C + d
        newM = np.maximum(self.M, self.C)
        g = newM - self.M
        self.M = newM
        return g


__all__ = ["BudgetWrapper", "RunningMaxCost"]
