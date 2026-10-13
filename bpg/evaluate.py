"""Monte Carlo evaluation of a fixed candidate through the budget wrapper.

The candidate is a policy callable obs -> action (use sb3_policy for a Stable-Baselines3 model) and
the environment factory returns a BudgetWrapper. Each episode yields the wrapper's 'budget_episode' record:
stopped utility U, interior loss I, failure index tau, length, minimum true reserve fraction and
the clipped return. From n fresh episodes the evaluation estimates

  p_T       frequency of tau <= T (the finite-horizon risk; T = None counts every failure),
  q_beta    mean of beta^tau 1{failed} (the geometric risk), or the plain failure frequency when
            the wrapper ends episodes by geometric stopping,
  U, I_m    stopped utility and interior loss,
  V_m       U - I_m - m E[tail weight] with the tail convention of the spec (absorbing or physical
            horizon) or the frozen-multiplier form when the stage does not clip,
  near      frequency of coming within a fraction of the allowance of exhaustion (failures included).

Under geometric stopping the episode may end by the clock before action T, so p_T is censored and
not estimated (nan); the Bernoulli event of each episode is then 'failure before the clock stops',
whose probability is q_beta itself, and validate() tests q_beta <= alpha. A finite-horizon bound
follows from beta^T p_T <= q_beta (Report uses it).

Evaluate the candidate that is deployed: the stochastic policy and its mean-action policy are
different candidates and are validated separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .spec import BudgetSpec, Stage, failure_tail
from .validation import DirectValidation, validate_direct


@dataclass
class Evaluation:
    n: int
    seed: Optional[int]
    T: Optional[int]
    beta: float
    m: float
    failures: int             # failures counted for p_T (tau <= T when T is given)
    failures_any: int         # failures at any time within the episodes
    p_T: float
    q_beta: float
    U: float
    U_se: float
    I_m: float
    V_m: float
    V_m_se: float
    mean_length: float
    near: Dict[float, float]
    min_reserve_quantiles: Dict[float, float]
    geometric_stop: bool
    event: str                # 'p_T' or 'q_beta': the Bernoulli event that `failures` counts
    tau: np.ndarray = field(repr=False)
    U_episodes: np.ndarray = field(repr=False)
    I_episodes: np.ndarray = field(repr=False)
    V_episodes: np.ndarray = field(repr=False)
    clipped_returns: np.ndarray = field(repr=False)
    failed: np.ndarray = field(repr=False)
    min_reserve: np.ndarray = field(repr=False)

    @property
    def q_se(self) -> float:
        w = np.where(self.failed, self.beta ** np.where(self.failed, self.tau, 0), 0.0) if not self.geometric_stop else self.failed.astype(float)
        return float(w.std(ddof=1) / np.sqrt(self.n)) if self.n > 1 else float("nan")

    def validate(self, alpha: float, zeta: float, family: int = 1) -> DirectValidation:
        """Exact binomial test of `event` <= alpha at error zeta / family: p_T (T of the evaluation), or
        q_beta under geometric stopping."""
        v = validate_direct(self.failures, self.n, alpha, zeta, family)
        v.event = self.event
        return v

    def summary(self) -> str:
        t = "any" if self.T is None else str(self.T)
        pt = f"p_T(T={t})={self.p_T:.4g}" if self.event == "p_T" else "p_T censored by the geometric clock"
        s = (f"n={self.n}: {pt} ({self.failures} failures counted for {self.event}), q_beta={self.q_beta:.4g}, "
             f"U={self.U:.4g}±{self.U_se:.2g}, I_m={self.I_m:.4g}, V_m={self.V_m:.4g}±{self.V_m_se:.2g}"
             + (f", mean length {self.mean_length:.1f}" if self.mean_length == self.mean_length else ""))
        near = ", ".join(f"{k:g}b: {v:.3g}" for k, v in self.near.items())
        return s + f"\nnear-exhaustion (min reserve <=): {near}"


def sb3_policy(model, deterministic: bool = False) -> Callable:
    """Adapter from a Stable-Baselines3 model to obs -> action."""
    def policy(obs):
        a, _ = model.predict(obs, deterministic=deterministic)
        return a
    return policy


def evaluate(env_factory: Callable[[], object], policy: Callable, n: int, *, spec: BudgetSpec, stage: Stage,
             T: Optional[int] = None, seed: Optional[int] = 0, max_steps: int = 100_000,
             near_thresholds: Sequence[float] = (0.02, 0.05, 0.1, 0.2), policy_seed: Optional[int] = None) -> Evaluation:
    """n fresh episodes of `policy` in `env_factory()` (a BudgetWrapper), seeds seed, seed+1, ...

    T: last action index for p_T; None counts every failure within the episode.
    The clipped value uses the stage's price and the spec's tail convention. If the policy is
    stochastic and samples with numpy's global state, set policy_seed for reproducibility.
    """
    env = env_factory()
    if policy_seed is not None:
        np.random.seed(policy_seed)
    beta, m = float(spec.beta), float(stage.m)
    H = spec.horizon
    tau = np.full(n, -1); failed = np.zeros(n, dtype=bool); U = np.zeros(n); I = np.zeros(n)
    length = np.zeros(n); minres = np.ones(n); clipped = np.zeros(n); geo = False
    for i in range(n):
        obs, _ = env.reset(seed=None if seed is None else int(seed) + i)
        rec = None
        for _ in range(max_steps):
            obs, r, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                rec = info.get("budget_episode")
                break
        if rec is None:
            raise RuntimeError("episode did not end within max_steps; the environment must terminate or truncate")
        geo = bool(rec.get("geometric_stop", False))
        failed[i] = rec["failed"]; tau[i] = -1 if rec["tau"] is None else rec["tau"]
        U[i], I[i], length[i], minres[i], clipped[i] = rec["U"], rec["I"], rec["length"], rec["min_reserve_fraction"], rec["clipped_return"]
    return evaluation_from_episodes(failed, tau, U, I, spec=spec, stage=stage, T=T, geometric_stop=geo, length=length,
                                    min_reserve=minres, clipped_returns=clipped, seed=seed, near_thresholds=near_thresholds)


def evaluation_from_episodes(failed, tau, U, I, *, spec: BudgetSpec, stage: Stage, T: Optional[int] = None,
                             geometric_stop: bool = False, length=None, min_reserve=None, clipped_returns=None,
                             seed: Optional[int] = None, near_thresholds: Sequence[float] = (0.02, 0.05, 0.1, 0.2)) -> Evaluation:
    """Build an Evaluation from per-episode arrays produced by any evaluator (for example a vectorized simulator run on
    common random numbers): failure flags, failure indices tau (-1 for no failure), stopped utilities U and interior
    losses I (discounted by beta, or undiscounted under geometric stopping), and optionally episode lengths, minimum
    true reserve fractions and clipped returns. The value, risks and validation follow the conventions of evaluate()."""
    failed = np.asarray(failed, dtype=bool); n = failed.size
    tau = np.asarray(tau, dtype=int); U = np.asarray(U, dtype=float); I = np.asarray(I, dtype=float)
    length = np.full(n, np.nan) if length is None else np.asarray(length, dtype=float)
    minres = np.ones(n) if min_reserve is None else np.asarray(min_reserve, dtype=float)
    beta, m = float(spec.beta), float(stage.m)
    H = spec.horizon
    geo = bool(geometric_stop)
    if geo:
        weight = failed.astype(float)                       # the episode itself is the geometric clock
        tail = m / (1.0 - beta) * weight
    else:
        weight = np.where(failed, beta ** np.where(failed, tau, 0), 0.0)
        tail = np.array([failure_tail(m, beta, int(t), H) if f else 0.0 for f, t in zip(failed, tau)])
    V = U - I - tail
    if stage.penalty == "multiplier":
        V = U - I - stage.multiplier * weight            # frozen-multiplier value U - nu q (I is the barrier part only)
    clipped = V.copy() if clipped_returns is None else np.asarray(clipped_returns, dtype=float)
    if geo:
        sel, event, pT = failed, "q_beta", float("nan")        # the clock censors p_T; the episode event is q_beta
    else:
        sel = failed if T is None else (failed & (tau <= T)); event = "p_T"; pT = float(sel.mean())
    k = int(sel.sum())
    near = {float(e): float(np.mean(minres <= e)) for e in near_thresholds}
    qs = {p: float(np.quantile(minres, p)) for p in (0.01, 0.05, 0.25, 0.5)}
    se = (lambda x: float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    return Evaluation(n, seed, T, beta, m, k, int(failed.sum()), pT, float(weight.mean()), float(U.mean()), se(U), float(I.mean()),
                      float(V.mean()), se(V), float(np.nanmean(length)) if np.isfinite(length).any() else float("nan"), near, qs, geo, event,
                      tau, U, I, V, clipped, failed, minres)


def stopped_action_relative_entropy(env_factory: Callable[[], object], policy_with_kl: Callable, n: int, *, beta: float,
                                    seed: int = 0, max_steps: int = 100_000, geometric_stop: Optional[bool] = None) -> dict:
    """Monte Carlo estimate of A = E sum_{t <= tau} beta^t KL(pi(.|y_t) || rho(.|y_t)).

    policy_with_kl(obs) -> (action, kl_at_obs). The sum includes the failing action and stops at
    the end of the episode. With geometric stopping (read from the wrapper unless given) the
    episode clock carries the discount and the terms are summed unweighted."""
    env = env_factory()
    geo = bool(getattr(env, "geometric_stop", False)) if geometric_stop is None else bool(geometric_stop)
    A = np.zeros(n)
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        for t in range(max_steps):
            a, kl = policy_with_kl(obs)
            A[i] += kl if geo else beta ** t * kl
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return dict(A=float(A.mean()), A_se=float(A.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))


__all__ = ["Evaluation", "sb3_policy", "evaluate", "evaluation_from_episodes", "stopped_action_relative_entropy"]
