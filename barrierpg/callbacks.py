"""Stable-Baselines3 callbacks and optimizer hooks: the regularizers and the adaptive multiplier on a finite actor.

  DualAscent        adaptive multiplier toward a prescribed tolerance, with a diagnosis of its trace;
  ActionAnchor      the action relative entropy to an action reference, charged as a reward term;
  ParameterAnchor   the pull of the particle form: weight decay toward a prior parameter theta_0
                    (the potential term, the finite-network part of a Gaussian prior);
  ParameterNoise    the noise of the particle form: Gaussian noise of variance 2 kappa1 x lr on every actor
                    parameter after each optimizer step (the entropy term of relative entropy);
  BudgetMonitor     per-rollout failure rate, stopped utility, interior loss, clipped value, near-exhaustion
                    and dispersion, with trend tests;
  fit_actor_mean, set_log_std   build a prior from an incumbent rule (the actor at theta_0 is the prior policy).

All are optional; importing this module without stable_baselines3 installed gives placeholders that raise
on use. The environments must be BudgetWrapper instances (possibly inside VecNormalize). The actor helpers
assume an on-policy Gaussian actor (PPO, A2C) with SB3's ActorCriticPolicy layout.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as np

from .convergence import MultiplierDiagnosis, multiplier_diagnosis, plot_training, training_convergence, trend_test
from .paper import cite
from .spec import failure_tail

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _HAVE_SB3 = True
except ImportError:  # pragma: no cover
    _HAVE_SB3 = False

    class BaseCallback:  # type: ignore
        def __init__(self, verbose=0):
            raise ImportError("stable_baselines3 is required for barrierpg.callbacks")


# ---------------------------------------------------------------------------------------------
# Adaptive multiplier
# ---------------------------------------------------------------------------------------------
class DualAscent(BaseCallback):
    """Adaptive multiplier for a prescribed tolerance: after each rollout,
    nu <- max{0, nu + step (rate - alpha)}, rate the failure frequency among the episodes that ended
    in the rollout; pushed to every environment through set_multiplier. The environments must run
    the stage penalty='multiplier'.

    The step must respect the two time scales of the method: the multiplier should move by a small
    fraction of its eventual level per rollout (dual_step_suggestion). With too large a step the
    multiplier increases without bound when every episode fails, because the flat penalty then carries no gradient.
    A tolerance below the risk floor of the class makes the multiplier increase without bound.
    """

    def __init__(self, alpha: float, step: float, multiplier0: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.alpha, self.step, self.multiplier = float(alpha), float(step), float(multiplier0)
        self.episodes = self.failures = 0
        self.trace: List[dict] = []

    def _on_training_start(self):
        self.training_env.env_method("set_multiplier", self.multiplier)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_failed" in info:
                self.episodes += 1
                self.failures += int(info["episode_failed"])
        return True

    def _on_rollout_end(self):
        rate = self.failures / self.episodes if self.episodes else float("nan")
        if self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.step * (rate - self.alpha))
            self.training_env.env_method("set_multiplier", self.multiplier)
        self.trace.append(dict(steps=int(self.num_timesteps), multiplier=self.multiplier, episodes=self.episodes, failure_rate=rate))
        self.episodes = self.failures = 0

    def diagnose(self, beta: float, window: int = 20) -> MultiplierDiagnosis:
        return multiplier_diagnosis(self.trace, self.alpha, beta, window)


def dual_step_suggestion(expected_multiplier: float, rollouts_to_reach: int = 20) -> float:
    """A step c such that one rollout at failure rate one moves nu by expected_multiplier / rollouts_to_reach.
    expected_multiplier is the price-equivalent penalty m/(1-beta) (absorbing) or the price's tail at the
    typical failure time (physical horizon); take it from the fixed-price run that attains about the tolerance."""
    return float(expected_multiplier) / float(rollouts_to_reach)


# ---------------------------------------------------------------------------------------------
# Action relative entropy
# ---------------------------------------------------------------------------------------------
class ActionAnchor(BaseCallback):
    """Charges eta [log pi(a|y) - log rho(a|y)] as a reduction of the reward, for an on-policy Gaussian-policy
    learner (PPO): the sample-wise form of the action relative entropy A to the action reference rho.

    log_reference(obs_batch, action_batch) -> log densities of rho in the same (pre-clipping) action variable as
    the learner's log-probabilities. include_failing_action=True (default) charges every decision up to and
    including the failing one, as the paper's A does; False charges the steps before failure only.
    If the VecEnv is a VecNormalize with reward normalization, the charge is divided by the running return scale
    so that it is applied in the learner's reward units (the convention of the wear-allowance study).

    What it does: holds the policy near the action reference and keeps its trajectory risk within
    kl(q || q_rho) <= A of the reference's ('Action regularization and trajectory risk'). It adds no curvature to the learning problem and
    is charged the deficit eta C_A in the convergence budget; an entropy bonus is the case of a uniform reference.
    """

    def __init__(self, eta: float, log_reference: Callable, include_failing_action: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.eta, self.log_reference = float(eta), log_reference
        self.include_failing_action = bool(include_failing_action)
        self.total = 0.0
        self.count = 0

    def _on_step(self):
        if self.eta == 0.0:
            return True
        obs = self.model._last_obs                        # states at which the actions were drawn
        act = np.asarray(self.locals["actions"])
        logp = np.asarray(self.locals["log_probs"].detach().cpu().numpy()).reshape(-1)
        ratio = logp - np.asarray(self.log_reference(obs, act), dtype=float).reshape(-1)
        if self.include_failing_action:
            charged = np.ones_like(ratio)
        else:
            charged = np.array([not info.get("failure", False) for info in self.locals["infos"]], dtype=float)
        scale = 1.0
        venv = self.model.get_vec_normalize_env()
        if venv is not None and getattr(venv, "norm_reward", False):
            scale = float(np.sqrt(venv.ret_rms.var + venv.epsilon))
        self.locals["rewards"][:] -= (self.eta * ratio * charged / scale).astype(self.locals["rewards"].dtype)
        self.total += float(np.sum(ratio * charged))
        self.count += int(charged.sum())
        return True

    @property
    def mean_log_ratio(self) -> float:
        return self.total / self.count if self.count else float("nan")


# ---------------------------------------------------------------------------------------------
# Parameter regularization on a finite actor: pull and noise
# ---------------------------------------------------------------------------------------------
def actor_parameters(model, include_log_std: bool = True) -> list:
    """Every parameter that shapes the policy of an SB3 ActorCriticPolicy: the actor's hidden layers, the action
    head, and (if include_log_std) the state-independent dispersion log_std. The critic is excluded."""
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    if include_log_std and hasattr(policy, "log_std"):
        params.append(policy.log_std)
    return params


class ParameterAnchor:
    """The pull of the particle form on a finite actor: adds kappa_p (theta - theta_0) to the actor's gradient after
    gradient clipping and before every optimizer step. This is weight decay toward theta_0, the potential
    regularizer (kappa_p/2) |theta - theta_0|^2, which is the part of a Gaussian prior N(theta_0, sigma^2 I) a finite
    network can carry (kappa_p = kappa/sigma^2; 'The Gaussian prior as entropy, weight decay, and a policy'). Under a potential, noiseless continuous-time training
    is the population flow from the network's own empirical law ('Finite-network training under a potential').

    The prior must cover every parameter that shapes the policy, including the dispersion (log_std): anchoring the
    mean network alone leaves the dispersion free and the learner can move the whole distance between the prior
    policy and the optimum through that one parameter (the paper's parameter-anchor study). The learner inherits the prior policy, the
    actor at theta_0 with its dispersion; evaluate it as a candidate before training.

    Usage: anchor = ParameterAnchor(model, kappa_p).install()  before model.learn(...).
    theta_0 defaults to the parameters at installation.
    """

    def __init__(self, model, kappa_p: float, theta0: Optional[Sequence] = None, include_log_std: bool = True):
        import torch
        self.torch = torch
        self.model, self.kappa_p, self.include_log_std = model, float(kappa_p), bool(include_log_std)
        self.params = actor_parameters(model, include_log_std)
        self.theta0 = [p.detach().clone() for p in self.params] if theta0 is None else [torch.as_tensor(t) for t in theta0]
        self._prior_dispersion = self.dispersion()      # the dispersion at installation, also for the mean-only case

    def install(self):
        opt = self.model.policy.optimizer
        base_step = opt.step
        params, theta0, kappa, torch = self.params, self.theta0, self.kappa_p, self.torch

        def step(closure=None):
            if kappa > 0:
                with torch.no_grad():
                    for p, p0 in zip(params, theta0):
                        if p.grad is not None:
                            p.grad.add_(kappa * (p - p0))
            return base_step(closure)

        opt.step = step
        return self

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def distance(self) -> float:
        """||theta - theta_0|| over the anchored parameters."""
        with self.torch.no_grad():
            return float(self.torch.sqrt(sum(((p - p0) ** 2).sum() for p, p0 in zip(self.params, self.theta0))))

    def spread(self) -> float:
        """Per-parameter spread ||theta - theta_0|| / sqrt(d), the quantity of the paper's roles study."""
        return self.distance() / math.sqrt(max(self.n_parameters, 1))

    def regularizer_value(self) -> float:
        """kappa_p/2 ||theta - theta_0||^2, the potential regularizer's value ('Potential regularizers and weight decay')."""
        return 0.5 * self.kappa_p * self.distance() ** 2

    def dispersion(self) -> Optional[float]:
        policy = self.model.policy
        if not hasattr(policy, "log_std"):
            return None
        return float(np.exp(policy.log_std.detach().cpu().numpy()).mean())

    def effects_inputs(self) -> dict:
        """Keyword arguments for barrierpg.regularizers.regularizer_effects: distance, dispersions, whether anchored."""
        return dict(parameter_distance=self.distance(), dispersion_final=self.dispersion(), dispersion_prior=self._prior_dispersion,
                    dispersion_anchored=self.include_log_std)


class ParameterNoise:
    """The noise of the particle form on a finite actor: after every optimizer step, each actor parameter receives
    independent Gaussian noise of variance 2 kappa1 x lr, lr the current learning rate (the step of flow time,
    clipped at zero because SB3's linear schedule dips below zero on the last iteration). This is the
    finite-network counterpart of the entropy term of relative entropy, which no standard learner injects
    (the paper's particle form of the flow). Combined with a ParameterAnchor of strength kappa2, kappa2 sets where the units are held and
    kappa1 how far they spread (the paper's roles study).

    What the paper measured with PPO and Adam: the per-parameter spread grows with kappa1 but in proportion to it
    (about sqrt(2) kappa1), not as sqrt(kappa1/(kappa1+kappa2)); risk and utility stay at the anchored values while
    the spread is small against the policy's dispersion and then move, with seeds splitting; noise alone acted as
    exploration and abandoned the prior. Treat kappa1 as a setting to vary and measure, not one to set from the formula.

    Usage: noise = ParameterNoise(model, kappa1, seed=0).install()  before model.learn(...); install it together
    with a ParameterAnchor in either order.
    """

    def __init__(self, model, kappa1: float, include_log_std: bool = True, seed: Optional[int] = None,
                 theta0: Optional[Sequence] = None):
        import torch
        self.torch = torch
        self.model, self.kappa1, self.include_log_std = model, float(kappa1), bool(include_log_std)
        self.params = actor_parameters(model, include_log_std)
        self.theta0 = [p.detach().clone() for p in self.params] if theta0 is None else [torch.as_tensor(t) for t in theta0]
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.steps = 0
        self.injected_variance = 0.0     # cumulative variance per parameter, sum of 2 kappa1 lr

    def install(self):
        opt = self.model.policy.optimizer
        base_step = opt.step
        torch = self.torch

        def step(closure=None):
            out = base_step(closure)
            if self.kappa1 > 0:
                lr = max(float(opt.param_groups[0]["lr"]), 0.0)
                var = 2.0 * self.kappa1 * lr
                std = math.sqrt(var)
                with torch.no_grad():
                    for p in self.params:
                        noise = torch.randn(p.shape, generator=self.generator, dtype=p.dtype)
                        p.add_(noise.to(p.device) * std)
                self.injected_variance += var
            self.steps += 1
            return out

        opt.step = step
        return self

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def spread(self) -> float:
        """Per-parameter spread ||theta - theta_0|| / sqrt(d) from the parameters at installation."""
        with self.torch.no_grad():
            d2 = float(sum(((p - p0) ** 2).sum() for p, p0 in zip(self.params, self.theta0)))
        return math.sqrt(d2 / max(self.n_parameters, 1))


def set_log_std(model, value: float) -> None:
    """Set the state-independent dispersion of an SB3 Gaussian actor (log of the standard deviation in the
    pre-clipping action variable). The prior policy is the actor at theta_0 WITH this dispersion."""
    import torch
    with torch.no_grad():
        model.policy.log_std.fill_(float(value))


def fit_actor_mean(model, target_mean: Callable[[np.ndarray], np.ndarray], observations: np.ndarray, *, steps: int = 4000,
                   batch_size: int = 512, lr: float = 1e-3, seed: int = 0) -> float:
    """Fit the actor's mean network (pre-clipping action variable) by least squares to an incumbent rule:
    target_mean(observations) -> (n, action_dim) means. Returns the final mean squared error. The fitted
    parameters are then theta_0 of a ParameterAnchor, and the optimizer state is reset so that training starts
    fresh. Set the dispersion separately with set_log_std (a matched prior uses the incumbent's own dispersion)."""
    import torch
    policy = model.policy
    rng = np.random.default_rng(seed)
    obs = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    target = torch.as_tensor(np.asarray(target_mean(np.asarray(observations)), dtype=np.float32)).reshape(len(obs), -1)
    params = actor_parameters(model, include_log_std=False)
    opt = torch.optim.Adam(params, lr=lr)

    def mean_of(o):
        feats = policy.extract_features(o, policy.pi_features_extractor)
        return policy.action_net(policy.mlp_extractor.forward_actor(feats))

    for _ in range(int(steps)):
        idx = torch.as_tensor(rng.integers(0, len(obs), batch_size))
        loss = ((mean_of(obs[idx]) - target[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mse = float(((mean_of(obs) - target) ** 2).mean())
    policy.optimizer = type(policy.optimizer)(policy.parameters(), **policy.optimizer_kwargs, lr=policy.optimizer.param_groups[0]["lr"])
    return mse


# ---------------------------------------------------------------------------------------------
# Training monitor
# ---------------------------------------------------------------------------------------------
class BudgetMonitor(BaseCallback):
    """Records, per rollout, the episode statistics the wrapper emits: failure rate, stopped utility, interior
    loss, clipped value (with the stage's tail convention), near-exhaustion frequency, mean length, and the
    policy's log_std if it has one. diagnostics() then reads the traces.

    beta, m, horizon: the spec's; near: reserve fraction counted as near-exhaustion. Under penalty='multiplier'
    the flat penalty charged at failures is the DualAscent callback's current multiplier (dual=...) or a frozen
    value (multiplier=...).

    Convergence monitoring: convergence() checks whether the run has stopped changing (training_convergence) and
    plot() draws the training curves. With check_every=k the check runs every k batches and is kept in `checks`
    (verbose=1 prints it); with stop_when_converged=True training stops once the run has been converged at
    `patience` consecutive checks and has taken at least `min_steps` steps."""

    def __init__(self, beta: float, m: float, horizon: Optional[int] = None, near: float = 0.05, penalty: str = "price",
                 dual: Optional[DualAscent] = None, multiplier: Optional[float] = None, verbose: int = 0,
                 check_every: int = 0, window: Optional[int] = None, rel_tol: float = 0.05,
                 stop_when_converged: bool = False, patience: int = 2, min_steps: int = 0):
        super().__init__(verbose)
        self.beta, self.m, self.horizon, self.near, self.penalty = float(beta), float(m), horizon, float(near), penalty
        self.dual, self.frozen = dual, multiplier
        self.check_every, self.window, self.rel_tol = int(check_every), window, float(rel_tol)
        self.stop_when_converged, self.patience, self.min_steps = bool(stop_when_converged), int(patience), int(min_steps)
        self.checks: List[dict] = []
        self.stopped_early = False
        self._stop = False
        self._reset_acc()
        self.trace: List[dict] = []

    def _reset_acc(self):
        self._U, self._I, self._V, self._fail, self._near, self._len = [], [], [], [], [], []

    def _flat_penalty(self) -> float:
        if self.dual is not None:
            return float(self.dual.multiplier)
        if self.frozen is not None:
            return float(self.frozen)
        return float("nan")

    def _on_step(self):
        for info in self.locals["infos"]:
            rec = info.get("budget_episode")
            if rec is None:
                continue
            tail = 0.0
            if rec["failed"]:
                if self.penalty == "multiplier":
                    tail = self._flat_penalty()
                elif rec.get("geometric_stop", False):
                    tail = self.m / (1.0 - self.beta)
                else:
                    tail = failure_tail(self.m, self.beta, int(rec["tau"]), self.horizon)
            self._U.append(rec["U"]); self._I.append(rec["I"]); self._V.append(rec["U"] - rec["I"] - tail)
            self._fail.append(float(rec["failed"])); self._near.append(float(rec["min_reserve_fraction"] <= self.near))
            self._len.append(rec["length"])
        return not self._stop

    def _on_rollout_end(self):
        n = len(self._fail)
        row = dict(steps=int(self.num_timesteps), episodes=n)
        if n:
            row.update(failure_rate=float(np.mean(self._fail)), U=float(np.mean(self._U)), I_m=float(np.mean(self._I)),
                       V_m=float(np.nanmean(self._V)) if np.isfinite(self._V).any() else float("nan"),
                       near=float(np.mean(self._near)), length=float(np.mean(self._len)))
        else:
            row.update(failure_rate=np.nan, U=np.nan, I_m=np.nan, V_m=np.nan, near=np.nan, length=np.nan)
        if hasattr(self.model.policy, "log_std"):
            row["log_std"] = float(np.mean(self.model.policy.log_std.detach().cpu().numpy()))
        if self.dual is not None:
            row["multiplier"] = float(self.dual.multiplier)
        self.trace.append(row)
        self._reset_acc()
        if self.check_every and len(self.trace) % self.check_every == 0:
            c = self.convergence()
            self.checks.append(dict(steps=row["steps"], status=c.status))
            if self.verbose:
                print(f"[{row['steps']} steps] " + c.explain().split("\n")[0])
            recent = [k["status"] for k in self.checks[-self.patience:]]
            if (self.stop_when_converged and len(recent) == self.patience and all(r == "converged" for r in recent)
                    and row["steps"] >= self.min_steps):
                self._stop = self.stopped_early = True

    def series(self, key: str) -> np.ndarray:
        return np.array([t.get(key, np.nan) for t in self.trace], dtype=float)

    def convergence(self, window: Optional[int] = None, rel_tol: Optional[float] = None):
        """Has the run stopped changing? See barrierpg.convergence.training_convergence."""
        return training_convergence(self.trace, window=self.window if window is None else window,
                                    rel_tol=self.rel_tol if rel_tol is None else rel_tol)

    def plot(self, keys=("failure_rate", "U", "log_std"), title: Optional[str] = None):
        """The training curves per batch, with the windows of the convergence check shaded (needs matplotlib)."""
        return plot_training(self.trace, self.convergence(), keys=keys, title=title)

    def diagnostics(self, window: int = 20) -> dict:
        """Trend tests over the last `window` rollouts of V_m, U, the failure rate, near-exhaustion and log_std."""
        out = {}
        for key in ("V_m", "U", "failure_rate", "near", "log_std"):
            s = self.series(key)
            if np.isfinite(s).sum() >= 4:
                out[key] = trend_test(s, window)
        return out

    def explain(self, window: int = 20) -> str:
        d = self.diagnostics(window)
        if not d:
            return "no completed episodes recorded"
        lines = [v.explain(k) for k, v in d.items()]
        last = self.trace[-1]
        lines.append(f"last rollout: failure rate {last.get('failure_rate', float('nan')):.4g}, U {last.get('U', float('nan')):.4g}, "
                     f"V_m {last.get('V_m', float('nan')):.4g}")
        if "log_std" in d and d["log_std"].trend_detected:
            lines.append("the action dispersion is still changing: the candidate is not yet fixed, and a policy with a different dispersion is a different candidate")
        lines.append(f"trace diagnostics are evidence about this run, not a convergence theorem [{cite('scope')}]")
        return "\n".join(lines)


__all__ = ["DualAscent", "dual_step_suggestion", "ActionAnchor", "actor_parameters", "ParameterAnchor", "ParameterNoise",
           "set_log_std", "fit_actor_mean", "BudgetMonitor"]
