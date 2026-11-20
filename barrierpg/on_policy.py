"""Inspectable single-constraint actor updates for small PyTorch policies.

The caller supplies a fresh on-policy batch and any critic estimates. CPO
uses the mean-cost surrogate. The Gaussian CVaR update uses the cost mean
and variance surrogates of TRC. Both use the dense trust-region solver in
alternatives.py. This module has no simulator or external trainer dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm

from .alternatives import CpoUpdate, cpo_update

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution
    from torch.nn import Module


@dataclass(frozen=True)
class OnPolicyBatch:
    observations: Tensor
    actions: Tensor
    reward_advantage: Tensor
    cost_advantage: Tensor
    cost_mean: float
    cost_scale: float = 1.0


@dataclass(frozen=True)
class CostVariance:
    """Current variance and per-decision advantages for its local surrogate.

    Use variance advantages in the sense of the TRC reference implementation,
    including any discount scaling in scale. Raw squared-cost advantages
    should first be converted to the corresponding variance advantages.
    """

    value: float
    advantage: Tensor
    scale: float = 1.0


@dataclass(frozen=True)
class ActorUpdate:
    update: CpoUpdate
    utility_gain: float
    risk_before: float
    risk_after: float
    mean_kl: float


def gaussian_tail_weight(tail_mass: float) -> float:
    """Coefficient on standard deviation in upper-tail Gaussian CVaR."""
    if not 0 < tail_mass <= 1:
        raise ValueError("tail_mass must lie in (0, 1]")
    if tail_mass == 1:
        return 0.0
    return float(np.exp(norm.logpdf(ndtri(tail_mass)) - np.log(tail_mass)))


def cpo_policy_update(
    actor: Module, batch: OnPolicyBatch, *, cost_limit: float, kl_limit: float,
    action_penalty: Callable[[Tensor, Distribution], Tensor] | None = None,
    parameter_penalty: Callable[[Tensor], Tensor] | None = None,
    damping: float = 1e-3, backtracks: int = 12,
) -> ActorUpdate:
    """Take one sampled CPO actor update with an expected-cost limit.

    actor(observations) returns a distribution with one log probability per
    sampled decision. Penalize its fixed-state action distributions or its
    flat trainable parameter vector with scalar, differentiable callables.
    Their strengths belong inside the callables. Cost units are set by the
    supplied baseline, advantages, and scale.
    """
    return _policy_update(actor, batch, cost_limit, kl_limit, action_penalty,
                          parameter_penalty, damping, backtracks)


def cvar_policy_update(
    actor: Module, batch: OnPolicyBatch, variance: CostVariance, *,
    cost_limit: float, tail_mass: float, kl_limit: float,
    action_penalty: Callable[[Tensor, Distribution], Tensor] | None = None,
    parameter_penalty: Callable[[Tensor], Tensor] | None = None,
    damping: float = 1e-3, backtracks: int = 12,
) -> ActorUpdate:
    """Take a TRC-style actor update using Gaussian CVaR of cumulative cost.

    The variance surrogate and the Gaussian approximation are part of this
    update. Use upper_cvar on independent trajectories to check empirical
    tail cost, especially for skewed or discrete losses.
    """
    return _policy_update(actor, batch, cost_limit, kl_limit, action_penalty,
                          parameter_penalty, damping, backtracks, variance, gaussian_tail_weight(tail_mass))


def _policy_update(actor, batch, cost_limit, kl_limit, action_penalty,
                   parameter_penalty, damping, backtracks, variance=None, tail_weight=0.0):
    import torch
    from torch.distributions import kl_divergence
    from torch.func import functional_call

    if (not all(isfinite(value) for value in (cost_limit, batch.cost_mean, batch.cost_scale, damping))
            or damping < 0 or batch.cost_scale <= 0):
        raise ValueError("costs and damping must be finite, with positive cost scale and nonnegative damping")
    if variance is not None and (not isfinite(variance.value) or variance.value < 0
                                 or not isfinite(variance.scale) or variance.scale <= 0):
        raise ValueError("expected a nonnegative finite variance and a positive finite scale")

    parameters = {name: p for name, p in actor.named_parameters() if p.requires_grad}
    if not parameters:
        raise ValueError("actor must have trainable parameters")
    sizes = [p.numel() for p in parameters.values()]
    initial = torch.cat([p.detach().reshape(-1) for p in parameters.values()])
    # Probes use a disposable copy of registered buffers. Rejected proposals
    # leave the actor parameters and buffers unchanged.
    buffers = {name: b.detach().clone() for name, b in actor.named_buffers()}
    observations, actions = batch.observations.detach(), batch.actions.detach()
    reward_advantage, cost_advantage = batch.reward_advantage.detach(), batch.cost_advantage.detach()
    with torch.no_grad():
        old = functional_call(actor, {name: b.clone() for name, b in buffers.items()}, (observations,))
        old_log_prob = old.log_prob(actions).detach()
    if (old_log_prob.ndim != 1 or not old_log_prob.numel()
            or any(adv.shape != old_log_prob.shape for adv in (reward_advantage, cost_advantage))
            or any(not torch.isfinite(x).all() for x in (old_log_prob, reward_advantage, cost_advantage))):
        raise ValueError("each transition needs a finite scalar log probability and finite reward and cost advantages")
    variance_advantage = None if variance is None else variance.advantage.detach()
    if variance_advantage is not None and (variance_advantage.shape != old_log_prob.shape
                                          or not torch.isfinite(variance_advantage).all()):
        raise ValueError("each transition needs one finite variance advantage")

    def candidate_parameters(vector):
        return {name: piece.view_as(param)
                for (name, param), piece in zip(parameters.items(), vector.split(sizes))}

    def distribution(vector):
        candidates = candidate_parameters(vector)
        candidates.update({name: b.clone() for name, b in buffers.items()})
        return functional_call(actor, candidates, (observations,))

    def mean_kl(vector):
        return kl_divergence(old, distribution(vector)).mean()

    def objective_and_risk(vector):
        new = distribution(vector)
        ratio = torch.exp(new.log_prob(actions) - old_log_prob)
        objective = (ratio * reward_advantage).mean()
        if action_penalty is not None:
            objective = objective - action_penalty(observations, new)
        if parameter_penalty is not None:
            objective = objective - parameter_penalty(vector)
        # Center the importance correction so the supplied baseline is
        # exactly the risk's mean component at the sampling policy.
        risk = batch.cost_mean + batch.cost_scale * ((ratio - 1) * cost_advantage).mean()
        if tail_weight:
            candidate_variance = variance.value + variance.scale * ((ratio - 1) * variance_advantage).mean()
            risk = risk + tail_weight * torch.sqrt(candidate_variance.clamp_min(1e-8))
        return objective, risk

    with torch.enable_grad():
        point = initial.detach().requires_grad_()
        objective, risk = objective_and_risk(point)
        if objective.ndim or risk.ndim or not torch.isfinite(objective) or not torch.isfinite(risk):
            raise ValueError("the objective and risk must be finite scalars")
        utility_gradient, = torch.autograd.grad(objective, point, retain_graph=True)
        cost_gradient, = torch.autograd.grad(risk, point)
        fisher = torch.autograd.functional.hessian(mean_kl, point).detach().cpu().double().numpy()
    fisher = (fisher + fisher.T) / 2 + damping * np.eye(initial.numel())
    baseline_utility, baseline_risk = float(objective.detach()), float(risk.detach())

    def evaluate_trial(step):
        with torch.no_grad():
            candidate = initial + torch.as_tensor(step, device=initial.device, dtype=initial.dtype)
            next_objective, next_risk = objective_and_risk(candidate)
            kl = float(mean_kl(candidate))
            # Floating-point roundoff can make a theoretically nonnegative KL slightly negative.
            return (float(next_objective) - baseline_utility, float(next_risk),
                    max(0.0, kl) if isfinite(kl) else kl)

    def evaluate_step(step):
        gain, next_risk, kl = evaluate_trial(step)
        return gain, next_risk - cost_limit, kl

    update = cpo_update(utility_gradient.detach().cpu().double().numpy(),
                        cost_gradient.detach().cpu().double().numpy(), fisher,
                        baseline_risk - cost_limit, kl_limit, evaluate_step, backtracks=backtracks)
    if update.status == "rejected":
        return ActorUpdate(update, 0.0, baseline_risk, baseline_risk, 0.0)

    gain, next_risk, kl = evaluate_trial(update.step)
    with torch.no_grad():
        final = initial + torch.as_tensor(update.step, device=initial.device, dtype=initial.dtype)
        for name, value in candidate_parameters(final).items():
            parameters[name].copy_(value)
    return ActorUpdate(update, gain, baseline_risk, next_risk, kl)
