"""Convex action penalties and parameter potentials for shared objectives."""
from __future__ import annotations

from math import isfinite

import numpy as np


def categorical_action_divergence(policy, reference, *, form: str = "kl"):
    """Divergence per state for categorical policies, summed over the last axis.

    NumPy arrays and differentiable Torch tensors are supported. Inputs are
    probabilities with a common action ordering. For KL the reference must
    give positive probability to every action used by the policy. A learned
    policy should have positive probabilities when differentiating Hellinger
    or KL, since their gradients can be singular at the boundary.
    """
    if hasattr(policy, "detach"):
        import torch as xp

        reference = xp.as_tensor(reference, dtype=policy.dtype, device=policy.device).detach()
    else:
        xp = np
        policy = np.asarray(policy, dtype=float)
        reference = np.asarray(reference, dtype=float)
    if (policy.ndim == 0 or reference.ndim == 0 or policy.shape[-1] != reference.shape[-1]
            or not policy.shape[-1]):
        raise ValueError("policy and reference need the same nonempty action axis")
    for probabilities in (policy, reference):
        if (not bool(xp.isfinite(probabilities).all()) or bool((probabilities < 0).any())
                or not bool(xp.allclose(probabilities.sum(-1), xp.ones_like(probabilities.sum(-1))))):
            raise ValueError("action probabilities must be nonnegative and sum to one")
    if form == "squared":
        return 0.5 * ((policy - reference) ** 2).sum(-1)
    if form == "hellinger":
        return 0.5 * ((xp.sqrt(policy) - xp.sqrt(reference)) ** 2).sum(-1)
    if form == "kl":
        positive = policy > 0
        if bool((positive & (reference == 0)).any()):
            raise ValueError("the reference must cover the policy's support for finite KL")
        safe_policy = xp.where(positive, policy, 1.0)
        safe_reference = xp.where(positive, reference, 1.0)
        return (policy * (xp.log(safe_policy) - xp.log(safe_reference))).sum(-1)
    raise ValueError("form must be 'kl', 'hellinger' or 'squared'")


def parameter_potential(displacement, *, form: str = "quadratic",
                        scale: float = 1.0, l1: float = 0.0):
    """Sum a coordinatewise convex potential of theta minus the reference.

    Supports differentiable Torch tensors and NumPy arrays. Pseudo-Huber has
    quadratic behavior near zero and a bounded slope far from the reference.
    Elastic net adds l1 * abs(displacement) to the quadratic penalty.
    """
    if not hasattr(displacement, "detach"):
        displacement = np.asarray(displacement, dtype=float)
    if form == "quadratic":
        return 0.5 * (displacement ** 2).sum()
    if form == "pseudo_huber":
        if not isfinite(scale) or scale <= 0:
            raise ValueError("pseudo-Huber scale must be positive and finite")
        return (displacement ** 2 / ((1 + (displacement / scale) ** 2) ** 0.5 + 1)).sum()
    if form == "elastic_net":
        if not isfinite(l1) or l1 <= 0:
            raise ValueError("the elastic-net l1 coefficient must be positive and finite")
        return (0.5 * displacement ** 2 + l1 * abs(displacement)).sum()
    raise ValueError("form must be 'quadratic', 'pseudo_huber' or 'elastic_net'")


def parameter_law_divergence(law, prior, *, weight: float = 0.0, power: float = 2.0):
    """KL plus a power divergence on a common finite parameter-law grid.

    The inputs give probability masses, with the prior covering the law.
    Power 2 adds Pearson chi-squared. This evaluates a discrete reference
    measure or a grid approximation to a density-based objective.
    """
    if not isfinite(weight) or weight < 0 or not isfinite(power) or power <= 1:
        raise ValueError("weight must be finite and nonnegative and power must exceed one")
    kl = categorical_action_divergence(law, prior, form="kl")
    if weight == 0:
        return kl
    if hasattr(law, "detach"):
        import torch as xp

        prior = xp.as_tensor(prior, dtype=law.dtype, device=law.device).detach()
    else:
        xp = np
        law, prior = np.asarray(law, dtype=float), np.asarray(prior, dtype=float)
    ratio = law / xp.where(prior > 0, prior, 1.0)
    if power == 2:
        divergence = (prior * (ratio - 1) ** 2).sum(-1)
    else:
        divergence = ((prior * ratio ** power).sum(-1) - 1) / (power - 1)
    return kl + weight * divergence


def regularized_utility(utility, *, action_penalty=0.0, eta: float = 0.0,
                        parameter_penalty=0.0, kappa: float = 0.0):
    """Combine comparable expected returns and pre-aggregated penalties.

    Pass Torch tensors to differentiate a scalar objective for a CPO step or
    NumPy arrays for candidate selection under a scenario risk constraint.
    Action penalties must already use the intended occupancy and discount.
    """
    if not all(isfinite(weight) and weight >= 0 for weight in (eta, kappa)):
        raise ValueError("regularization strengths must be finite and nonnegative")
    if eta:
        utility = utility - eta * action_penalty
    if kappa:
        utility = utility - kappa * parameter_penalty
    return utility
