"""What each parameter regularizer does to one unit: the particle form of the flow (the paper's 'The prior and the
particle form of the flow').

Take the admissible regularizer

    H = kappa1 [ KL(. || Gamma) + omega (int G(rho) dGamma - G(1)) ] + kappa2 int P dmu,

a divergence with strength kappa1 plus a potential with strength kappa2. Along the population flow each
unit's parameters x_i move by

    dx_i = [policy gradient of V_m - eta A]  dl
           - [ kappa1 grad W(x_i) + kappa1 omega grad G'(rho_n)(x_i) + kappa2 grad P(x_i) ] dl
           + sqrt(2 kappa1) dB_i .

Each term has its own role:

  noise     the entropy part of the relative entropy is Langevin noise of variance 2 kappa1 per unit time on
            every parameter; no standard learner injects it, and it is what lets units leave a region where
            the value gradient vanishes;
  pull      the prior's potential W pulls deterministically toward the prior (for N(x0, sigma^2 I), weight
            decay toward x0 of weight kappa1/sigma^2), and a potential P is a second pull toward a reference of
            one's choice (an anchor), with no noise attached;
  spread    a faster-growing integrand G pushes units out of regions crowded relative to the prior; it needs a
            density, so a finite network carries it only through a density estimate.

Relative entropy alone (omega = 0, kappa2 = 0) ties noise and pull to one strength. The general class unties
them: kappa1 sets the noise, kappa2 the pull, the integrand the spread, and all add to the curvature budget,
kappa1 lambda_W + kappa2 lambda_P. On the solvable flow a potential alone collapses the units to the prior's
center and never leaves the plateau; relative entropy escapes; the sum escapes once its stationary spread is
comparable to the plateau's width (the paper's study of the roles of the regularizers).

Finite networks. The anchor (callbacks.ParameterAnchor) is the potential term; injected Gaussian noise of
variance 2 kappa1 times the learning rate after each optimizer step (callbacks.ParameterNoise) is the noise term.
Under a plain gradient step the stationary spread under a quadratic pull is sqrt(kappa1/(kappa1/sigma^2 +
kappa2 lambda_P)). The anchor's default form realizes that step: the pull is applied to the parameters after the
optimizer's step (decoupled), so every coordinate contracts by 1 - lr kappa2 per step whatever the optimizer. The
paper's runs used the coupled form, the pull added to the gradient before an Adam step, which rescales it
coordinate by coordinate to a step of about lr whatever the distance; there the paper measured a spread
proportional to kappa1 instead (about sqrt(2) kappa1, adam_stationary_spread). The decoupling of roles survived
the finite network; the stationary-spread formula holds for the decoupled pull and not for the coupled one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .paper import cite


@dataclass(frozen=True)
class RegularizerMix:
    """A divergence with strength kappa1 (prior N(x0, sigma^2 I), integrand weight omega) plus a quadratic
    potential lambda_P |x - x_ref|^2 / 2 with strength kappa2 (lambda_P = 1 is plain weight decay or anchoring)."""
    kappa1: float = 0.0
    kappa2: float = 0.0
    sigma: float = 1.0
    lambda_P: float = 1.0
    omega: float = 0.0
    same_center: bool = True

    def __post_init__(self):
        if self.kappa1 < 0 or self.kappa2 < 0 or self.sigma <= 0 or self.lambda_P < 0 or self.omega < 0:
            raise ValueError("strengths, omega and lambda_P are nonnegative, sigma positive")

    # --- roles -----------------------------------------------------------------------------
    @property
    def noise_variance_rate(self) -> float:
        """Variance of the Langevin noise per unit of flow time, per parameter: 2 kappa1."""
        return 2.0 * self.kappa1

    @property
    def pull_strength(self) -> float:
        """Total linear pull per parameter, kappa1/sigma^2 + kappa2 lambda_P (toward a common center when same_center)."""
        return self.kappa1 / self.sigma ** 2 + self.kappa2 * self.lambda_P

    @property
    def curvature(self) -> float:
        """kappa1 lambda_W + kappa2 lambda_P, the curvature the mix supplies to the curvature budget of the convergence theorem."""
        return self.kappa1 / self.sigma ** 2 + self.kappa2 * self.lambda_P

    def stationary_spread(self) -> float:
        """Per-parameter standard deviation of the units around the common center under a gradient pull, ignoring
        the value gradient: sqrt(kappa1 / (kappa1/sigma^2 + kappa2 lambda_P)); sqrt(kappa1/(kappa1+kappa2)) for the
        paper's N(0,1) prior with weight decay. Zero without noise (the units collapse)."""
        if self.kappa1 == 0.0:
            return 0.0
        return math.sqrt(self.kappa1 / self.pull_strength)

    def noise_std_per_step(self, learning_rate: float) -> float:
        """Standard deviation of the noise to add per parameter after an optimizer step with this learning rate,
        sqrt(2 kappa1 lr): the learning rate is the step of flow time."""
        return math.sqrt(2.0 * self.kappa1 * max(learning_rate, 0.0))

    @property
    def collapses(self) -> bool:
        """A potential alone (no noise, no divergence) moves the population to a single point mass in the certified
        regime: all units at one parameter, whatever the width ('Collapse under a potential alone')."""
        return self.kappa1 == 0.0 and self.kappa2 > 0.0

    def finite_network_form(self) -> List[str]:
        out = []
        if self.kappa1 > 0:
            out.append(f"noise: add N(0, 2 x {self.kappa1:g} x lr) to every actor parameter after each optimizer step (ParameterNoise)")
            out.append(f"prior pull: weight decay toward the prior center with weight {self.kappa1 / self.sigma ** 2:g} (ParameterAnchor)")
        if self.kappa2 > 0:
            out.append(f"potential pull: theta <- theta - lr x {self.kappa2 * self.lambda_P:g} (theta - theta_ref) after each step (ParameterAnchor, decoupled)")
        if self.omega > 0:
            out.append("integrand: no finite-network form without a density estimate of the units")
        return out

    def explain(self) -> str:
        lines = [f"regularizer mix: kappa1 = {self.kappa1:g} (divergence to N(x0, {self.sigma:g}^2 I)), omega = {self.omega:g}, "
                 f"kappa2 = {self.kappa2:g} (potential, curvature {self.lambda_P:g})",
                 f"  noise: variance {self.noise_variance_rate:g} per unit time per parameter"
                 + (" (none: units cannot leave a region where the value gradient vanishes)" if self.kappa1 == 0 else ""),
                 f"  pull: {self.pull_strength:g} per parameter toward " + ("the common center" if self.same_center else "two centers (prior and reference)"),
                 f"  curvature supplied: {self.curvature:g}",
                 f"  stationary spread under a gradient pull: {self.stationary_spread():.4g} per parameter"]
        if self.collapses:
            lines.append(f"  a potential alone collapses the population to one point mass in the certified regime [{cite('collapse')}]")
        if self.kappa1 > 0:
            lines.append(f"  with the coupled pull under Adam (the paper's runs) the measured spread was about sqrt(2) kappa1 = "
                         f"{adam_stationary_spread(self.kappa1):.3g} instead (post-hoc explanation, not a result) [{cite('roles')}]")
        if self.kappa1 > 0 and self.kappa2 == 0 and self.omega == 0:
            lines.append("  relative entropy alone: noise and pull tied to one strength")
        lines.append(f"  [{cite('particle')}]")
        return "\n".join(lines)


def stationary_spread(kappa1: float, kappa2: float, sigma: float = 1.0, lambda_P: float = 1.0) -> float:
    """sqrt(kappa1 / (kappa1/sigma^2 + kappa2 lambda_P)): see RegularizerMix.stationary_spread. It applies to a
    decoupled pull (ParameterAnchor's default) with ParameterNoise; not for the coupled pull under Adam."""
    return RegularizerMix(kappa1, kappa2, sigma, lambda_P).stationary_spread()


def adam_stationary_spread(kappa1: float) -> float:
    """sqrt(2) kappa1: the stationary spread of a random walk of diffusion 2 kappa1 lr against a pull of fixed step
    size lr, a Laplace law independent of kappa2 and lr. It describes the COUPLED form of the anchor (the pull added to
    the gradient before an Adam step, ParameterAnchor(decoupled=False), the form of the paper's PPO runs), where Adam's
    per-coordinate normalization makes the restoring step about lr whatever the distance. The explanation the paper
    gave after its runs; post hoc, one example. The decoupled form (the default) follows stationary_spread instead."""
    return math.sqrt(2.0) * kappa1


def kappa1_for_spread(spread: float, kappa2: float, sigma: float = 1.0, lambda_P: float = 1.0) -> float:
    """The noise strength giving a target stationary spread s under a gradient pull: kappa1 = s^2 kappa2 lambda_P /
    (1 - s^2/sigma^2); needs s < sigma (with the prior's own pull alone the spread cannot exceed sigma)."""
    s2 = spread ** 2
    if s2 >= sigma ** 2:
        return math.inf
    return s2 * kappa2 * lambda_P / (1.0 - s2 / sigma ** 2)


__all__ = ["RegularizerMix", "stationary_spread", "adam_stationary_spread", "kappa1_for_spread"]
