"""Choosing regularizers and reading their effects.

Two regularizers act on different objects.

The action relative entropy (weight eta) is a reward term on the action law. It holds the policy near an
action reference rho and keeps the trajectory risk within kl(q || q_rho) <= A of the reference's
('Action regularization and trajectory risk'). It convexifies the action-law problem but not the network's, so the convergence budget
credits it with no curvature and charges it the deficit eta C_A.

The parameter regularizer (strength kappa) acts on the law of the network's units. It supplies the
curvature that makes the population flow converge ('Fixed-stage population convergence'), and its bias is toward the prior policy,
whose risk and utility the learner inherits ('The prior and the particle form of the flow'). In the particle form of the flow its parts play
different roles: a divergence injects noise into every unit's parameters, the prior's potential and any
added potential pull them toward a reference, and an integrand of faster growth limits how far the law
spreads (bpg.roles). Relative entropy alone ties noise and pull to one strength; admissible regularizers
add, so the roles can be weighted separately with one convergence argument for all of them.

Joint use: the action term pulls toward rho, the parameter term toward pi_Gamma; they coincide under the
centering construction, and their biases add in every certificate as R = eta A + kappa H.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .certificates import action_risk_bound
from .convergence import CurvatureBudget, FeatureConstants, curvature_budget, gaussian_prior_curvature
from .paper import cite
from .roles import adam_stationary_spread, stationary_spread
from .spec import BudgetSpec, Stage


# ---------------------------------------------------------------------------------------------
# Before training: which regularizer, at what strength, toward what
# ---------------------------------------------------------------------------------------------
@dataclass
class Recommendation:
    mechanism: str      # 'action anchor', 'parameter anchor', 'prior', 'strength', 'noise', 'integrand', ...
    setting: str        # what to set
    reason: str         # the condition that makes it the right tool
    result: str         # the paper result it rests on

    def line(self) -> str:
        return f"{self.mechanism}: {self.setting}\n      because {self.reason}  [{self.result}]"


@dataclass
class RegularizerAdvice:
    recommendations: List[Recommendation] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    curvature: Optional[CurvatureBudget] = None

    def explain(self) -> str:
        lines = ["Regularizers (recommended by condition; a regularizer creates no safety its target lacks):"]
        lines += ["  - " + r.line() for r in self.recommendations]
        if self.curvature is not None:
            lines.append("  curvature budget:\n    " + self.curvature.explain().replace("\n", "\n    "))
        lines += ["  caution: " + c for c in self.cautions]
        return "\n".join(lines)

    def mechanisms(self) -> List[str]:
        return [r.mechanism for r in self.recommendations]


def advise_regularizers(spec: BudgetSpec, stage: Stage, *, action_reference_trusted: Optional[bool] = None,
                        price_trusted: Optional[bool] = None, model_trusted: Optional[bool] = None,
                        incumbent_available: bool = False, risk_free_policy_available: bool = False,
                        starting_from_finite_network: bool = True, prior_sigma: Optional[float] = None,
                        want_unit_spread: bool = False, specified_divergence: Optional[str] = None,
                        optimizer_well_behaved_without_regularizer: Optional[bool] = None,
                        stalls_where_gradient_vanishes: Optional[bool] = None,
                        features=None, lambda_H: Optional[float] = None) -> RegularizerAdvice:
    """Recommend regularizers from the conditions of the application.

    action_reference_trusted   a reference action kernel rho (an operating rule, an incumbent controller) is trusted
                               to have acceptable risk;
    price_trusted / model_trusted   the price of failure risk, or the simulator, is trusted;
    incumbent_available        an incumbent policy is in operation and an actor can be fitted to it;
    risk_free_policy_available a risk-free or hard-feasible policy exists (for example standing still);
    starting_from_finite_network   the learner is a finite network (potential regularizers apply directly);
    prior_sigma                width of a Gaussian prior if one is used;
    want_unit_spread           a diversity requirement on the units relative to the prior;
    specified_divergence       a divergence the application prescribes ('kl', 'chi2', 'power r');
    optimizer_well_behaved_without_regularizer   if known from a pilot;
    stalls_where_gradient_vanishes   the learner gets stuck on a plateau of the value (a pilot shows no movement);
    features, lambda_H         FeatureConstants or KernelConstants and the regularizer's curvature, for the budget.
    """
    R, C = [], []
    beta = spec.beta
    # --- action relative entropy ----------------------------------------------------------
    if action_reference_trusted and (price_trusted is False or model_trusted is False):
        R.append(Recommendation("action anchor", f"charge eta [log pi - log rho] toward the trusted action reference (eta > 0; current eta = {stage.eta:g})",
                                "the reference is trusted and the price or the model is not: the anchor keeps the trajectory risk within "
                                "kl(q || q_rho) <= A of the reference's, which protects against a mispriced failure risk; it is not a substitute for the price",
                                f"{cite('action_risk')}; {cite('wear_study')}; {cite('guidance')}"))
        C.append("the action anchor bounds departure from the reference's risk, not improvement over it; its bound is informative only for small A, "
                 "that is, when the learned policy stays close to the reference and gains little")
    elif action_reference_trusted and price_trusted is not False and model_trusted is not False:
        R.append(Recommendation("action anchor", "not needed (eta = 0)",
                                "the price and the model are trusted: a correctly chosen price dominates the action anchor, which costs utility at a correct price",
                                f"{cite('wear_study')}; {cite('guidance')}"))
    elif action_reference_trusted is False and stage.eta > 0:
        C.append(f"a high-risk action reference argues against a large eta: the anchor pulls the learner toward the reference's risk [{cite('action_risk')}]")
    if stage.eta > 0:
        C.append("the action relative entropy is a reshaped reward: it convexifies the action-law problem, not the network's, so the convergence "
                 f"budget credits it with no curvature and charges it eta C_A; an entropy bonus is the case of a uniform action reference [{cite('flow')}]")
    # --- the prior: what the parameter regularizer is toward ------------------------------
    if incumbent_available:
        R.append(Recommendation("parameter anchor", "weight decay kappa_p (theta - theta_0) toward the parameters theta_0 fitted to the incumbent "
                                "(fit_actor_mean), covering every parameter that shapes the policy, the dispersion included (set_log_std)",
                                "an incumbent is in operation: the anchor holds the learner near it in parameter space with the curvature the action anchor "
                                "lacks; a prior on the mean network alone leaves the dispersion free and the learner escapes through it",
                                f"{cite('particle')}; {cite('gaussian_prior')}; {cite('anchor_study')}"))
        C.append("the learner inherits the prior policy's risk and utility, not the incumbent rule's: the actor at theta_0 with the learner's default "
                 "dispersion is a different policy from the rule (several times the rule's breakdowns in the paper's study); match the dispersion and "
                 "evaluate the prior policy as a candidate before training")
    elif risk_free_policy_available:
        R.append(Recommendation("prior", "a prior inducing the risk-free or hard-feasible policy (p_T(Gamma) = 0)",
                                "a risk-free policy exists: the bias then lowers risk outright, and learning improves on that policy within a risk slack "
                                "of order kappa^(-1/2)",
                                f"{cite('particle')}; {cite('prior_improvement')}"))
    else:
        R.append(Recommendation("prior", "the centering construction: center the feature at the prior so that the prior policy is the action reference rho",
                                "nothing is trusted: the neutral prior toward which the parameter regularizer and the action relative entropy pull alike, "
                                "differing only in geometry",
                                f"{cite('centering')}; {cite('particle')}"))
    R.append(Recommendation("prior policy", "evaluate the policy the prior induces as a candidate before training (its q_beta, U, near-exhaustion)",
                            "the regularized solution inherits the prior policy's risk and utility; specify the prior as the policy accepted when learning gains nothing",
                            f"{cite('particle')}; {cite('prior_improvement')}"))
    # --- strength ------------------------------------------------------------------------
    if optimizer_well_behaved_without_regularizer:
        R.append(Recommendation("strength kappa", "small or zero: the smallest kappa at which the optimizer is well behaved",
                                "the optimizer behaves without the regularizer; the regularizer then buys a bias toward the prior and no convergence benefit",
                                f"{cite('bias')}; {cite('guidance')}"))
    else:
        R.append(Recommendation("strength kappa", "the smallest kappa at which the optimizer is well behaved; never the certified threshold",
                                "the threshold is sufficient, not necessary, and grows with the price; at the threshold the minimizer lies close to the prior, "
                                "while the solvable flow converges orders of magnitude below it; a failed certificate is the signal to lower kappa or raise the price",
                                f"{cite('flow')}; {cite('bias')}; {cite('flow_test')}"))
    if prior_sigma is not None:
        lam = gaussian_prior_curvature(prior_sigma)
        R.append(Recommendation("Gaussian prior width", f"sigma = {prior_sigma:g}: weight-decay weight kappa/sigma^2 = {stage.kappa * lam:.4g} at kappa = {stage.kappa:g}",
                                "kappa KL(. || N(x0, sigma^2 I)) is negative entropy of weight kappa (the noise) plus weight decay toward x0 of weight kappa/sigma^2 "
                                "(the pull); the rate depends on (kappa, sigma) only through kappa/sigma^2, the minimizer on both; the width also changes the "
                                "induced prior policy unless the features are centered at the prior, so do not concentrate the units where the features are flat",
                                f"{cite('gaussian_prior')}; {cite('flow_test')}"))
    # --- roles: noise, pull, spread --------------------------------------------------------
    if stalls_where_gradient_vanishes:
        R.append(Recommendation("noise", "inject parameter noise of variance 2 kappa1 x lr after each optimizer step (ParameterNoise), with an anchor kappa2 "
                                "holding the units; vary kappa1 and measure the per-parameter spread",
                                "a regularizer without noise cannot move units off a region where the value gradient vanishes: a potential alone collapses the "
                                "units to the prior's center and never leaves the plateau; noise lets them reach where the feature acts. Under Adam the measured "
                                "spread was proportional to kappa1, and seeds split once it approached the policy's dispersion",
                                f"{cite('particle')}; {cite('roles')}; {cite('collapse')}"))
    if starting_from_finite_network:
        R.append(Recommendation("regularizer family", "a convex potential on the parameters: plain weight decay |x|^2/2 or an anchor |x - x_ref|^2/2 (curvature one)",
                                "the learner is a finite network: a potential is finite on its empirical law, and noiseless continuous-time training under it is "
                                "the population flow from that law, with the same contraction and rate; the relative-entropy families are infinite on an atomic law. "
                                "In the certified regime a potential alone collapses all units to one parameter",
                                f"{cite('finite_training')}; {cite('weight_decay')}; {cite('collapse')}"))
    if specified_divergence:
        R.append(Recommendation("integrand", f"match the specified divergence ({specified_divergence}); same rate and threshold, different bias",
                                "the integrand contributes no curvature, so the rate and threshold are common to the family; only the bias kappa H(mu_bar) depends on it",
                                cite("admissible_families")))
    elif want_unit_spread:
        R.append(Recommendation("integrand", "a power integrand G(v) = v^r/(r-1), r > 1, added to relative entropy",
                                "an integrand of faster growth pushes units out of regions crowded relative to the prior and limits how far the law spreads; "
                                "it also moves the minimizer closer to the prior, and needs a density estimate on a finite network",
                                f"{cite('admissible_families')}; {cite('particle')}"))
    elif not starting_from_finite_network:
        R.append(Recommendation("integrand", "relative entropy alone",
                                "it has the smallest bias of the family, H_{omega,G} >= KL(. || Gamma)",
                                cite("admissible_families")))
    # --- joint use -------------------------------------------------------------------------
    if stage.eta > 0 and stage.kappa > 0:
        R.append(Recommendation("joint effect", "both anchors: the biases add, R(mu_bar) = eta A(mu_bar) + kappa H(mu_bar), in every certificate",
                                "the action term pulls toward rho in the action law, the parameter term toward pi_Gamma in the unit law; under centering the two "
                                "targets coincide; the action term adds eta C_A to the curvature the parameter term must supply",
                                f"{cite('fixed_price', 'iv')}; {cite('matched_risk')}; {cite('flow')}"))
    curv = None
    if features is not None and lambda_H is not None:
        curv = curvature_budget(features, beta, stage.m, spec.ubar, stage.kappa, lambda_H, stage.eta)
    return RegularizerAdvice(R, C, curv)


# ---------------------------------------------------------------------------------------------
# After training: the measured effects of each regularizer on the candidate
# ---------------------------------------------------------------------------------------------
@dataclass
class RegularizerEffects:
    action: Optional[Dict] = None
    parameter: Optional[Dict] = None
    noise: Optional[Dict] = None
    inheritance: Optional[Dict] = None
    path: Optional[List[Dict]] = None
    bias: Optional[Dict] = None
    joint: Optional[str] = None
    lines: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def explain(self) -> str:
        out = ["Regularizer effects on the candidate:"] + ["  - " + x for x in self.lines]
        out += ["  flag: " + f for f in self.flags]
        return "\n".join(out)


def regularizer_effects(spec: BudgetSpec, stage: Stage, candidate: Dict, *, action_entropy: Optional[float] = None,
                        reference_risk: Optional[float] = None, parameter_distance: Optional[float] = None,
                        dispersion_final: Optional[float] = None, dispersion_prior: Optional[float] = None,
                        dispersion_anchored: bool = True, prior_policy: Optional[Dict] = None,
                        path: Optional[Sequence[Dict]] = None, A_bar: Optional[float] = None, H_bar: Optional[float] = None,
                        centered: bool = False, T: Optional[int] = None, noise_kappa1: Optional[float] = None,
                        anchor_kappa2: Optional[float] = None, spread: Optional[float] = None) -> RegularizerEffects:
    """Read the measured effects.

    candidate     dict(q_beta=, U=, q_se=, U_se=, V_m=) of the deployed candidate (V_m optional);
    action_entropy, reference_risk   A to the action reference and the reference's own risk;
    parameter_distance   ||theta - theta_0|| at the end of training; dispersion_final / dispersion_prior the action
                  dispersion at the end and at theta_0; dispersion_anchored whether the prior covered it;
    prior_policy  dict(q_beta=, U=, q_se=, U_se=, V_m=) of the prior policy evaluated as a candidate;
    path          [dict(kappa=, q_beta=, U=), ...] of candidates at several strengths (the regularization path);
    A_bar, H_bar  the regularization costs of the comparison law used in the certificates;
    centered      whether the centering construction makes rho = pi_Gamma;
    noise_kappa1, anchor_kappa2, spread   injected noise strength, anchor strength and the measured per-parameter
                  spread ||theta - theta_0||/sqrt(d) (ParameterNoise.spread or ParameterAnchor.spread).
    """
    beta, ubar = spec.beta, spec.ubar
    m = stage.m
    E = RegularizerEffects()
    # action anchor
    if action_entropy is not None and reference_risk is not None:
        b = action_risk_bound(action_entropy, reference_risk, beta, T)
        inside = candidate["q_beta"] <= b.value + 1e-12 and candidate["q_beta"] >= b.inputs["lower"] - 1e-12
        E.action = dict(eta=stage.eta, A=action_entropy, q_ref=reference_risk, bound=b.value, lower=b.inputs["lower"], informative=b.informative,
                        pinsker=b.inputs["pinsker_half_width"], inside=inside)
        E.lines.append(f"action anchor (eta = {stage.eta:g}): A = {action_entropy:.4g} to the reference (risk {reference_risk:.4g}); "
                       f"kl(q || q_rho) <= A gives q in [{b.inputs['lower']:.4g}, {b.value:.4g}], " + ("informative" if b.informative else "uninformative")
                       + f"; the candidate's q_beta = {candidate['q_beta']:.4g} " + ("lies inside the interval" if inside else "lies OUTSIDE the interval (check A and q_rho)")
                       + f" [{cite('action_risk')}]")
        if not b.informative:
            E.flags.append("the action-anchor bound is uninformative at this A: the learned policy has moved far from the reference; the anchor is not protecting the reference's risk")
    # parameter anchor (the pull)
    if parameter_distance is not None:
        esc = False
        if dispersion_final is not None and dispersion_prior is not None and dispersion_prior > 0:
            ratio = dispersion_final / dispersion_prior
            esc = (not dispersion_anchored) and (ratio > 1.5 or ratio < 1 / 1.5)
        E.parameter = dict(kappa=stage.kappa, distance=parameter_distance, dispersion_final=dispersion_final, dispersion_prior=dispersion_prior,
                           dispersion_anchored=dispersion_anchored, dispersion_escape=esc)
        line = f"parameter anchor (kappa = {stage.kappa:g}): ||theta - theta_0|| = {parameter_distance:.4g}"
        if dispersion_final is not None and dispersion_prior is not None:
            line += f"; action dispersion {dispersion_prior:.3g} -> {dispersion_final:.3g}" + ("" if dispersion_anchored else " (not anchored)")
        E.lines.append(line + f" [{cite('gaussian_prior')}; {cite('anchor_study')}]")
        if esc:
            E.flags.append("dispersion escape: the mean network stayed near theta_0 while the unanchored dispersion changed by more than 50%; a distance negligible "
                           "in parameter space is then the whole distance between the prior policy and the optimum in behavior; anchor every parameter that shapes the policy")
    # injected noise (the noise role)
    if noise_kappa1 is not None and spread is not None:
        k2 = anchor_kappa2 if anchor_kappa2 is not None else 0.0
        nominal = stationary_spread(noise_kappa1, k2) if k2 > 0 else float("nan")
        adam = adam_stationary_spread(noise_kappa1)
        E.noise = dict(kappa1=noise_kappa1, kappa2=k2, spread=spread, nominal_gradient_spread=nominal, adam_spread=adam)
        line = (f"parameter noise (kappa1 = {noise_kappa1:g}" + (f", pull kappa2 = {k2:g}" if k2 > 0 else ", no pull") + f"): per-parameter spread {spread:.3g}")
        if k2 > 0:
            line += f"; a gradient pull would hold it at sqrt(kappa1/(kappa1+kappa2)) = {nominal:.3g}"
        line += f"; under Adam the paper measured about sqrt(2) kappa1 = {adam:.3g} [{cite('roles')}]"
        E.lines.append(line)
        if dispersion_final is not None and spread >= 0.5 * dispersion_final:
            E.flags.append("the parameter spread is comparable to the action dispersion: in the paper's study risk and utility moved away from the anchored values "
                           "there, with seeds splitting between drift toward the unanchored control and degenerate policies; check every seed")
        if k2 == 0:
            E.flags.append("noise without a pull abandons the prior: in the paper's study it acted as exploration, raised the clipped objective and moved the "
                           "policy far from the incumbent")
    # inheritance
    if prior_policy is not None:
        dq = candidate["q_beta"] - prior_policy["q_beta"]; dU = candidate["U"] - prior_policy["U"]
        se_q = float(np.hypot(candidate.get("q_se", 0.0), prior_policy.get("q_se", 0.0)))
        se_U = float(np.hypot(candidate.get("U_se", 0.0), prior_policy.get("U_se", 0.0)))
        at_prior = abs(dq) <= 2 * se_q + 1e-12 and abs(dU) <= 2 * se_U + 1e-12
        E.inheritance = dict(dq=dq, dU=dU, se_q=se_q, se_U=se_U, at_prior=at_prior)
        E.lines.append(f"inheritance: candidate minus prior policy, q_beta {dq:+.4g} (se {se_q:.2g}), U {dU:+.4g} (se {se_U:.2g}); "
                       + ("the candidate sits at the prior policy within sampling error: the regularization returned the prior (no learning gain)"
                          if at_prior else "the candidate has moved from the prior policy toward the unregularized optimum")
                       + f" [{cite('particle')}; {cite('prior_improvement')}]")
        if "V_m" in candidate and "V_m" in prior_policy and candidate["V_m"] < prior_policy["V_m"] - 2 * np.hypot(candidate.get("V_se", 0.0), prior_policy.get("V_se", 0.0)):
            E.flags.append("the candidate's clipped value is below the prior policy's: for the population flow started at the prior the clipped value never falls "
                           f"below the prior's [{cite('prior_improvement')}]; check the price, the optimization and the candidate's dispersion")
        if dU < -2 * se_U and dq < -2 * se_q:
            E.lines.append(f"lower utility than the prior policy came with lower risk: the signed comparison charges the risk-price term m[q_hat - q_prior]/(1-beta) "
                           f"= {m * dq / (1 - beta):.4g}, so an improved clipped value does not by itself preserve a positive-risk prior's utility [{cite('matched_risk')}]")
    # regularization path
    if path:
        pts = sorted(path, key=lambda d: -d["kappa"])
        E.path = [dict(p) for p in pts]
        start, end = pts[0], pts[-1]
        E.lines.append(f"regularization path over kappa from {start['kappa']:g} to {end['kappa']:g}: (q_beta, U) moves from ({start['q_beta']:.4g}, {start['U']:.4g}) "
                       f"to ({end['q_beta']:.4g}, {end['U']:.4g}); the path starts at the prior policy and kappa sets how far along it the learner is "
                       f"[{cite('flow_test')}; {cite('anchor_study')}]")
        mono_q = all(a["q_beta"] <= b["q_beta"] + 1e-12 for a, b in zip(pts[:-1], pts[1:])) or all(a["q_beta"] >= b["q_beta"] - 1e-12 for a, b in zip(pts[:-1], pts[1:]))
        if not mono_q:
            E.flags.append("the risk along the regularization path is not monotone in kappa: every policy lies on one curve from the prior policy to the optimum "
                           "only when the class is one-dimensional; otherwise seeds and optimization error can reorder the path")
    # bias terms
    if A_bar is not None or H_bar is not None:
        RA = stage.eta * (A_bar or 0.0); RH = stage.kappa * (H_bar or 0.0); Rb = RA + RH
        risk_term = (1.0 - beta) * Rb / (m + ubar) if m + ubar > 0 else float("inf")
        E.bias = dict(eta_A_bar=RA, kappa_H_bar=RH, R=Rb, risk_contribution=risk_term, utility_contribution=Rb)
        E.lines.append(f"bias terms of the comparison law: eta A(mu_bar) = {RA:.4g}, kappa H(mu_bar) = {RH:.4g}; they enter the risk bound as at most "
                       f"(1-beta) R/(m+ubar) = {risk_term:.4g} and the utility shortfall as R = {Rb:.4g} [{cite('fixed_price', 'iv')}; {cite('matched_risk')}; {cite('bias')}]")
    # joint
    if stage.eta > 0 and stage.kappa > 0:
        E.joint = ("both anchors act: the action term pulls the action law toward rho, the parameter term pulls the unit law toward pi_Gamma"
                   + ("; under the centering construction these coincide, so the two pulls have one target and differ only in geometry" if centered else
                      "; without centering the two targets differ and the learner ends between them")
                   + "; the biases add in every certificate and the action term adds eta C_A to the curvature the parameter term must supply")
        E.lines.append("joint effect: " + E.joint + f" [{cite('centering')}; {cite('flow')}]")
    return E


__all__ = ["Recommendation", "RegularizerAdvice", "advise_regularizers", "RegularizerEffects", "regularizer_effects"]
