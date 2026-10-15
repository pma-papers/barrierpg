"""After training: in what sense does the candidate solve the problem, and what is not established.

The report assembles, for one deployed candidate, the fixed-price reading (the fixed-price proposition), the
certificates that apply to its measured quantities, the validation at the specified horizon (the
deployment decision), the measured effects of the regularizers, and the explicit non-claims the paper
draws (eventual risk, almost-sure feasibility, other horizons, other candidates, discrete stochastic
training). If the optimization error is unknown, no optimality claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .bounds import Bound
from .certificates import (action_risk_bound, comparison_law_risk, finite_from_geometric, near_exhaustion_bound,
                           price_risk_law, risk_from_value)
from .convergence import MultiplierDiagnosis, SeedDispersion, TrendTest
from .evaluate import Evaluation
from .regularizers import RegularizerEffects
from .spec import BudgetSpec, Requirement, Stage, price_from_multiplier
from .paper import cite
from .validation import DirectValidation, lump_sanction_risk, physical_horizon_risk


@dataclass
class Report:
    spec: BudgetSpec
    requirement: Requirement
    stage: Stage
    evaluation: Evaluation
    validation: Optional[DirectValidation]
    bounds: List[Bound] = field(default_factory=list)
    effects: Optional[RegularizerEffects] = None
    reading: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    not_established: List[str] = field(default_factory=list)
    candidate: str = "the final stochastic policy"
    multiplier: Optional[MultiplierDiagnosis] = None

    @property
    def certified(self) -> bool:
        return self.validation is not None and self.validation.passed

    def best_bound(self) -> Optional[Bound]:
        cands = [b for b in self.bounds if b.name.startswith(f"p_{self.requirement.T}")]
        return min(cands, key=lambda b: b.value) if cands else None

    def summary(self, resource: str = "the budget", period: str = "steps") -> str:
        """The verdict in plain words, without the theory: how often the candidate ran out on fresh episodes, the
        validated bound on its risk, whether the requirement is met, its utility, and what the test does not show.
        text() gives every bound with its assumptions."""
        r, s, e, v = self.requirement, self.stage, self.evaluation, self.validation
        steps = f"{r.T + 1} {period}"
        who = self.candidate[0].upper() + self.candidate[1:]
        if e.event == "q_beta":
            lines = [f"{who} was run on {e.n} fresh episodes, each ending at a random time (the geometric clock). "
                     f"It ran out of {resource} in {e.failures} of them ({e.failures / e.n:.2%})."]
        else:
            lines = [f"{who} was run on {e.n} fresh episodes. It ran out of {resource} within {steps} in {e.failures} of them "
                     f"({e.failures / e.n:.2%})."]
        if v is not None:
            share = f" (the error {r.zeta:g} is shared among {r.family} candidates)" if r.family > 1 else ""
            what = "of running out before the episode's random end" if v.event == "q_beta" else f"of running out within {steps}"
            lines.append(f"With confidence {1 - r.zeta:g}, its probability {what} is at most {v.upper:.3%}{share}.")
            if v.passed:
                lines.append(f"Requirement (at most {r.alpha:.2%}): met.")
            else:
                floor = (f" With this sample, even no failures would show no less than {v.floor:.3%}." if v.failures == 0 else "")
                lines.append(f"Requirement (at most {r.alpha:.2%}): not shown. The test does not prove a violation; a larger test "
                             f"may certify the policy, or the policy may not meet the tolerance.{floor}")
        lines.append(f"Utility per episode: {e.U:.4g} (standard error {e.U_se:.2g})."
                     + (f" The reserve margin cost {e.I_m:.4g} of it." if s.theta else ""))
        near = e.near.get(0.05)
        if near is not None:
            lines.append(f"Episodes that came within 5% of running out, or ran out: {near:.1%}.")
        if self.multiplier is not None:
            lines.append({"stable": "The adaptive multiplier converged during training.",
                          "trending": "The adaptive multiplier was still moving when training ended.",
                          "increasing": "The adaptive multiplier rose while every episode failed; the run did not learn to avoid failure.",
                          "inactive": "The adaptive multiplier stayed at zero: the tolerance did not constrain the policy.",
                          "off-target": "The adaptive multiplier stopped moving away from the tolerance.",
                          "short": "The adaptive multiplier's record is too short to judge."}[self.multiplier.status])
        lines += ["", "What this test does not show:",
                  "  - that the policy is the best possible at this risk: how far training fell short of the optimum is not measured;",
                  f"  - its risk over more than {steps}, or that of other policies (its mean action, a retrained network);",
                  "  - its behavior outside the simulator.",
                  "Every bound with its assumptions: report.text()."]
        return "\n".join(lines)

    def text(self) -> str:
        r, s, e = self.requirement, self.stage, self.evaluation
        if s.penalty == "price":
            who = f"Candidate under price m = {s.m:g}"
        elif self.spec.horizon is not None:
            who = f"Candidate under the flat failure penalty nu = {s.multiplier:g}, charged once under the physical horizon H = {self.spec.horizon}"
        else:
            who = (f"Candidate under the flat failure penalty nu = {s.multiplier:g} "
                   f"(price-equivalent m = (1-beta) nu = {price_from_multiplier(s.multiplier, self.spec.beta):.4g})")
        head = (who
                + (f", barrier theta = {s.theta:g}" if s.theta else "")
                + f"; requirement p_{r.T} <= {r.alpha:g} at error {r.zeta:g}" + (f"/{r.family}" if r.family > 1 else "") + ".")
        lines = [head, "", "Measured on fresh episodes:", "  " + e.summary().replace("\n", "\n  "), ""]
        lines.append("Verdict: " + ("CERTIFIED by direct validation at the specified horizon." if self.certified else
                                    "NOT certified at the specified horizon (failure to certify does not prove violation)."))
        lines += ["", "In what sense the candidate is a solution:"] + ["  - " + x for x in self.reading]
        lines += ["", "Evidence and bounds:"] + ["  - " + x for x in self.evidence]
        if self.effects is not None:
            lines += ["", self.effects.explain()]
        if self.bounds:
            lines += ["", "Bounds (value, source, hypotheses):"] + ["  " + b.statement().replace("\n", "\n  ") for b in self.bounds]
        lines += ["", "Not established by this evidence:"] + ["  - " + x for x in self.not_established]
        return "\n".join(lines)


def interpret(spec: BudgetSpec, requirement: Requirement, stage: Stage, evaluation: Evaluation, *,
              delta: Optional[float] = None, sup_U: Optional[float] = None, comparator: Optional[dict] = None,
              action_entropy: Optional[float] = None, reference_risk: Optional[float] = None,
              lower_barrier: Optional[float] = None, multiplier: Optional[MultiplierDiagnosis] = None,
              trends: Optional[dict] = None, seeds: Optional[dict] = None, exact_frontier_gap: Optional[float] = None,
              candidate_name: str = "the final stochastic policy", effects: Optional[RegularizerEffects] = None) -> Report:
    """Compose the report for one candidate.

    delta          optimization error (measured against an exact frontier, or a conservative allowance);
                   None means unknown and the optimality statements are stated conditionally.
    sup_U          upper bound on the stopped utility over the class (price-risk law).
    comparator     dict(U0=, I0=, q0=) of a comparison policy (idle by default) for the price-risk law;
                   optionally also C_B= and R= for the learning form of the fixed-price proposition.
    action_entropy, reference_risk   A of the candidate and the risk of the action reference (the action-risk proposition).
    lower_barrier  underline B(eps) at a chosen reserve threshold (near-exhaustion certificate).
    multiplier     diagnosis of a dual-ascent trace (adaptive configurations).
    trends         BudgetMonitor.diagnostics() of the training run.
    seeds          dict of name -> SeedDispersion across training seeds.
    exact_frontier_gap  relative shortfall from an exact frontier if one exists.
    effects        bpg.regularizers.regularizer_effects(...) of the candidate (action anchor, parameter anchor,
                   inheritance, regularization path, bias terms, joint effect).
    """
    beta, ubar, T = spec.beta, spec.ubar, requirement.T
    m = stage.m if stage.penalty == "price" else price_from_multiplier(stage.multiplier, beta)   # price-equivalent of a flat penalty
    bounds, reading, evidence, missing = [], [], [], []
    # --- validation at the specified horizon --------------------------------------------
    val = None
    if evaluation.event == "q_beta":
        # geometric clock: the validated event is q_beta; for a finite requirement use beta^T p_T <= q_beta
        target = requirement.alpha if requirement.kind == "geometric" else requirement.alpha * beta ** T
        val = evaluation.validate(target, requirement.zeta, requirement.family)
        bounds.append(val.bound())
        evidence.append(f"direct validation under the geometric clock: {val.failures} failures in {val.n} episodes; exact binomial upper endpoint "
                        f"{val.upper:.4g} on q_beta at error {val.err:.3g} (resolution floor {val.floor:.3g})")
        if requirement.kind != "geometric":
            b = finite_from_geometric(val.upper, beta, T); b.name = f"p_{T} upper bound (from validated q_beta)"
            b.source = cite("validation") + " of q_beta, with beta^T p_T <= q_beta"; bounds.append(b)
            evidence.append(f"finite-horizon requirement validated through q_beta <= alpha beta^T = {target:.4g}: " + ("passes" if val.passed else "does not pass")
                            + f" (p_{T} <= {b.value:.4g}); p_T itself is censored by the clock, so evaluate without geometric stopping to test it directly")
        else:
            evidence.append(("passes" if val.passed else "does not pass") + f" the geometric tolerance {requirement.alpha:g}")
    elif evaluation.T == T:
        val = evaluation.validate(requirement.alpha, requirement.zeta, requirement.family)
        bounds.append(val.bound(T))
        evidence.append(f"direct validation: {val.failures} failures in {val.n} episodes through action {T}; exact binomial upper endpoint "
                        f"{val.upper:.4g} at error {val.err:.3g} (resolution floor {val.floor:.3g}); " + ("passes" if val.passed else "does not pass")
                        + f" the tolerance {requirement.alpha:g}")
    else:
        missing.append(f"the evaluation counted failures at T = {evaluation.T}, not at the requirement's T = {T}: re-evaluate at T to validate")
    # --- the fixed-price reading ---------------------------------------------------------
    if delta is None:
        reading.append("Optimization error is unknown: no optimality or convergence claim is made for this candidate. "
                       "The finite-price objective trades stopped utility and interior loss against failure risk; "
                       "direct validation assesses the executed candidate independently of optimization.")
        if stage.penalty == "multiplier" and multiplier is not None:
            evidence.append(f"multiplier diagnosis: {multiplier.status}; {multiplier.explanation}")
    elif stage.penalty == "price" and m > 0:
        d = "exactly" if delta == 0 else (f"up to the optimization error delta = {delta:.3g}" if delta is not None else "up to its (unmeasured) optimization error")
        reading.append(f"Fixed-price reading ({cite('fixed_price', 'i')}): among the policies of its class with no greater geometric risk, "
                       f"{candidate_name} maximizes U - I_m {d}; it is optimal at the risk it attains, q_beta = {evaluation.q_beta:.4g}. "
                       "Where on the frontier it lies is not controlled by m. "
                       + (f"Its risk is below the geometric level alpha beta^T = {requirement.alpha * beta ** T:.4g}; the unused risk costs utility at the price, "
                          f"m[alpha beta^T - q]/(1-beta) = {m * (requirement.alpha * beta ** T - evaluation.q_beta) / (1 - beta):.4g} ({cite('matched_risk')})."
                          if requirement.alpha * beta ** T > evaluation.q_beta else
                          f"Its risk is above the geometric level alpha beta^T = {requirement.alpha * beta ** T:.4g}, so it has no unused risk at that level; "
                          "validation decides the finite-horizon requirement."))
    elif stage.penalty == "multiplier":
        if spec.horizon is not None:
            reading.append(f"Frozen-multiplier reading: the flat penalty nu = {stage.multiplier:g} is charged once at the failing action (a lump sanction "
                           f"under the physical horizon H = {spec.horizon}); the objective is U - nu E[beta^tau], and the fixed-price results apply with "
                           f"I_m = 0 [{cite('fixed_price')}]: among the policies of its class with no greater discounted failure probability it maximizes U"
                           + (f" up to the optimization error delta = {delta:.3g}" if delta is not None else " up to its (unmeasured) optimization error") + ".")
        else:
            reading.append(f"Frozen-multiplier reading: a multiplier that has converged to nu returns the fixed-price optimum for m = (1-beta) nu = "
                           f"{m:.4g} ({cite('fixed_price')} with I_m = 0): among the policies of its class with no greater geometric risk it maximizes U"
                           + (f" up to the optimization error delta = {delta:.3g}" if delta is not None else " up to its (unmeasured) optimization error")
                           + f"; it is optimal at the risk it attains, q_beta = {evaluation.q_beta:.4g}. The certificates below use the price-equivalent m.")
        if multiplier is not None:
            evidence.append(f"multiplier diagnosis: {multiplier.status}; {multiplier.explanation}")
            if multiplier.status != "stable":
                missing.append("the multiplier has not converged, so the candidate is not a fixed-price optimum at any identified price")
    else:
        reading.append("Unpriced learner (m = 0): no risk is priced; the candidate maximizes utility and has no risk bound from the formulation.")
    if stage.theta > 0:
        reading.append(f"Interior barrier: the candidate pays an interior loss I_m = {evaluation.I_m:.4g} for keeping a margin before exhaustion; "
                       f"its near-exhaustion frequency is {evaluation.near.get(0.05, float('nan')):.3g} at 5% of the allowance.")
    # --- certificates from the measured value ------------------------------------------
    if m > 0:
        V_low = evaluation.V_m - 2.0 * evaluation.V_m_se
        b = risk_from_value(V_low, m, ubar, beta, T)
        b.note = (b.note or "") + " ; V_m taken as the point estimate minus two standard errors (descriptive, not a Hoeffding bound)"
        bounds.append(b)
        if sup_U is not None and delta is not None:
            comp = dict(U0=0.0, I0=0.0, q0=0.0) if comparator is None else comparator
            bounds.append(price_risk_law(m, beta, sup_U, comp.get("U0", 0.0), comp.get("I0", 0.0), comp.get("q0", 0.0), delta or 0.0, ubar))
            bounds.append(finite_from_geometric(bounds[-1].value, beta, T))
            if "C_B" in comp and "R" in comp:
                bounds.append(comparison_law_risk(comp.get("q0", 0.0), comp["C_B"], comp["R"], delta or 0.0, m, ubar, beta, T=T))
        if stage.penalty == "price" and spec.horizon is not None and not evaluation.geometric_stop and T < spec.horizon:
            bounds.append(physical_horizon_risk(evaluation.V_m, spec.horizon, m, ubar, beta, T, N=evaluation.n,
                                                zeta=requirement.error_per_statement))
        if stage.penalty == "multiplier" and spec.horizon is not None and stage.multiplier > 0 and not evaluation.geometric_stop:
            bounds.append(lump_sanction_risk(evaluation.V_m, spec.horizon, stage.multiplier, ubar, beta, T, N=evaluation.n,
                                             zeta=requirement.error_per_statement, u_minus=spec.u_minus))
        if lower_barrier is not None and stage.theta > 0:
            bounds.append(near_exhaustion_bound(V_low, m, ubar, beta, T, lower_barrier))
    if action_entropy is not None and reference_risk is not None:
        bounds.append(action_risk_bound(action_entropy, reference_risk, beta))
        evidence.append(f"action relative entropy A = {action_entropy:.4g} to the action reference (risk {reference_risk:.4g}): the departure bound "
                        f"is {'informative' if bounds[-1].informative else 'uninformative'} at this A [{cite('action_risk')}]")
    # --- convergence evidence -----------------------------------------------------------
    if trends:
        for k, t in trends.items():
            evidence.append("training trace, " + t.explain(k))
        if any(t.trend_detected for t in trends.values()):
            missing.append("some monitored quantities of the training run were still changing; the candidate is the policy at the end of the run, not a converged one")
    if seeds:
        for k, sdp in seeds.items():
            evidence.append(sdp.explain(k))
    if exact_frontier_gap is not None:
        evidence.append(f"shortfall from the exact frontier at the attained risk: {exact_frontier_gap:.2%} (the measured optimization error)")
    elif delta is None:
        missing.append("the optimization error is not measured: price-comparison bounds are not evaluated with delta = 0; "
                       "independent validation is the operative deployment evidence. Descriptive value bounds remain conditional on a valid value lower bound")
    # --- standing non-claims --------------------------------------------------------------
    missing += [
        f"eventual risk: a small q_beta or a certificate at T does not bound P(tau < inf) ({cite('delay_example')}); a hard almost-sure budget is the case alpha = 0 of that requirement",
        f"other horizons: certification at T = {T} does not certify a longer horizon; the finite-horizon bounds carry the factor beta^-T",
        "other candidates: the mean-action policy, a policy with reduced dispersion, or a retrained one is a different candidate and needs its own validation",
        f"convergence of this training run: the convergence theorem concerns the exact population flow [{cite('flow')}], and continuous-time "
        f"training of a finite network under a potential is that flow [{cite('finite_training')}]; discrete steps with PPO's or SAC's estimators are "
        f"outside both, so the trace diagnostics are evidence about this run, not a convergence theorem [{cite('scope')}]",
        "model error: every certificate and validation is conditional on the simulator",
    ]
    if stage.kappa > 0:
        missing.append("the regularization bias is toward the prior policy; the certificates are evaluated on the learned policy, so a harmful bias shows as a "
                       f"failed certificate, never as a passed one [{cite('bias')}]")
    if effects is not None:
        missing += ["regularizer flag: " + f for f in effects.flags]
    return Report(spec, requirement, stage, evaluation, val, bounds, effects, reading, evidence, missing, candidate_name, multiplier)


__all__ = ["Report", "interpret"]
