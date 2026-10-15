"""Before training: turn a requirement into a configuration and a validation budget, with reasons.

The plan follows the paper's 'Using the bounded barrier': specify the requirement first; fix the price, then certify;
buy margin deliberately; choose regularizers for their role and the prior from the problem; validate the
deployed candidate. It recommends by conditions and states, before anything else, the conditions of
the paper's 'Which formulation to prefer' under which another method is the better choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .bounds import Bound
from .certificates import conservative_multiplier, conservative_price
from .convergence import CurvatureBudget, curvature_budget
from .paper import cite
from .regularizers import RegularizerAdvice, advise_regularizers
from .spec import BudgetSpec, Requirement, Stage, discounted_total, failure_tail
from .validation import episodes_for_tolerance, resolution_floor, value_resolution_check


# ---------------------------------------------------------------------------------------------
# When another method is the better choice ('Which formulation to prefer')
# ---------------------------------------------------------------------------------------------
@dataclass
class Condition:
    name: str
    applies: Optional[bool]       # True, False, or None when the inputs do not say
    better_tool: str
    reason: str

    def line(self) -> str:
        status = {True: "APPLIES", False: "does not apply", None: "not stated"}[self.applies]
        return f"[{status}] {self.name}: {self.better_tool}. Reason: {self.reason}."


def _both(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is False or b is False:
        return False
    if a is True and b is True:
        return True
    return None


def when_to_use_another_method(*, failure_permissible_at_deployment: Optional[bool] = None,
                               controlled_invariant_set_known: Optional[bool] = None,
                               prescribed_level_attainable_with_slack: Optional[bool] = None,
                               multiplier_step_tunable: Optional[bool] = None,
                               violations_during_training_matter: Optional[bool] = None,
                               loss_grows_with_excess_consumption: Optional[bool] = None,
                               model_known_and_small: Optional[bool] = None,
                               price_exceeds_episode_earnings: Optional[bool] = None,
                               allowance_error_known: Optional[bool] = None) -> List[Condition]:
    """The cases of the paper's 'Which formulation to prefer' in which another method is the better choice: five
    that concern the requirement and the problem, and two that concern the barrier itself. The clipped barrier is the
    choice when failures can be permitted in simulation and deployment tolerates a positive failure probability or
    carries a finite failure cost, when the tolerance is uncertain or may be unattainable, when the failure cost is
    known, and when a reserve margin is worth its cost. The paper quantifies these choices; it does not assert an
    advantage over other methods."""
    no_failure = None if failure_permissible_at_deployment is None else (not failure_permissible_at_deployment)
    return [
        Condition("no failure is permissible and a controlled-invariant set is known", _both(no_failure, controlled_invariant_set_known),
                  "an almost-sure method (an infinite failure penalty, a control-barrier safety layer, or a method that learns which actions keep "
                  "probability-one budget constraints)",
                  "a finite price prices failure risk and does not prohibit failure; a class of full-support policies can have a positive risk floor"),
        Condition("a prescribed level is attainable with slack and the multiplier's step can be tuned",
                  _both(prescribed_level_attainable_with_slack, multiplier_step_tunable),
                  "an adaptive multiplier (a Lagrangian chance-constrained policy gradient)",
                  "a fixed price leaves the attained risk to the optimizer, and unused risk costs utility; with too large a step the multiplier "
                  "increases without bound when the initial policy fails often, as in the fuel-tank study"),
        Condition("violations during training matter", violations_during_training_matter,
                  "constrained policy optimization",
                  "the clipped barrier is trained where failures are permitted, in a simulator"),
        Condition("the loss grows with the amount by which the consumption exceeds the budget", loss_grows_with_excess_consumption,
                  "the CVaR constraint on the cumulative consumption",
                  "a failure price charges the event of exhaustion, not its size"),
        Condition("the model is known and small", model_known_and_small,
                  "an exact dynamic program (or a structured policy, when one is available); the toolbox's certificates and validation apply to it too",
                  "in the emissions application the exact dynamic program, and a re-planning rule tuned for each sanction, were better than either learner"),
        Condition("the required price charges far more for a failure than an episode earns", price_exceeds_episode_earnings,
                  "reward normalization or a prescribed-level formulation",
                  "a large price degraded the controllers of the fuel-tank study; increasing the price is not an optimization guarantee"),
        Condition("the size of a possible allowance error is known", allowance_error_known,
                  "a correspondingly tightened allowance instead of the interior margin",
                  "the interior barrier is the graded margin for an error of unknown size; a known error is covered more cheaply by tightening"),
    ]


@dataclass
class Plan:
    spec: BudgetSpec
    requirement: Requirement
    stage: Stage
    price_bound: Optional[Bound]
    validation_episodes: int
    direct_floor: float
    value_check: Optional[dict]
    multiplier_equivalent: float
    dual_step: Optional[float]
    curvature: Optional[CurvatureBudget]
    regularizers: Optional[RegularizerAdvice] = None
    conditions: List[Condition] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def another_method_preferable(self) -> bool:
        return any(c.applies for c in self.conditions)

    def explain(self) -> str:
        r, s, sp = self.requirement, self.stage, self.spec
        lines = [f"Requirement: p_T <= {r.alpha:g} at T = {r.T} ({r.T + 1} actions), validation error {r.zeta:g}"
                 + (f" shared by {r.family} statements" if r.family > 1 else "") + f"; kind = {r.kind}."]
        stated = [c for c in self.conditions if c.applies is not None]
        if stated:
            lines.append(f"When another method is the better choice [{cite('when_to_use')}]:")
            lines += ["  " + c.line() for c in self.conditions]
        if s.penalty == "multiplier":
            lines.append(f"Stage: flat failure penalty nu = {s.multiplier:g}" + (f", interior barrier theta = {s.theta:g} (clipping at -m, m = {s.m:g})" if s.theta else "")
                         + (f", action relative entropy eta = {s.eta:g}" if s.eta else "") + (f", parameter regularizer kappa = {s.kappa:g}" if s.kappa else "")
                         + "; the frozen-multiplier objective U - nu q_beta.")
        else:
            lines.append(f"Stage: price m = {s.m:g}" + (f", interior barrier theta = {s.theta:g}" if s.theta else ", failure-only price")
                         + (f", action relative entropy eta = {s.eta:g}" if s.eta else "") + (f", parameter regularizer kappa = {s.kappa:g}" if s.kappa else "")
                         + ("" if s.clip_utility else ", frozen-multiplier form (pre-failure reward unclipped)") + ".")
        if self.price_bound is not None:
            lines.append(("Penalty: " if s.penalty == "multiplier" else "Price: ") + self.price_bound.statement())
        if s.penalty == "multiplier":
            lines.append(f"Failure charge: the flat penalty nu = {s.multiplier:g}, charged once at the failing action whatever its time "
                         "(a lump sanction); it is the fixed-price optimum at m = (1-beta) nu only in the absorbing form.")
        elif sp.horizon is not None:
            lines.append(f"Failure tail under the physical horizon H = {sp.horizon}: m(1-beta^(H-t))/(1-beta) at a failure at action t, "
                         f"{failure_tail(s.m, sp.beta, 0, sp.horizon):.4g} at t = 0. A flat penalty nu matches it only at one failure time "
                         "(spec.price_from_multiplier_physical).")
        else:
            lines.append(f"Failure tail charged at a failure: {discounted_total(s.m, sp.beta):.4g} (absorbing). A multiplier method that converges to "
                         f"nu = {self.multiplier_equivalent:.4g} returns the same optimum.")
        if self.dual_step is not None:
            lines.append(f"Dual-ascent step for an adaptive multiplier at tolerance {r.alpha:g}: about {self.dual_step:.4g} per rollout "
                         f"(moves nu by a twentieth of its expected level when every episode fails).")
        lines.append(f"Validation: {self.validation_episodes} fresh episodes of the deployed candidate through action {r.T} are needed "
                     f"for the direct binomial test to be able to certify {r.alpha:g} at error {r.error_per_statement:.3g} "
                     f"(resolution floor of that many failure-free episodes: {self.direct_floor:.3g}).")
        if self.value_check is not None:
            vc = self.value_check
            lines.append(("The necessary condition for the value-based test to certify the tolerance holds"
                          if vc["attainable"] else "The value-based test cannot certify the tolerance")
                         + f" with the stated N and H: the condition's left side is {vc['floor']:.3g} against alpha = {r.alpha:g} "
                         f"(tail term {vc['tail_term']:.3g}, statistical term {vc['statistical_term']:.3g})."
                         + ("" if vc["attainable"] else " Increasing m cannot repair this; longer validation or the direct test can."))
        if self.regularizers is not None:
            lines.append(self.regularizers.explain())
        elif self.curvature is not None:
            lines.append(f"Curvature budget ({cite('flow')}):\n  " + self.curvature.explain().replace("\n", "\n  "))
        for n in self.notes:
            lines.append("Note: " + n)
        for w in self.warnings:
            lines.append("Warning: " + w)
        return "\n".join(lines)


def plan(spec: BudgetSpec, requirement: Requirement, *, sup_U: Optional[float] = None, comparator: Optional[dict] = None,
         delta: float = 0.0, stage: Optional[Stage] = None, expected_failure_rate: float = 0.0,
         validation_H: Optional[int] = None, validation_N: Optional[int] = None,
         features=None, lambda_H: Optional[float] = None,
         allowance_error_known: Optional[bool] = None, failure_permissible_at_deployment: Optional[bool] = None,
         controlled_invariant_set_known: Optional[bool] = None, prescribed_level_attainable_with_slack: Optional[bool] = None,
         multiplier_step_tunable: Optional[bool] = None, violations_during_training_matter: Optional[bool] = None,
         loss_grows_with_excess_consumption: Optional[bool] = None, model_known_and_small: Optional[bool] = None,
         episode_earnings: Optional[float] = None, regularizers: Optional[dict] = None) -> Plan:
    """Build a Plan.

    sup_U         an upper bound on the stopped utility over the class (for example the unpriced optimum's utility, or
                  ubar/(1-beta)); needed for the conservative price or penalty.
    comparator    dict(U0=, I0=, q0=) of a comparison policy of the class (default the idle policy with q0 = U0 = I0 = 0,
                  valid when standing still never fails).
    delta         optimization error to allow for.
    stage         a stage to assess; if None, the conservative price is used with no barrier.
    expected_failure_rate  the failure rate the candidate is expected to have (below alpha), to size the validation sample.
    validation_H, validation_N  for the value-based test's resolution check.
    features, lambda_H  FeatureConstants or KernelConstants and the regularizer's curvature, for the curvature budget.
    allowance_error_known  whether the size of a possible allowance error is known (margin advice).
    failure_permissible_at_deployment, controlled_invariant_set_known, prescribed_level_attainable_with_slack,
    multiplier_step_tunable, violations_during_training_matter, loss_grows_with_excess_consumption, model_known_and_small,
    episode_earnings
                  the cases of 'Which formulation to prefer' (when_to_use_another_method); episode_earnings is the utility an
                  episode earns (for example a landing bonus), against which the failure charge at the start of an
                  episode is compared: above three times, the case for another method applies; above once, a note asks
                  to watch the return in pilot runs (the tutorial campaigns: a navigation robot's return collapsed at
                  about four times, while a legged robot and a retailer did well at two to three times).
    regularizers  keyword arguments of barrierpg.regularizers.advise_regularizers (conditions of the application).
    """
    notes, warnings = [], []
    beta, ubar, u_minus = spec.beta, spec.ubar, spec.u_minus
    comp = dict(U0=0.0, I0=0.0, q0=0.0) if comparator is None else comparator
    alpha_geo = requirement.alpha * beta ** requirement.T if requirement.kind == "finite" else requirement.alpha
    # --- price or penalty ----------------------------------------------------------------------
    price_bound = None
    if sup_U is not None:
        if stage is not None and stage.penalty == "multiplier":
            price_bound = conservative_multiplier(alpha_geo, sup_U, comp["U0"], comp["q0"], delta)
        else:
            price_bound = conservative_price(alpha_geo, beta, sup_U, comp["U0"], comp["I0"], comp["q0"], delta, u_minus)
        if requirement.kind == "finite":
            notes.append(f"the price-risk law bounds q_beta; the finite-horizon target p_T <= alpha was converted to q_beta <= alpha beta^T = {alpha_geo:.4g}, "
                         "which can be very conservative at large T; validation at T is the sharper route")
        if stage is None:
            stage = Stage(m=price_bound.value if math.isfinite(price_bound.value) else 0.0)
    if stage is None:
        stage = Stage(m=0.0)
        warnings.append("no price given and no sup_U to derive one: the stage is the unpriced policy, which has no risk bound")
    # --- the cases of 'Which formulation to prefer' -----------------------------------------------
    exceeds = None
    if episode_earnings is not None and episode_earnings > 0:
        charge = stage.multiplier if stage.penalty == "multiplier" else failure_tail(stage.m, beta, 0, spec.horizon)
        ratio = charge / episode_earnings
        exceeds = ratio > 3.0
        if exceeds:
            warnings.append(f"a failure at the start of an episode is charged {charge:.4g}, {ratio:.1f} times what an episode earns "
                            f"({episode_earnings:.4g}): a learner may give up earning in order not to fail; normalize the reward, lower "
                            "the price, or use a prescribed-level method")
        elif ratio > 1.0:
            notes.append(f"a failure at the start of an episode is charged {charge:.4g}, {ratio:.1f} times what an episode earns "
                         f"({episode_earnings:.4g}); watch the return in pilot runs: a return that collapses as the price rises "
                         "means the price charges too much")
    no_failure = requirement.alpha == 0.0 or failure_permissible_at_deployment is False
    conditions = when_to_use_another_method(
        failure_permissible_at_deployment=(False if no_failure else failure_permissible_at_deployment),
        controlled_invariant_set_known=controlled_invariant_set_known,
        prescribed_level_attainable_with_slack=prescribed_level_attainable_with_slack, multiplier_step_tunable=multiplier_step_tunable,
        violations_during_training_matter=violations_during_training_matter,
        loss_grows_with_excess_consumption=loss_grows_with_excess_consumption, model_known_and_small=model_known_and_small,
        price_exceeds_episode_earnings=exceeds, allowance_error_known=allowance_error_known)
    if violations_during_training_matter:
        warnings.append("violations during training matter: the clipped barrier is trained where failures are permitted, in a simulator; "
                        "constrained policy optimization addresses violations during training")
    if requirement.kind == "eventual":
        warnings.append("an eventual requirement P(tau < inf) <= alpha is not certified by a small q_beta or by a finite-horizon test "
                        f"[{cite('delay_example')}]; it needs cost-tail structure ({cite('eventual_risk')}) or almost-sure feasibility")
    if no_failure:
        warnings.append("no failure is permissible: a positive price never prohibits failure, no multiplier converges when every policy of the "
                        "class has positive risk, and a zero-failure validation certifies only the resolution floor"
                        + ("; with a known controlled-invariant set an almost-sure method is the better choice" if controlled_invariant_set_known else
                           "; an almost-sure method needs a known controlled-invariant set"))
    if stage.penalty == "price" and stage.m < u_minus and stage.clip_utility:
        warnings.append(f"m = {stage.m:g} is below u_minus = {u_minus:g}: clipping at -m erases utility values below -m (for example a crash penalty); "
                        "choose m >= u_minus or use the frozen-multiplier form (clip_utility=False)")
    if stage.theta > 0:
        notes.append("the interior barrier buys graded protection against a misjudged allowance at a utility cost bounded by its interior loss; "
                     f"the fixed-price results apply net of that loss [{cite('fixed_price')}]")
        if allowance_error_known:
            notes.append("the size of the allowance error is known: a correspondingly tightened allowance is the more economical margin")
    elif allowance_error_known is not None:
        notes.append("a failure-only price leaves no reserve: its optimal policies exhaust the allowance whenever profitable; "
                     + ("tighten the allowance by the known error" if allowance_error_known else "an interior barrier (theta > 0) is the graded margin for an error of unknown size"))
    # --- validation -----------------------------------------------------------------------------
    err = requirement.error_per_statement
    try:
        N = episodes_for_tolerance(requirement.alpha, err, expected_failure_rate) if requirement.alpha > 0 else 0
    except ValueError:
        N = episodes_for_tolerance(requirement.alpha, err, 0.0)
        warnings.append("the expected failure rate is not below the tolerance: no validation sample certifies it; retrain at a higher price")
    if validation_N is not None and validation_N < N:
        warnings.append(f"the planned validation sample N = {validation_N} is below the {N} episodes the direct test needs at this tolerance "
                        f"(floor {resolution_floor(validation_N, err):.3g})")
    floor = resolution_floor(max(N, 1), err)
    value_check = None
    if validation_H is not None and validation_N is not None:
        value_check = value_resolution_check(beta, validation_H, requirement.T, validation_N, err, requirement.alpha)
    if requirement.family > 1:
        notes.append(f"error {requirement.zeta:g} is allocated over {requirement.family} simultaneous statements (candidates, horizons, tests); "
                     "allocate before taking minima or selecting a policy")
    # --- multiplier and dual step ------------------------------------------------------------------
    if stage.penalty == "multiplier":
        nu = stage.multiplier
        dual_step = (price_bound.value / 20.0) if (price_bound is not None and math.isfinite(price_bound.value)) else None
    else:
        nu = discounted_total(stage.m, beta) if spec.horizon is None else failure_tail(stage.m, beta, 0, spec.horizon)
        dual_step = nu / 20.0 if stage.m > 0 else None
    notes.append("a prescribed level has to be set below the tolerance by the width of the confidence bound (a multiplier that converges at the tolerance "
                 "itself does not pass validation at that tolerance), just as a fixed price has to be chosen with one")
    notes.append(f"a sequence of prices reaches a tolerance only when the learner's error grows slower than the price; check it on short pilot runs at a few prices "
                 f"with frontier.price_sequence_check [{cite('price_sequence')}]")
    # --- curvature -----------------------------------------------------------------------------------
    curv = None
    if features is not None and lambda_H is not None:
        m_eff = stage.m if stage.penalty == "price" else (1.0 - beta) * stage.multiplier
        curv = curvature_budget(features, beta, m_eff, ubar, stage.kappa, lambda_H, stage.eta)
        notes.append("set kappa for conditioning rather than at the certified threshold (near which the minimizer lies close to the prior); "
                     "treat a failed certificate as the signal to lower kappa or raise the price; the bias is toward the prior policy, "
                     "whose risk and utility the learned policy inherits")
    if stage.eta > 0:
        notes.append("the action relative entropy controls departure from the action reference's risk, not improvement: use it when the action "
                     "reference is trusted and the price or the model is not; otherwise a correctly chosen price dominates it")
    notes.append("validate the candidate that is deployed, at its specified horizon: the stochastic policy, its mean-action policy, and a policy "
                 "with a changed dispersion are different candidates")
    advice = None
    if regularizers is not None:
        advice = advise_regularizers(spec, stage, features=features, lambda_H=lambda_H, **regularizers)
    return Plan(spec, requirement, stage, price_bound, N, floor, value_check, nu, dual_step, curv, advice, conditions, notes, warnings)


__all__ = ["Condition", "when_to_use_another_method", "Plan", "plan"]
