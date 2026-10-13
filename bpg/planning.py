"""Before training: turn a requirement into a configuration and a validation budget, with reasons.

The plan follows Section 7.1 of the paper: specify the requirement first; fix the price, then certify;
buy margin deliberately; choose regularizers for their role and the prior from the problem; validate the
deployed candidate. It recommends by conditions and states, before anything else, the conditions of
Section 1.2 under which another method is the better choice.
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
# When another method is the better choice (Section 1.2)
# ---------------------------------------------------------------------------------------------
@dataclass
class Condition:
    name: str
    applies: Optional[bool]       # True, False, or None when the inputs do not say
    better_tool: str
    reason: str

    def line(self) -> str:
        status = {True: "APPLIES", False: "does not apply", None: "not stated"}[self.applies]
        return f"[{status}] {self.name}: {self.better_tool}. {self.reason}"


def when_to_use_another_method(*, failure_permissible_at_deployment: Optional[bool] = None,
                               violations_during_learning_acceptable: Optional[bool] = None,
                               prescribed_tolerance_least_conservatism: Optional[bool] = None,
                               expected_cost_requirement: Optional[bool] = None,
                               structured_solution_available: Optional[bool] = None,
                               price_far_above_stakes: Optional[bool] = None) -> List[Condition]:
    """The five conditions of Section 1.2 under which another approach is preferable. The bounded barrier is a
    reasonable first method when failures can be permitted in simulation and deployment has a positive tolerance or
    a finite failure cost; the results quantify these choices rather than assert an advantage over other methods."""
    no_failure = None
    if failure_permissible_at_deployment is not None or violations_during_learning_acceptable is not None:
        no_failure = (failure_permissible_at_deployment is False) or (violations_during_learning_acceptable is False)
    return [
        Condition("no deployment failure, or no failure during physical learning, is permissible", no_failure,
                  "an almost-sure formulation (Saute RL with an infinite penalty), a model-based control-barrier safety layer, or a method "
                  "that learns which actions can satisfy probability-one budget constraints",
                  "a finite price prices failure risk; it does not prohibit failure, and a full-support class can have a positive risk floor"),
        Condition("a prescribed tolerance is to be met with the least conservatism", prescribed_tolerance_least_conservatism,
                  "a Lagrangian chance-constrained policy gradient that targets the level directly (when it lies above the risk floor)",
                  "a fixed price leaves the attained risk to the optimizer and unused risk costs utility; the multiplier and learning errors still need control"),
        Condition("the requirement is an expected cumulative cost", expected_cost_requirement,
                  "constrained policy optimization or constrained Markov decision process methods",
                  "a chance requirement concerns a failure event instead"),
        Condition("the problem can be solved by a dynamic program or a structured approximation", structured_solution_available,
                  "use that structure; the toolbox's certificates and validation apply to such policies too",
                  "in the emissions study the exact dynamic program, and a tuned re-planning rule, were better than either learner"),
        Condition("the required price is far above the stakes of the task", price_far_above_stakes,
                  "reward normalization or a prescribed-level formulation",
                  "a large price degraded the controllers of the fuel-tank study; increasing the price is not an optimization guarantee"),
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
            lines.append(f"Failure tail charged at a failure: {discounted_total(s.m, sp.beta):.4g} (absorbing). A multiplier method settling at "
                         f"nu = {self.multiplier_equivalent:.4g} returns the same optimum.")
        if self.dual_step is not None:
            lines.append(f"Dual-ascent step for an adaptive multiplier at tolerance {r.alpha:g}: about {self.dual_step:.4g} per rollout "
                         f"(moves nu by a twentieth of its expected level when every episode fails).")
        lines.append(f"Validation: {self.validation_episodes} fresh episodes of the deployed candidate through action {r.T} are needed "
                     f"for the direct binomial test to be able to certify {r.alpha:g} at error {r.error_per_statement:.3g} "
                     f"(resolution floor of that many failure-free episodes: {self.direct_floor:.3g}).")
        if self.value_check is not None:
            vc = self.value_check
            lines.append(("The value-based test can" if vc["attainable"] else "The value-based test cannot") +
                         f" attain the tolerance with the stated N and H: its floor is {vc['floor']:.3g} "
                         f"(tail term {vc['tail_term']:.3g}, statistical term {vc['statistical_term']:.3g}). Increasing m cannot repair this.")
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
         allowance_error_known: Optional[bool] = None, violations_during_learning_acceptable: bool = True,
         failure_permissible_at_deployment: Optional[bool] = None, prescribed_tolerance_least_conservatism: Optional[bool] = None,
         expected_cost_requirement: Optional[bool] = None, structured_solution_available: Optional[bool] = None,
         task_stakes: Optional[float] = None, regularizers: Optional[dict] = None) -> Plan:
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
    violations_during_learning_acceptable, failure_permissible_at_deployment, prescribed_tolerance_least_conservatism,
    expected_cost_requirement, structured_solution_available, task_stakes
                  the conditions of Section 1.2; task_stakes is the utility at stake in one episode (for example a landing
                  bonus), against which the discounted total of the price is compared.
    regularizers  keyword arguments of bpg.regularizers.advise_regularizers (conditions of the application).
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
        warnings.append("no price given and no sup_U to derive one: the stage is the unpriced learner, which has no risk bound")
    # --- the conditions of Section 1.2 ------------------------------------------------------------
    far_above = None
    if task_stakes is not None and task_stakes > 0:
        charge = stage.multiplier if stage.penalty == "multiplier" else (failure_tail(stage.m, beta, 0, spec.horizon))
        far_above = charge > 10.0 * task_stakes
    conditions = when_to_use_another_method(
        failure_permissible_at_deployment=(False if requirement.alpha == 0.0 else failure_permissible_at_deployment),
        violations_during_learning_acceptable=violations_during_learning_acceptable,
        prescribed_tolerance_least_conservatism=prescribed_tolerance_least_conservatism,
        expected_cost_requirement=expected_cost_requirement, structured_solution_available=structured_solution_available,
        price_far_above_stakes=far_above)
    if not violations_during_learning_acceptable:
        warnings.append("violations during learning are unacceptable: this framework permits exploratory violations in simulation and is unsuitable "
                        "without an external certified mechanism")
    if requirement.kind == "eventual":
        warnings.append("an eventual requirement P(tau < inf) <= alpha is not certified by a small q_beta or by a finite-horizon test "
                        f"[{cite('delay_example')}]; it needs cost-tail structure ({cite('eventual_risk')}) or almost-sure feasibility")
    if requirement.alpha == 0.0:
        warnings.append("a zero tolerance is a hard almost-sure requirement; a positive price never prohibits failure, no multiplier settles "
                        "when every policy of the class has positive risk, and a zero-failure validation certifies only the resolution floor")
    if far_above:
        warnings.append("the discounted total of the price is more than ten times the utility at stake in an episode: in the fuel-tank study such a "
                        "price degraded the learned controllers; try reward normalization or a prescribed-level formulation")
    if stage.penalty == "price" and stage.m < u_minus and stage.clip_utility:
        warnings.append(f"m = {stage.m:g} is below u_minus = {u_minus:g}: clipping at -m erases utility values below -m (for example a crash penalty); "
                        "choose m >= u_minus or use the frozen-multiplier form (clip_utility=False)")
    if stage.theta > 0:
        notes.append("the interior barrier buys graded protection against a misjudged allowance at a utility cost bounded by its interior loss; "
                     f"{cite('fixed_price')} applies net of that loss")
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
    notes.append("a prescribed level has to be set below the tolerance by the width of the confidence bound (a multiplier settling at the tolerance "
                 "itself does not pass validation at that tolerance), just as a fixed price has to be chosen with one")
    notes.append(f"a sequence of prices reaches a tolerance only when the learner's error grows slower than the price; check it on a pilot sweep "
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
