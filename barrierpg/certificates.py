"""Risk certificates and performance comparisons (the paper's model, risk-certificate and deployment sections).

Every function returns a `Bound` (value, source, hypotheses, inputs, note) or a plain number where
the quantity is an identity rather than a bound. Inputs are the quantities of the paper:

  V_m    clipped value (from an exact solver, a Monte Carlo lower confidence bound, or the training
         objective); J the regularized objective -V_m + eta A + kappa H;
  U      stopped utility; I_m interior loss; q_beta geometric risk; p_T finite-horizon risk;
  R      regularization cost eta A + kappa H of a comparison law;
  delta  optimization error; eps_r reward approximation error; A action relative entropy.

Citations go through `barrierpg.paper.cite`, so they follow the paper's current numbering.
"""
from __future__ import annotations

import math
from typing import Optional

from .bounds import Bound, clip01
from .paper import cite

_H_UBAR = "ubar is a valid upper bound on the one-step utility"
_H_VM = "V_m is a valid value (or a valid lower bound on it) of the policy under the stated reward convention"
_H_SIGN = "-u_minus <= u <= ubar and m >= u_minus (clipping does not act on the utility itself)"
_H_OPT = "pi_hat maximizes V_m over the class up to delta"


# ---------------------------------------------------------------------------------------------
# The identity behind everything, and the two risk criteria
# ---------------------------------------------------------------------------------------------
def clipped_value_identity(U: float, I_m: float, q_beta: float, m: float, beta: float) -> float:
    """V_m = U - I_m - m q_beta / (1 - beta): splitting the clipped return at the failure time
    (exact under -u_minus <= u <= ubar and m >= u_minus)."""
    return float(U - I_m - m * q_beta / (1.0 - beta))


def geometric_from_finite(p_T: float, beta: float, T: int) -> Bound:
    """beta^T p_T <= q_beta: a finite-horizon risk gives a lower bound on the geometric risk."""
    return Bound("q_beta lower bound", beta ** T * p_T, "beta^T p_T <= q_beta, since beta^tau >= beta^T on tau <= T",
                 note="a small q_beta is not an eventual-risk guarantee")


def finite_from_geometric(q_beta: float, beta: float, T: int) -> Bound:
    """p_T <= min{1, q_beta / beta^T}: valid, and uninformative once beta^-T is large."""
    return Bound(f"p_{T} upper bound", clip01(q_beta / beta ** T), "p_T <= q_beta / beta^T, since beta^tau >= beta^T on tau <= T",
                 inputs=dict(q_beta=q_beta, beta=beta, T=T),
                 note="beta^-T can make this uninformative while valid; validate p_T directly when it is")


def earliest_failures_time(q_beta: float, beta: float, eps: float) -> Bound:
    """What a geometric risk says about when failures happen: with theta = log(1/beta), q_beta = E exp(-theta tau) is
    the exponential risk measure of the failure time, and for every eps in (0, 1] the earliest eps-fraction of failures
    occurs on average no earlier than log(eps / q_beta) / theta periods. A smaller q_beta pushes this bound up without
    limit, whereas a bound on p_T at one horizon never says more than T + 1. Pass a validated upper bound on q_beta to
    obtain a valid lower bound on the time."""
    if not 0.0 < eps <= 1.0:
        raise ValueError("eps must lie in (0, 1]")
    theta = math.log(1.0 / beta)
    t = max(0.0, math.log(eps / max(q_beta, 1e-300)) / theta)
    return Bound("average time of the earliest failures, lower bound", float(t), cite("failure_time"),
                 ["q_beta is the geometric risk of the policy, or a valid upper bound on it"],
                 dict(q_beta=q_beta, beta=beta, eps=eps, theta=theta),
                 note=f"the earliest {eps:.0%} of failures occur on average no earlier than this many periods")


def risk_floor(p_T_reference: float, L0: float, T: int) -> Bound:
    """p_T(pi_mu) >= e^{-2 L0 (T+1)} p_T(rho) for every softmax policy with |psi| <= L0: bounded scores
    cannot remove a dangerous action sequence that the action reference can draw. A target below
    this floor is unattainable within the class, whatever the price (the paper's represented class)."""
    return Bound(f"p_{T} lower bound (risk floor of the class)", float(math.exp(-2.0 * L0 * (T + 1)) * p_T_reference),
                 cite("risk_floor"), ["softmax class with |psi| <= L0 and action reference rho"],
                 dict(p_T_reference=p_T_reference, L0=L0, T=T),
                 note="no price, multiplier or regularizer lowers the risk below this floor within the class")


def prescribed_level_attainable(alpha: float, risk_floor_value: float) -> dict:
    """A prescribed level below the risk floor of the class is unattainable, and the dual of the
    prescribed-level problem is unbounded below: a multiplier iteration cannot converge (the paper's fixed-price section),
    whereas a fixed price stays well posed at every m."""
    ok = alpha > risk_floor_value
    return dict(attainable=ok, alpha=alpha, floor=risk_floor_value,
                explanation=("the tolerance lies above the risk floor of the class" if ok else
                             "the tolerance lies below the risk floor: no multiplier converges; a fixed price stays well posed "
                             "and its certificates bound the risk it attains [" + cite("fixed_price_section") + "]"))


# ---------------------------------------------------------------------------------------------
# Value and objective certificates
# ---------------------------------------------------------------------------------------------
def risk_from_value(V_m: float, m: float, ubar: float, beta: float, T: Optional[int] = None) -> Bound:
    """q_beta <= (ubar - (1-beta) V_m) / (m + ubar); with T, p_T <= min{1, that / beta^T}.
    Holds for every admissible policy; no optimization guarantee is needed."""
    q = clip01((ubar - (1.0 - beta) * V_m) / (m + ubar))
    if T is None:
        return Bound("q_beta upper bound", q, cite("value_certificate"), [_H_UBAR, _H_VM],
                     dict(V_m=V_m, m=m, ubar=ubar, beta=beta))
    return Bound(f"p_{T} upper bound", clip01(q / beta ** T), cite("value_certificate") + ", finite horizon",
                 [_H_UBAR, _H_VM], dict(V_m=V_m, m=m, ubar=ubar, beta=beta, T=T),
                 note="the factor beta^-T can make this uninformative while valid")


def risk_from_objective(J: float, m: float, ubar: float, beta: float, eps_r: float = 0.0, T: Optional[int] = None) -> Bound:
    """Same with the regularized objective J = -V_m + R in place of -V_m (R >= 0): numerator
    ubar + (1-beta) J + eps_r, where eps_r bounds the reward approximation."""
    q = clip01((ubar + (1.0 - beta) * J + eps_r) / (m + ubar))
    b = Bound("q_beta upper bound", q if T is None else clip01(q / beta ** T), cite("value_certificate") + ", objective form",
              [_H_UBAR, "J is the regularized objective of the policy, regularizers nonnegative", "||r~_m - r_m|| <= eps_r"],
              dict(J=J, m=m, ubar=ubar, beta=beta, eps_r=eps_r))
    if T is not None:
        b.name = f"p_{T} upper bound"
    return b


# ---------------------------------------------------------------------------------------------
# Matched-risk performance comparison: utility at the attained risk
# ---------------------------------------------------------------------------------------------
def matched_risk_utility_gap(I_bar: float, R_bar: float, delta: float, m: float, beta: float, alpha: float, q_hat: float,
                             eps_r: float = 0.0, continuing: bool = False, ubar: float = 0.0, u_minus: float = 0.0) -> Bound:
    """U(mu_bar) - U(mu_hat) <= I_m(mu_bar) + R(mu_bar) + delta + 2 eps_r/(1-beta) + m[alpha - q_hat]/(1-beta)
    for every comparison law with q(mu_bar) <= alpha. The last term is the unused risk alpha - q_hat
    charged at the price: excessive conservatism costs utility. continuing=True adds
    (ubar alpha + u_minus q_hat)/(1-beta) for the continuing utility V."""
    if alpha < q_hat:
        note = "alpha < q_hat: the unused-risk term is negative; the bound still holds for comparison laws with q <= alpha"
    else:
        note = f"unused risk charged at the price, m[alpha - q_hat]/(1-beta) = {m * (alpha - q_hat) / (1 - beta):.4g}"
    val = I_bar + R_bar + delta + 2.0 * eps_r / (1.0 - beta) + m * (alpha - q_hat) / (1.0 - beta)
    src = cite("matched_risk")
    if continuing:
        val += (ubar * alpha + u_minus * q_hat) / (1.0 - beta)
        src += ", continuing utility"
    return Bound("utility shortfall upper bound", float(val), src,
                 [_H_SIGN, "mu_hat is a delta-optimizer of the regularized objective; the comparison law has q(mu_bar) <= alpha"],
                 dict(I_bar=I_bar, R_bar=R_bar, delta=delta, m=m, beta=beta, alpha=alpha, q_hat=q_hat), note)


def matched_risk_signed(U_bar: float, I_bar: float, R_bar: float, delta: float, m: float, beta: float, q_hat: float, q_bar: float,
                        eps_r: float = 0.0) -> Bound:
    """The signed form of the matched-risk comparison: U(mu_hat) >= U(mu_bar) - I_m(mu_bar) - R(mu_bar) - delta - 2 eps_r/(1-beta)
    + m[q_hat - q_bar]/(1-beta). With the prior as comparison law (H(Gamma) = 0) the risk-price term remains:
    it is negative when the learned policy has less risk than the prior, so an improved clipped value
    does not by itself preserve the utility of a positive-risk prior."""
    lower = U_bar - I_bar - R_bar - delta - 2.0 * eps_r / (1.0 - beta) + m * (q_hat - q_bar) / (1.0 - beta)
    risk_term = m * (q_hat - q_bar) / (1.0 - beta)
    return Bound("stopped utility lower bound", float(lower), cite("matched_risk") + ", signed form",
                 [_H_SIGN, "mu_hat is a delta-optimizer of the regularized objective"],
                 dict(U_bar=U_bar, I_bar=I_bar, R_bar=R_bar, delta=delta, m=m, beta=beta, q_hat=q_hat, q_bar=q_bar, risk_price_term=risk_term),
                 note=f"risk-price term m[q_hat - q_bar]/(1-beta) = {risk_term:.4g}"
                      + ("; negative: lower risk than the comparison law was bought with utility" if risk_term < 0 else ""))


# ---------------------------------------------------------------------------------------------
# Fixed price of failure risk
# ---------------------------------------------------------------------------------------------
def price_risk_law(m: float, beta: float, sup_U: float, U0: float = 0.0, I0: float = 0.0, q0: float = 0.0,
                   delta: float = 0.0, ubar: Optional[float] = None) -> Bound:
    """q(pi_hat) <= q(pi_0) + (1-beta)[sup U - U(pi_0) + I_m(pi_0) + delta] / m for any comparison
    policy pi_0 of the class (the idle policy gives q0 = U0 = I0 = 0). With ubar, the second bound
    q(pi_0) + (1-beta)[ubar (1 - q(pi_0))/(1-beta) - U(pi_0) + I_m(pi_0) + delta] / (m + ubar), which needs
    nothing about the class, is also computed and the smaller is returned (both are in inputs)."""
    if m <= 0:
        return Bound("q_beta upper bound", 1.0, cite("fixed_price", "ii"), [_H_SIGN],
                     dict(m=m), note="an unpriced learner has no price-risk bound")
    law = clip01(q0 + (1.0 - beta) * (sup_U - U0 + I0 + delta) / m)
    inputs = dict(m=m, beta=beta, sup_U=sup_U, U0=U0, I0=I0, q0=q0, delta=delta, law=law)
    val, src = law, cite("fixed_price", "ii")
    if ubar is not None:
        variant = clip01(q0 + (1.0 - beta) * (ubar * (1.0 - q0) / (1.0 - beta) - U0 + I0 + delta) / (m + ubar))
        inputs["variant"] = variant
        if variant < law:
            val, src = variant, cite("fixed_price", "ii") + ", second bound with the (m+ubar) denominator"
    return Bound("q_beta upper bound", val, src,
                 [_H_SIGN, _H_OPT, "sup_U bounds the stopped utility over the class", "(U0, I0, q0) belong to a policy of the same class"],
                 inputs, note="conservative: charges the entire utility at stake to the risk (a factor 20-30 in the wear study)")


def conservative_price(alpha: float, beta: float, sup_U: float, U0: float = 0.0, I0: float = 0.0, q0: float = 0.0,
                       delta: float = 0.0, u_minus: float = 0.0) -> Bound:
    """The price at which the price-risk law certifies q_beta <= alpha with the comparison policy
    (U0, I0, q0): m >= (1-beta)[sup U - U0 + I0 + delta] / (alpha - q0), and m >= u_minus. Infinite if
    alpha <= q0. The paper's suggested starting price ('Using the clipped barrier'): a safe start, not the price to aim for."""
    if alpha <= q0:
        return Bound("price m certifying alpha", math.inf, cite("fixed_price", "ii") + " inverted", [_H_SIGN],
                     note="the comparison policy's own risk exceeds the tolerance; no price is certified by this law")
    m = max(u_minus, (1.0 - beta) * (sup_U - U0 + I0 + delta) / (alpha - q0))
    return Bound("price m certifying alpha", float(m), cite("fixed_price", "ii") + " inverted (conservative price)", [_H_SIGN],
                 dict(alpha=alpha, beta=beta, sup_U=sup_U, U0=U0, I0=I0, q0=q0, delta=delta),
                 note="start here, let certificates and validation bound the attained risk, retrain higher only if the tolerance is missed")


def conservative_multiplier(alpha: float, sup_U: float, U0: float = 0.0, q0: float = 0.0, delta: float = 0.0) -> Bound:
    """The flat failure penalty nu at which the frozen-multiplier objective U - nu q_beta certifies
    q_beta <= alpha: nu >= [sup U - U0 + delta] / (alpha - q0). This is the price-risk law with I_m = 0 and
    nu = m/(1-beta); use it for a lump sum charged once at failure."""
    if alpha <= q0:
        return Bound("flat penalty nu certifying alpha", math.inf, cite("fixed_price", "ii") + " with I_m = 0, inverted",
                     note="the comparison policy's own risk exceeds the tolerance")
    nu = (sup_U - U0 + delta) / (alpha - q0)
    return Bound("flat penalty nu certifying alpha", float(nu), cite("fixed_price", "ii") + " with I_m = 0, inverted",
                 ["the objective is U - nu q_beta (flat penalty at failure, pre-failure reward unclipped)", _H_OPT],
                 dict(alpha=alpha, sup_U=sup_U, U0=U0, q0=q0, delta=delta),
                 note="conservative; under a physical horizon q_beta is computed with the episode's own discount")


def price_monotonicity(q1: float, m1: float, m2: float, beta: float, Delta: float = 0.0, delta1: float = 0.0, delta2: float = 0.0) -> Bound:
    """q(pi_2) <= q(pi_1) + (1-beta)[Delta + delta_1 + delta_2] / (m_2 - m_1) for optimizers at
    prices u_minus <= m_1 < m_2, Delta = I_{m2}(pi_1) - I_{m1}(pi_1) >= 0 (zero for a failure-only price)."""
    if m2 <= m1:
        raise ValueError("m2 > m1 required")
    val = clip01(q1 + (1.0 - beta) * (Delta + delta1 + delta2) / (m2 - m1))
    return Bound("q_beta at the higher price, upper bound", val, cite("fixed_price", "iii"), [_H_SIGN],
                 dict(q1=q1, m1=m1, m2=m2, Delta=Delta, delta1=delta1, delta2=delta2),
                 note="for a failure-only price and exact optimizers the attained risk is nonincreasing in the price")


def frozen_multiplier_value(U: float, q_beta: float, nu: float) -> float:
    """U - nu q_beta: the frozen-multiplier objective, to which the fixed-price proposition applies verbatim with I_m = 0."""
    return float(U - nu * q_beta)


def learning_form_risk(m: float, beta: float, ubar: float, q_bar: float, U_bar: float, I_bar: float, R_bar: float, delta: float,
                       eps_r: float = 0.0, sup_U: Optional[float] = None, T: Optional[int] = None) -> Bound:
    """The learning form of the fixed-price proposition: both bounds of (ii) for a delta-minimizer of the regularized
    objective, with a comparison law's own quantities and delta' = delta + R(mu_bar) + 2 eps_r/(1-beta):
      q_hat <= q_bar + (1-beta)[ubar (1 - q_bar)/(1-beta) - U_bar + I_bar + delta'] / (m + ubar),
      q_hat <= q_bar + (1-beta)[sup U - U_bar + I_bar + delta'] / m          (when sup_U is given).
    I_bar is the comparison law's clipped interior loss (finite even at a zero reserve)."""
    dprime = delta + R_bar + 2.0 * eps_r / (1.0 - beta)
    b = price_risk_law(m, beta, sup_U if sup_U is not None else math.inf, U_bar, I_bar, q_bar, dprime, ubar)
    b.source = cite("fixed_price", "iv")
    b.hypotheses = ["mu_hat is a delta-minimizer of the regularized objective (or beats the comparison law by delta)",
                    "(q_bar, U_bar, I_bar) are the comparison law's geometric risk, stopped utility and clipped interior loss; R_bar its regularization cost",
                    _H_SIGN, "||r~_m - r_m|| <= eps_r"]
    b.inputs.update(delta_prime=dprime, R_bar=R_bar, eps_r=eps_r)
    b.note = "the regularization bias enters only through R(mu_bar); the bound is only as good as its comparison law"
    if T is not None:
        b = Bound(f"p_{T} upper bound", clip01(b.value / beta ** T), b.source + ", finite horizon", b.hypotheses, b.inputs, b.note)
    return b


def comparison_law_risk(q_bar: float, C_B: float, R_bar: float, delta: float, m: float, ubar: float, beta: float,
                        eps_r: float = 0.0, T: Optional[int] = None) -> Bound:
    """q(mu_hat) <= q(mu_bar) + ((1-beta)[C_B + R(mu_bar) + delta] + 2 eps_r) / (m + ubar), with the unclipped
    pre-failure cost C_B = E sum_{t<tau} beta^t [ubar - u + B]. It follows from the second bound of
    the learning form, since ubar (1 - q_bar)/(1-beta) - U_bar + I_bar <= C_B; learning_form_risk is never looser."""
    q = clip01(q_bar + ((1.0 - beta) * (C_B + R_bar + delta) + 2.0 * eps_r) / (m + ubar))
    hyp = [_H_UBAR, "mu_hat is a delta-minimizer of the regularized objective (or beats the comparison law by delta)",
           "C_B(mu_bar) < inf: the comparison law has finite pre-failure barrier cost",
           "R(mu_bar) = eta A(mu_bar) + kappa H(mu_bar) is its regularization cost"]
    src = cite("fixed_price", "iv") + ", with the unclipped barrier cost"
    if T is None:
        return Bound("q_beta upper bound", q, src, hyp,
                     dict(q_bar=q_bar, C_B=C_B, R_bar=R_bar, delta=delta, m=m, ubar=ubar, beta=beta, eps_r=eps_r),
                     note="only as good as its comparison law; can be minimized over comparison laws under the infimum hypothesis")
    return Bound(f"p_{T} upper bound", clip01(q / beta ** T), src + ", finite horizon", hyp,
                 dict(q_bar=q_bar, C_B=C_B, R_bar=R_bar, delta=delta, m=m, ubar=ubar, beta=beta, eps_r=eps_r, T=T))


def learning_slack(m: float, ubar: float, beta: float, alpha: float, T: int, q_bar: float, U_bar: float, I_bar: float,
                   R_bar: float, sup_U: Optional[float] = None) -> float:
    """The optimization error that the learning-error bound can absorb while still certifying p_T <= alpha with this
    comparison law: the larger of
      D  = m (alpha beta^T - q_bar)/(1-beta) - [sup U - U_bar + I_bar + R_bar]                    (sup_U given),
      D' = (m+ubar)(alpha beta^T - q_bar)/(1-beta) - [ubar (1-q_bar)/(1-beta) - U_bar + I_bar + R_bar].
    D > 0: the bound certifies the target once the learning error is below D (convergence.learning_time_for_slack).
    D <= 0: longer training alone cannot make this bound certify the target; this does not prove infeasibility."""
    ag = alpha * beta ** T
    d2 = (m + ubar) * (ag - q_bar) / (1.0 - beta) - (ubar * (1.0 - q_bar) / (1.0 - beta) - U_bar + I_bar + R_bar)
    if sup_U is None:
        return float(d2)
    d1 = m * (ag - q_bar) / (1.0 - beta) - (sup_U - U_bar + I_bar + R_bar)
    return float(max(d1, d2))


def comparison_law_slack(m: float, ubar: float, beta: float, alpha: float, T: int, q_bar: float, C_B: float, R_bar: float) -> float:
    """D_m = (m + ubar)[alpha beta^T - q(mu_bar)] / (1 - beta) - C_B - R(mu_bar): the slack of comparison_law_risk,
    with a pre-failure cost C_B (clipped or unclipped). Never larger than learning_slack with the same law."""
    return float((m + ubar) * (alpha * beta ** T - q_bar) / (1.0 - beta) - C_B - R_bar)


# ---------------------------------------------------------------------------------------------
# Action regularization and failure risk
# ---------------------------------------------------------------------------------------------
def kl_bernoulli(p: float, q: float) -> float:
    """kl(p || q) with the conventions 0 log(0/q) = 0 and p log(p/0) = +inf for p > 0."""
    p, q = float(p), float(q)
    out = 0.0
    if p > 0:
        out += math.inf if q <= 0 else p * math.log(p / q)
    if p < 1:
        out += math.inf if q >= 1 else (1 - p) * math.log((1 - p) / (1 - q))
    return out


def kl_upper_inverse(q_ref: float, A: float) -> float:
    """Largest p in [q_ref, 1] with kl(p || q_ref) <= A (the inverted bound of the action-risk proposition)."""
    if A == math.inf or q_ref >= 1.0:
        return 1.0
    lo, hi = float(q_ref), 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mid, q_ref) <= A:
            lo = mid
        else:
            hi = mid
    return lo


def kl_lower_inverse(q_ref: float, A: float) -> float:
    """Smallest p in [0, q_ref] with kl(p || q_ref) <= A."""
    if A == math.inf or q_ref <= 0.0:
        return 0.0
    lo, hi = 0.0, float(q_ref)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mid, q_ref) <= A:
            hi = mid
        else:
            lo = mid
    return hi


def action_risk_bound(A: float, q_ref: float, beta: float, T: Optional[int] = None) -> Bound:
    """kl(q(mu) || q(rho)) <= A, and kl(p_T(mu) || p_T(rho)) <= beta^-T A: the departure of the learned
    policy's risk from the action reference's. Returns the inverted upper bound; the lower end and the
    Pinsker half-width sqrt(A/2) (or sqrt(A/(2 beta^T))) are in inputs. It controls departure, not
    improvement, and applies to any policy kernel with bounded rewards and finite statewise KL to rho."""
    AA = A if T is None else A / beta ** T
    up = kl_upper_inverse(q_ref, AA)
    name = "q_beta upper bound" if T is None else f"p_{T} upper bound"
    return Bound(name, clip01(up), cite("action_risk"),
                 ["A is the stopped action relative entropy of the learned policy to the action reference rho (failing action included)",
                  "q_ref is the risk of drawing actions from rho (needs a model justification or independent validation)"],
                 dict(A=A, q_ref=q_ref, beta=beta, pinsker_half_width=math.sqrt(AA / 2.0), lower=kl_lower_inverse(q_ref, AA)),
                 note="informative only for small A; a high-risk action reference argues against a large eta")


# ---------------------------------------------------------------------------------------------
# Supplementary results of the toolbox (proofs in docs/under-the-hood/supplementary-results.md)
# ---------------------------------------------------------------------------------------------
def near_exhaustion_bound(V_m: float, m: float, ubar: float, beta: float, T: int, lower_barrier: float) -> Bound:
    """P(exists t <= T, k: z_{t+1,k} <= eps_k) <= min{1, (ubar - (1-beta) V_m) / ((1-beta) beta^T min{m+ubar, B_(eps)})},
    where lower_barrier = inf{B(z): z_k <= eps_k for some k} > 0 (LogBarrier.lower_level). An interior
    barrier charges near-exhaustion, which a failure-only price never does."""
    if lower_barrier <= 0:
        return Bound("near-exhaustion probability upper bound", 1.0, cite("near_exhaustion"), [_H_UBAR],
                     note="needs a positive lower barrier level at the threshold; a failure-only price never charges near-exhaustion")
    den = (1.0 - beta) * beta ** T * min(m + ubar, lower_barrier)
    return Bound("near-exhaustion probability upper bound", clip01((ubar - (1.0 - beta) * V_m) / den),
                 cite("near_exhaustion"), [_H_UBAR, _H_VM, "B_(eps) > 0 at the threshold eps"],
                 dict(V_m=V_m, m=m, ubar=ubar, beta=beta, T=T, lower_barrier=lower_barrier))


def unresolved_threshold(gbar: float, gamma: float) -> float:
    """c_k = gbar_k / (1 - gamma_k): a reserve at or above c_k can never fail again (needs gamma_k < 1)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("the structural certificate needs 0 < gamma_k < 1")
    return float(gbar) / (1.0 - float(gamma))


def eventual_risk_structural(p_T: float, P_unresolved: float) -> Bound:
    """P(tau < inf) <= p_T + P(E_T), E_T = {tau > T, some z_{T+1,k} < gbar_k/(1-gamma_k)}; both terms are
    observable from T+1 actions and can be validated directly. Without such cost-tail structure, a small
    q_beta or a certificate at one T does not bound the eventual risk."""
    return Bound("eventual risk upper bound", clip01(p_T + P_unresolved), cite("eventual_risk"),
                 ["0 <= g_k <= gbar_k on all admissible pairs and 0 < gamma_k < 1"],
                 dict(p_T=p_T, P_unresolved=P_unresolved),
                 note="vanishing unresolved mass makes the bound approach the true eventual risk; it does not make that risk small")


# ---------------------------------------------------------------------------------------------
# Population-to-network transfer: from a learned law to one finite network
# ---------------------------------------------------------------------------------------------
def finite_network_transfer(n: int, L0: float, beta: float, zeta: float, m: float, ubar: float, u_minus: float = 0.0,
                            T: Optional[int] = None) -> dict:
    """eps_n = min{1, L0 / (2 sqrt(n (1-beta) zeta))}; with probability >= 1 - zeta over the sampled units,
    |q(pi_n) - q(mu)| <= eps_n, |V_m(pi_n) - V_m(mu)| <= (m+ubar) eps_n/(1-beta),
    |U or V difference| <= (u_minus+ubar) eps_n/(1-beta); for fixed T, on a separate event,
    |p_T(pi_n) - p_T(mu)| <= min{1, L0 sqrt(T+1) / (2 sqrt(n zeta))}. Use zeta/2 in both for simultaneity."""
    eps = min(1.0, L0 / (2.0 * math.sqrt(n * (1.0 - beta) * zeta)))
    out = dict(eps_n=eps, risk=eps, value=(m + ubar) * eps / (1.0 - beta), utility=(u_minus + ubar) * eps / (1.0 - beta),
               source=cite("transfer"))
    if T is not None:
        out["p_T"] = min(1.0, L0 * math.sqrt(T + 1) / (2.0 * math.sqrt(n * zeta)))
    return out


def width_for_target(L0: float, beta: float, zeta: float, Delta: float) -> int:
    """n >= L0^2 / (4 (1-beta) zeta Delta^2) makes q_beta(pi_n) within Delta of the population law's (the population-to-network transfer);
    with a population bound of slack Delta = alpha beta^T - R_l > 0 this certifies p_T(pi_n) <= alpha.
    With L0 = 1, beta = 0.95, zeta = 0.05 a transfer error of one percent needs about a million units."""
    if Delta <= 0:
        return 0
    return int(math.ceil(L0 ** 2 / (4.0 * (1.0 - beta) * zeta * Delta ** 2)))


def width_for_direct_target(L0: float, T: int, zeta: float, Delta_T: float) -> int:
    """n >= L0^2 (T+1) / (4 zeta Delta_T^2) from a direct population bound p_T(mu) <= alpha - Delta_T."""
    if Delta_T <= 0:
        return 0
    return int(math.ceil(L0 ** 2 * (T + 1) / (4.0 * zeta * Delta_T ** 2)))


__all__ = ["clipped_value_identity", "geometric_from_finite", "finite_from_geometric", "earliest_failures_time", "risk_floor", "prescribed_level_attainable",
           "risk_from_value", "risk_from_objective", "matched_risk_utility_gap", "matched_risk_signed",
           "price_risk_law", "conservative_price", "conservative_multiplier", "price_monotonicity", "frozen_multiplier_value",
           "learning_form_risk", "comparison_law_risk", "learning_slack", "comparison_law_slack",
           "kl_bernoulli", "kl_upper_inverse", "kl_lower_inverse", "action_risk_bound", "near_exhaustion_bound",
           "unresolved_threshold", "eventual_risk_structural", "finite_network_transfer", "width_for_target",
           "width_for_direct_target"]
