"""Convergence: the curvature budget of the population flow, and diagnostics for finite learners.

Part 1 (population flow; the paper's 'Convergence under general convex regularization'). The clipped value and the action relative entropy
fall short of convex along transport by at most C_m + eta C_A, their curvature deficit; an admissible
parameter regularizer supplies the curvature kappa lambda_H. The curvature budget is

    Lambda_m = kappa lambda_H - C_m - eta C_A,

and when it is positive the population flow converges exponentially ('Fixed-stage population convergence'): W2 at rate
Lambda_m, the objective gap at rate 2 Lambda_m. The deficits are computed from bounds on how fast the
policy kernel and its relative entropy to the action reference change when the units move (Proposition
B.1): B1, B2 for the kernel, K0, K1, K2 for the relative entropy. The softmax class has B1 = L1,
B2 = 2(L2 + L1^2), K0 = 2 L0, K1 = L0 L1, K2 = L0 L2 + (1 + 2 L0) L1^2, which gives

    C_m = (m + ubar) c_psi,  c_psi = L2/(1-beta)^2 + L1^2/(1-beta)^3,
    C_A = L0 (1+3 beta) L2/(1-beta)^2 + [(1+2 L0)/(1-beta) + 2 beta L0 (3-beta)/(1-beta)^3] L1^2.

The threshold is sufficient, not necessary: the solvable flow converges orders of magnitude below it,
and at the threshold the minimizer lies close to the prior. Set the strength for conditioning; the
theorem explains the mechanism (a curvature budget), it does not certify the strengths used in practice.

Part 2 (finite learners). What a practitioner can measure: the decay of an objective gap, the absence
of a trend in the last part of a trace, whether an adaptive multiplier converges or increases without bound, and the
dispersion across seeds. These are evidence about one optimizer run, not instances of the theorem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .paper import cite
from .spec import price_from_multiplier


# ---------------------------------------------------------------------------------------------
# Part 1: constants of the actor class
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelConstants:
    """Bounds of 'Value and action-law curvature' along every coupling interpolation of the units, D the coupling distance:
    ||d/dt pi|| <= B1 D, ||d2/dt2 pi|| <= B2 D^2 (total variation, uniform over states), and for the
    statewise relative entropy k to the action reference: 0 <= k <= K0, |dk| <= K1 D, |d2k| <= K2 D^2.
    Any policy class with these bounds is covered by the convergence theorem after its own verification."""
    B1: float
    B2: float
    K0: float = 0.0
    K1: float = 0.0
    K2: float = 0.0

    @staticmethod
    def from_softmax(fc: "FeatureConstants") -> "KernelConstants":
        """The paper's softmax class with feature bounds L0, L1, L2 (the feature assumption)."""
        return KernelConstants(B1=fc.L1, B2=2.0 * (fc.L2 + fc.L1 ** 2), K0=2.0 * fc.L0, K1=fc.L0 * fc.L1,
                               K2=fc.L0 * fc.L2 + (1.0 + 2.0 * fc.L0) * fc.L1 ** 2)


@dataclass(frozen=True)
class FeatureConstants:
    """The feature assumption for the softmax class: |psi| <= L0, |grad_x psi| <= L1, ||hess_x psi|| <= L2."""
    L0: float
    L1: float
    L2: float

    @staticmethod
    def tanh_unit(scale: float, input_bound: float) -> "FeatureConstants":
        """psi(y, a, x) = scale * tanh(x . phi(y, a)) with |phi(y, a)| <= input_bound:
        L0 = scale, L1 = scale r, L2 = scale r^2 sup|tanh''| = scale r^2 4/(3 sqrt 3)."""
        r = float(input_bound)
        return FeatureConstants(float(scale), float(scale) * r, float(scale) * r * r * 4.0 / (3.0 * math.sqrt(3.0)))

    @staticmethod
    def from_grid(values, grads, hessians) -> "FeatureConstants":
        """Suprema over a sample of (y, a, x): values |psi|, gradient norms, Hessian operator norms.
        A sample gives a lower estimate of a supremum; use a fine grid or an analytic bound."""
        return FeatureConstants(float(np.max(np.abs(values))), float(np.max(grads)), float(np.max(hessians)))

    def kernel(self) -> KernelConstants:
        return KernelConstants.from_softmax(self)


Constants = Union[FeatureConstants, KernelConstants]


def curvature_constants(constants: Constants, beta: float, m: float, ubar: float) -> Dict[str, float]:
    """The deficits of 'Value and action-law curvature', with d = 1 - beta:
        C_m = (m + ubar)/2 [B2/d^2 + 2 beta B1^2/d^3],
        C_A = K2/d + (beta B2 K0 + 2 beta B1 K1)/d^2 + 2 beta^2 B1^2 K0/d^3,
    and c_psi = C_m/(m + ubar). For softmax these are the closed forms in the module docstring."""
    k = constants.kernel() if isinstance(constants, FeatureConstants) else constants
    d = 1.0 - beta
    c_psi = 0.5 * (k.B2 / d ** 2 + 2.0 * beta * k.B1 ** 2 / d ** 3)
    C_m = (m + ubar) * c_psi
    C_A = k.K2 / d + (beta * k.B2 * k.K0 + 2.0 * beta * k.B1 * k.K1) / d ** 2 + 2.0 * beta ** 2 * k.B1 ** 2 * k.K0 / d ** 3
    return dict(c_psi=c_psi, C_m=C_m, C_A=C_A, B1=k.B1)


# ---------------------------------------------------------------------------------------------
# Part 1: curvature supplied by the parameter regularizer
# ---------------------------------------------------------------------------------------------
def gaussian_prior_curvature(sigma: float) -> float:
    """lambda_W = sigma^-2 for the prior N(x0, sigma^2 I). Relative entropy to it is negative entropy of
    weight kappa plus weight decay toward x0 of weight kappa/sigma^2, so the rate depends on (kappa, sigma)
    only through kappa/sigma^2 while the minimizer depends on both ('The Gaussian prior as entropy, weight decay, and a policy')."""
    return 1.0 / float(sigma) ** 2


def supplied_curvature(kappa1: float = 0.0, lambda_W: float = 1.0, kappa2: float = 0.0, lambda_P: float = 1.0) -> float:
    """kappa1 lambda_W + kappa2 lambda_P: a divergence of the admissible families with strength kappa1 plus a
    lambda_P-convex potential with strength kappa2 is admissible with this curvature. The integrand of the
    divergence adds no curvature, so the rate and threshold are the same for every integrand."""
    return kappa1 * lambda_W + kappa2 * lambda_P


def tv_constant(kind: str = "kl", omega: float = 0.0) -> float:
    """c_H with ||mu - Gamma||_TV^2 <= c_H H(mu): 2 for any H >= KL (Pinsker), 2/(1+2 omega) for KL + omega chi^2."""
    if kind == "kl+chi2":
        return 2.0 / (1.0 + 2.0 * omega)
    return 2.0


def rate(kappa: float, lambda_H: float, C_m: float, C_A: float = 0.0, eta: float = 0.0) -> float:
    """Lambda_m = kappa lambda_H - C_m - eta C_A, the curvature budget of the convergence theorem."""
    return kappa * lambda_H - C_m - eta * C_A


def certified_kappa(lambda_H: float, C_m: float, C_A: float = 0.0, eta: float = 0.0, margin: float = 1.05) -> float:
    """The certified threshold (C_m + eta C_A)/lambda_H, times a margin (1.05 in the paper's solvable-flow test)."""
    return margin * (C_m + eta * C_A) / lambda_H


def time_to_fraction(Lambda: float, fraction: float) -> float:
    """Time at which the objective gap has shrunk by the factor `fraction`: log(1/fraction) / (2 Lambda)."""
    if Lambda <= 0:
        return math.inf
    return math.log(1.0 / fraction) / (2.0 * Lambda)


@dataclass
class CurvatureBudget:
    """Everything the convergence theorem says for a given stage and actor class, in one place."""
    constants: Dict[str, float]
    supply: float
    kappa: float
    eta: float
    Lambda: float
    kappa_certified: float
    lambda_H: float

    @property
    def certified(self) -> bool:
        return self.Lambda > 0

    def explain(self) -> str:
        c = self.constants
        lines = [f"curvature deficit: C_m = {c['C_m']:.4g} (clipped value), eta C_A = {self.eta * c['C_A']:.4g} (action relative entropy)",
                 f"curvature supplied by the parameter regularizer: {self.supply:.4g}",
                 f"budget Lambda_m = {self.Lambda:.4g}: " + ("certified; the objective gap decays like exp(-2 Lambda l), W2 like exp(-Lambda l)"
                                                           if self.certified else "not certified by the convergence theorem (a sufficient condition; the flow may still converge)"),
                 f"certified threshold (margin 1.05): kappa >= {self.kappa_certified:.4g} at curvature {self.lambda_H:.4g} per unit strength; "
                 "at that strength the minimizer lies close to the prior"]
        if self.certified:
            lines.append(f"time to shrink the objective gap by 1e-3: {time_to_fraction(self.Lambda, 1e-3):.4g}")
        lines.append("the deficit grows linearly with m + ubar and like (1-beta)^-3; set the strength for conditioning, not at the threshold "
                     f"[{cite('flow')}; {cite('bias')}]")
        return "\n".join(lines)


def curvature_budget(constants: Constants, beta: float, m: float, ubar: float, kappa: float, lambda_H: float, eta: float = 0.0,
                     kappa2: float = 0.0, lambda_P: float = 1.0) -> CurvatureBudget:
    """The curvature budget for a stage (m, eta, kappa) and an actor class. kappa lambda_H is the divergence part
    (lambda_H = 1/sigma^2 for a Gaussian prior); kappa2 lambda_P adds a potential such as weight decay."""
    c = curvature_constants(constants, beta, m, ubar)
    supply = kappa * lambda_H + kappa2 * lambda_P
    Lambda = supply - c["C_m"] - eta * c["C_A"]
    return CurvatureBudget(c, supply, kappa, eta, Lambda, certified_kappa(lambda_H, c["C_m"], c["C_A"], eta), lambda_H)


# ---------------------------------------------------------------------------------------------
# Part 1: learning error and budget risk; finite-network training under a potential
# ---------------------------------------------------------------------------------------------
def learning_risk_bound(ell: float, *, m: float, beta: float, ubar: float, q_bar: float, U_bar: float, I_bar: float,
                        R_bar: float, E0: float, Lambda: float, sup_U: Optional[float] = None, T: Optional[int] = None):
    """'Learning error and budget risk': for the flow with Lambda_m > 0 at learning time ell, both bounds of the learning form hold
    with the learning error E0 exp(-2 Lambda ell) in place of delta and the comparison law's regularization cost R."""
    from .certificates import learning_form_risk
    if Lambda <= 0:
        raise ValueError("the learning-error bound needs Lambda_m > 0")
    b = learning_form_risk(m, beta, ubar, q_bar, U_bar, I_bar, R_bar, E0 * math.exp(-2.0 * Lambda * ell), sup_U=sup_U, T=T)
    b.source = cite("learning_error")
    b.inputs.update(ell=ell, E0=E0, Lambda=Lambda)
    return b


def learning_time_for_slack(E0: float, D: float, Lambda: float) -> float:
    """Sufficient learning time ell >= [log(E0 / D)]_+ / (2 Lambda) for the learning-error bound to certify a target whose
    slack is D > 0 (certificates.learning_slack); inf when D <= 0 or Lambda <= 0."""
    if D <= 0 or Lambda <= 0:
        return math.inf
    if E0 <= 0:
        return 0.0
    return max(0.0, math.log(E0 / D)) / (2.0 * Lambda)


def initial_gap_upper(J0: float, ubar: float, beta: float) -> float:
    """E0 <= J(mu_0) + ubar / (1 - beta): a valid bound on the initial objective gap."""
    return J0 + ubar / (1.0 - beta)


def finite_training_transfer(W2_initial: float, Lambda: float, ell: float, *, m: float, ubar: float, beta: float,
                             B1: float) -> dict:
    """'Finite-network training under a potential': under a convex potential with Lipschitz gradient (weight decay, an anchor), noiseless
    training of n units in continuous time IS the population flow from their empirical law, so
        W2(mu_n(ell), mu_ell) <= exp(-Lambda ell) W2(mu_n(0), mu_0)
    for every sign of Lambda, and the clipped values and geometric risks of the two laws differ by at most
    (m+ubar) B1/(2(1-beta)^2) and B1/(2(1-beta)) times that distance (B1 = L1 for softmax).
    Discrete steps, minibatch noise and PPO's or SAC's estimators are outside the proposition."""
    W2 = math.exp(-Lambda * ell) * W2_initial
    return dict(W2=W2, value=(m + ubar) * B1 / (2.0 * (1.0 - beta) ** 2) * W2, risk=B1 / (2.0 * (1.0 - beta)) * W2,
                value_lipschitz=(m + ubar) * B1 / (2.0 * (1.0 - beta) ** 2), risk_lipschitz=B1 / (2.0 * (1.0 - beta)),
                source=cite("finite_training"),
                note="continuous-time noiseless training under a potential; in the certified regime the units collapse to one parameter "
                     f"[{cite('collapse')}]")


# ---------------------------------------------------------------------------------------------
# Part 1: improvement over the prior policy
# ---------------------------------------------------------------------------------------------
def risk_slack_from_prior(L0: float, T: int, c_H: float, R_Gamma: float, kappa: float) -> float:
    """p_T(mu_ell) <= p_T(Gamma) + (L0/2) sqrt((T+1) c_H R_Gamma / kappa) for the flow started at the prior, with
    R_Gamma = ubar/(1-beta) - V_m(Gamma) ('Improvement over the prior policy'; no sign condition on Lambda_m). The behavior lies far inside
    this slack (two to five orders of magnitude in the solvable flow): the learner inherits the prior policy's risk."""
    return 0.5 * L0 * math.sqrt((T + 1) * c_H * R_Gamma / kappa)


def kappa_for_tolerance(L0: float, T: int, c_H: float, R_Gamma: float, alpha: float, alpha0: float) -> float:
    """kappa >= L0^2 (T+1) c_H R_Gamma / (4 (alpha - alpha0)^2) keeps every policy of the flow started at the prior
    within p_T <= alpha when p_T(Gamma) = alpha0 < alpha (the proof of 'Improvement over the prior policy')."""
    if alpha <= alpha0:
        return math.inf
    return L0 ** 2 * (T + 1) * c_H * R_Gamma / (4.0 * (alpha - alpha0) ** 2)


def action_slack_from_prior(R_Gamma: float, eta: float, beta: float, T: int) -> float:
    """|p_T(mu_ell) - p_T(Gamma)| <= sqrt(R_Gamma / (2 eta beta^T)) when eta > 0 and the action reference is the prior
    policy (the same proof); the smaller of the two slacks applies."""
    if eta <= 0:
        return math.inf
    return math.sqrt(R_Gamma / (2.0 * eta * beta ** T))


# ---------------------------------------------------------------------------------------------
# Part 2: diagnostics on traces of a finite learner
# ---------------------------------------------------------------------------------------------
def fit_exponential_rate(times: Sequence[float], values: Sequence[float], lo: float = 1e-8, hi: float = 1e-2) -> float:
    """Least-squares slope of log(values) over the window where values/values[0] lies in [lo, hi]
    (the paper's 'observed rate'); nan with fewer than four points."""
    t = np.asarray(times, dtype=float); y = np.asarray(values, dtype=float)
    if y.size == 0 or y[0] <= 0:
        return float("nan")
    rel = y / y[0]
    sel = (rel <= hi) & (rel >= lo) & (y > 0)
    if sel.sum() < 4:
        return float("nan")
    a = np.polyfit(t[sel], np.log(y[sel]), 1)
    return float(-a[0])


@dataclass
class TrendTest:
    slope_per_step: float
    slope_se: float
    window: int
    mean: float
    noise_sd: float
    trend_detected: bool
    relative_drift: float      # |slope| * window / max(|mean|, noise)

    def explain(self, name: str = "quantity") -> str:
        if self.trend_detected:
            d = "rising" if self.slope_per_step > 0 else "falling"
            return f"{name}: still {d} over the last {self.window} points (drift {self.relative_drift:.2%} of its level per window)"
        return f"{name}: no trend detectable at two standard errors over the last {self.window} points (mean {self.mean:.4g}, noise sd {self.noise_sd:.3g})"


def trend_test(values: Sequence[float], window: Optional[int] = None) -> TrendTest:
    """Ordinary least-squares slope over the last `window` points with its standard error; a trend is
    'detected' when |slope| > 2 SE. Evidence that a quantity has stopped changing, not proof of convergence."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    w = n if window is None else min(int(window), n)
    y = y[-w:]
    if w < 4:
        return TrendTest(float("nan"), float("nan"), w, float(np.mean(y)) if w else float("nan"), float("nan"), False, float("nan"))
    x = np.arange(w, dtype=float)
    X = np.column_stack([np.ones(w), x])
    coef, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = float(resid @ resid) / max(w - 2, 1)
    se = math.sqrt(s2 / float(((x - x.mean()) ** 2).sum()))
    slope = float(coef[1])
    mean = float(y.mean()); noise = math.sqrt(s2)
    detected = abs(slope) > 2.0 * se
    rel = abs(slope) * w / max(abs(mean), noise, 1e-300)
    return TrendTest(slope, se, w, mean, noise, bool(detected), rel)


@dataclass
class MultiplierDiagnosis:
    status: str                # 'stable', 'increasing', 'inactive', 'trending', 'off-target', 'short'
    final_multiplier: float
    equivalent_price: float    # (1 - beta) nu in the absorbing form
    window_failure_rate: float
    target: float
    explanation: str


def multiplier_diagnosis(trace: Sequence[dict], alpha: float, beta: float, window: int = 20, tol: float = 0.25) -> MultiplierDiagnosis:
    """Reads a dual-ascent trace (dicts with 'multiplier' and 'failure_rate') by two criteria and their complements.

    The multiplier is the running sum of its updates, so whether it still moves is tested on the updates: their mean
    over the last rollouts is its drift, detected when it exceeds two standard errors of that mean. A slope fitted to
    the levels of a running sum would report a trend in almost any window, also after convergence.

    stable      the multiplier shows no drift over the last rollouts and the failure rate is near the tolerance
                (within tol, relative, plus the sampling error of the window);
    increasing  the multiplier rose in nearly every rollout while the failure rate stayed far above the tolerance.
                When every episode fails, the flat penalty carries no gradient (the fuel-tank study with its
                step fixed before its campaign); reduce the dual step, use a fixed price, or start from a controller that sometimes
                survives;
    inactive    the multiplier is zero and the failure rate is below the tolerance: the tolerance does not constrain
                the unpriced optimum;
    trending    the multiplier is still rising or falling: train longer, or size the step from the level the
                multiplier has to reach. A multiplier that keeps rising while the failure rate stays above the
                tolerance is the sign of a tolerance below the risk floor of the class (then no multiplier converges);
    off-target  no detectable drift, but the failure rate is not near the tolerance: the rollouts are too few or too
                small to resolve the tolerance;
    short       too few rollouts.
    """
    nu = np.array([t["multiplier"] for t in trace], dtype=float)
    fr = np.array([t.get("failure_rate", np.nan) for t in trace], dtype=float)
    if len(nu) < max(6, window // 2):
        return MultiplierDiagnosis("short", float(nu[-1]) if len(nu) else 0.0, 0.0, float("nan"), alpha, "too few rollouts to diagnose")
    w = min(window, len(nu))
    rate_w = float(np.nanmean(fr[-w:]))
    steps = np.diff(nu[-(w + 1):])
    drift = float(steps.mean())
    drift_se = float(steps.std(ddof=1) / math.sqrt(len(steps))) if len(steps) > 1 else float("inf")
    drifting = abs(drift) > 2.0 * drift_se if drift_se > 0 else abs(drift) > 0
    last, eq = float(nu[-1]), price_from_multiplier(float(nu[-1]), beta)
    increases = np.mean(np.diff(nu[-w:]) > 0) if w > 1 else 0.0
    if last == 0.0 and rate_w <= alpha:
        return MultiplierDiagnosis("inactive", last, eq, rate_w, alpha,
                                   f"multiplier at zero with failure rate {rate_w:.3g} <= tolerance {alpha:g}: the tolerance does not constrain the unpriced optimum")
    if increases >= 0.9 and rate_w >= max(0.5, 5.0 * alpha):
        return MultiplierDiagnosis("increasing", last, eq, rate_w, alpha,
                                   f"multiplier rose in {increases:.0%} of the last {w} rollouts while the failure rate stayed at {rate_w:.2f}, far above "
                                   f"the tolerance {alpha:g}: when every episode fails the flat penalty carries no gradient; reduce the dual step, use a "
                                   "fixed price, or start from a controller that sometimes survives")
    close = abs(rate_w - alpha) <= tol * alpha + 1.0 / math.sqrt(max(1, sum(t.get("episodes", 1) for t in trace[-w:])))
    if not drifting and close:
        return MultiplierDiagnosis("stable", last, eq, rate_w, alpha,
                                   f"multiplier {last:.4g} (price-equivalent m = (1-beta) nu = {eq:.4g}) shows no drift over the last {w} rollouts "
                                   f"({drift:+.3g} per rollout, standard error {drift_se:.2g}), and the "
                                   f"failure rate {rate_w:.3g} is near the tolerance {alpha:g}. A multiplier that has converged returns the fixed-price "
                                   "optimum at that price; a level reached at the tolerance itself does not pass a validation test at that tolerance")
    if not drifting and not close:
        return MultiplierDiagnosis("off-target", last, eq, rate_w, alpha,
                                   f"multiplier shows no drift but the failure rate {rate_w:.3g} is not near the tolerance {alpha:g}: the rollouts are too "
                                   "few or too small to resolve the tolerance")
    return MultiplierDiagnosis("trending", last, eq, rate_w, alpha,
                               f"multiplier still {'rising' if drift > 0 else 'falling'} over the last {w} rollouts ({drift:+.3g} per rollout, standard "
                               f"error {drift_se:.2g}; failure rate {rate_w:.3g}, tolerance {alpha:g}): it has not converged")


@dataclass
class SeedDispersion:
    n: int
    mean: float
    sd: float
    min: float
    max: float

    def explain(self, name: str) -> str:
        return f"{name}: mean {self.mean:.4g}, between-seed sd {self.sd:.3g}, range [{self.min:.4g}, {self.max:.4g}] over {self.n} seeds"


def seed_dispersion(values: Sequence[float]) -> SeedDispersion:
    v = np.asarray(values, dtype=float)
    return SeedDispersion(len(v), float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, float(v.min()), float(v.max()))


def objective_gap_trace(objective: Sequence[float], final: Optional[float] = None) -> np.ndarray:
    """J(mu_l) - J* along a run, with J* the final (or supplied) value; for a maximized value use -V."""
    J = np.asarray(objective, dtype=float)
    Jf = float(J[-1]) if final is None else float(final)
    return J - Jf


# ---------------------------------------------------------------------------------------------
# Convergence of a training run, read from its per-batch record
# ---------------------------------------------------------------------------------------------
_LABELS = {"failure_rate": "the failure rate", "U": "the utility", "V_m": "the priced objective",
           "log_std": "the action dispersion (log std)", "multiplier": "the adaptive multiplier"}


@dataclass
class QuantityCheck:
    key: str
    previous: float            # mean over the window before the last
    last: float                # mean over the last window
    change: float
    noise: float               # standard error of the change, from the batch-to-batch variation
    scale: float               # the size against which the change is judged
    status: str                # 'stable', 'rising' or 'falling'

    @property
    def label(self) -> str:
        return _LABELS.get(self.key, self.key)

    def line(self) -> str:
        if self.status == "stable":
            return f"{self.label}: stable ({self.previous:.4g} then {self.last:.4g})"
        return f"{self.label}: still {self.status} ({self.previous:.4g} then {self.last:.4g}, a change of {abs(self.change) / self.scale:.0%})"


@dataclass
class TrainingConvergence:
    """The convergence check of a training run: the last `window` batches against the `window` before them."""
    status: str                # 'converged', 'not converged' or 'too short'
    window: int
    batches: int
    rel_tol: float
    checks: List[QuantityCheck] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    @property
    def moving(self) -> List[QuantityCheck]:
        return [c for c in self.checks if c.status != "stable"]

    def explain(self) -> str:
        if self.status == "too short":
            return (f"Training convergence: too short to judge ({self.batches} batches recorded; the check compares two windows "
                    f"of at least 10 batches). A run this short has rarely converged: train longer before comparing or deploying.")
        head = (f"Training convergence over the last {self.window} batches, compared with the {self.window} before: "
                + ("converged." if self.converged else "not converged."))
        lines = [head] + ["  " + c.line() for c in self.checks]
        if self.converged:
            lines.append(f"Every quantity changed by less than {self.rel_tol:.0%} or by less than its batch-to-batch noise. This means "
                         "that the run stopped changing, not that its policy is good: a learning rate that decays to zero makes any "
                         "run stop changing at its end, and a run can stop in a bad place. A longer run may still improve; the "
                         "validation decides whether the policy meets the requirement.")
        else:
            lines.append("Train longer before comparing this run with others or deploying its policy; a policy taken from a run "
                         "that is still changing is the policy at the end of the run, and it should be validated as such.")
        return "\n".join(lines)


def training_convergence(trace: Sequence[dict], window: Optional[int] = None, rel_tol: float = 0.05,
                         keys: Sequence[str] = ("failure_rate", "U", "V_m", "log_std", "multiplier")) -> TrainingConvergence:
    """Has a training run stopped changing? Reads the per-batch record of a run (BudgetMonitor.trace, or the 'monitor'
    of a campaign record) and compares, for each quantity, the mean over the last `window` batches with the mean over the
    `window` batches before. A quantity is still changing when its change exceeds both twice its standard error (from
    the batch-to-batch variation) and `rel_tol` times its size (the larger of its level and its batch-to-batch
    variation). The default window is a tenth of the batches, at least 10, and fewer than two windows of batches give
    'too short'. Evidence about one run, not a convergence guarantee: the theory's convergence results concern the exact
    population flow, not stochastic training, and a learning rate that decays to zero makes any run stop changing."""
    n = len(trace)
    w = int(window) if window is not None else max(10, n // 10)
    if n < 2 * w or w < 5:
        return TrainingConvergence("too short", w, n, rel_tol)
    checks = []
    for key in keys:
        x = np.array([t.get(key, np.nan) for t in trace], dtype=float)
        prev, last = x[-2 * w:-w], x[-w:]
        prev, last = prev[np.isfinite(prev)], last[np.isfinite(last)]
        if len(prev) < 3 or len(last) < 3:
            continue
        mp, ml = float(prev.mean()), float(last.mean())
        noise = float(np.sqrt(prev.var(ddof=1) / len(prev) + last.var(ddof=1) / len(last)))
        spread = float(np.sqrt(0.5 * (prev.var(ddof=1) + last.var(ddof=1))))
        scale = max(abs(mp), abs(ml), spread, 1e-12)
        change = ml - mp
        moving = abs(change) > 2.0 * noise and abs(change) > rel_tol * scale
        checks.append(QuantityCheck(key, mp, ml, change, noise, scale,
                                    "stable" if not moving else ("rising" if change > 0 else "falling")))
    status = "converged" if checks and all(c.status == "stable" for c in checks) else "not converged"
    return TrainingConvergence(status, w, n, rel_tol, checks)


def plot_training(trace: Sequence[dict], report: Optional[TrainingConvergence] = None, keys: Sequence[str] = ("failure_rate", "U", "log_std"),
                  axes=None, title: Optional[str] = None):
    """Training curves per batch (needs matplotlib), with the two windows of the convergence check shaded."""
    import matplotlib.pyplot as plt
    keys = [k for k in keys if any(np.isfinite(t.get(k, np.nan)) for t in trace)]
    if axes is None:
        fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.6))
    axes = np.atleast_1d(axes)
    steps = np.array([t.get("steps", i) for i, t in enumerate(trace)], dtype=float)
    unit, scale = ("million steps", 1e6) if steps.max() >= 1e6 else ("thousand steps", 1e3)
    report = training_convergence(trace) if report is None else report
    for ax, key in zip(axes, keys):
        ax.plot(steps / scale, [t.get(key, np.nan) for t in trace], lw=0.8)
        if report.status != "too short":
            w = report.window
            ax.axvspan(steps[-2 * w] / scale, steps[-w - 1] / scale, color="0.85", lw=0)
            ax.axvspan(steps[-w] / scale, steps[-1] / scale, color="0.7", lw=0)
        ax.set_title(_LABELS.get(key, key), fontsize=8); ax.set_xlabel(unit, fontsize=8)
    fig = axes[0].figure
    fig.suptitle((title + ": " if title else "") + {"converged": "converged", "not converged": "not converged",
                                                   "too short": "too short to judge"}[report.status], fontsize=9)
    fig.tight_layout()
    return fig


__all__ = ["KernelConstants", "FeatureConstants", "curvature_constants", "gaussian_prior_curvature", "supplied_curvature", "tv_constant",
           "rate", "certified_kappa", "time_to_fraction", "CurvatureBudget", "curvature_budget", "learning_risk_bound",
           "learning_time_for_slack", "initial_gap_upper", "finite_training_transfer", "risk_slack_from_prior", "kappa_for_tolerance",
           "action_slack_from_prior", "fit_exponential_rate", "TrendTest", "trend_test",
           "MultiplierDiagnosis", "multiplier_diagnosis", "SeedDispersion", "seed_dispersion", "objective_gap_trace",
           "QuantityCheck", "TrainingConvergence", "training_convergence", "plot_training"]
