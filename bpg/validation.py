"""Validation of the deployed candidate (Section 5.2 and Algorithm 1 of the paper).

Two tests, both conditional on training and network construction, both on fresh trajectories of
the fixed candidate:
  direct   the Bernoulli events {tau_i <= T}: one-sided exact binomial upper endpoint; at zero
           failures it is 1 - zeta^(1/N), never zero;
  value    a one-sided Hoeffding lower confidence bound on V_m from returns truncated after H
           actions, inserted into the value certificate (Proposition 3.1).
zeta is an estimation error probability; alpha is the tolerated failure probability; they are not
interchangeable. Error probabilities are allocated across candidates, horizons and tests before
taking minima or selecting a policy (a union bound, which needs no independence).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .bounds import Bound, clip01
from .certificates import risk_from_value
from .paper import cite
from .spec import discounted_sum_to


def binomial_upper(k: int, n: int, err: float) -> float:
    """One-sided exact (Clopper-Pearson) upper endpoint at confidence 1 - err: the (1-err) quantile of
    Beta(k+1, n-k); 1 when k = n."""
    if n <= 0:
        raise ValueError("n >= 1")
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - err ** (1.0 / n)
    from scipy.stats import beta as beta_dist
    return float(beta_dist.ppf(1.0 - err, k + 1, n - k))


def resolution_floor(n: int, err: float) -> float:
    """Smallest tolerance that n failure-free episodes can certify at error err: 1 - err^(1/n)."""
    return 1.0 - err ** (1.0 / n)


def episodes_for_tolerance(alpha: float, err: float, expected_failure_rate: float = 0.0) -> int:
    """Episodes needed so that the direct test can certify alpha: with no failures,
    N >= log(err) / log(1 - alpha); with an expected failure rate p < alpha, the smallest N whose
    upper endpoint at k = round(p N) failures is at most alpha (searched numerically)."""
    if alpha <= 0:
        return 0 if alpha < 0 else int(1e18)
    n0 = int(math.ceil(math.log(err) / math.log(1.0 - alpha)))
    if expected_failure_rate <= 0:
        return n0
    if expected_failure_rate >= alpha:
        raise ValueError("the expected failure rate must lie below the tolerance to be certifiable")
    n = n0
    while n < 10_000_000:
        k = int(round(expected_failure_rate * n))
        if binomial_upper(k, n, err) <= alpha:
            return n
        n = int(n * 1.15) + 1
    return n


def allocate_error(zeta: float, family: int) -> float:
    """Per-statement error for `family` simultaneous statements at overall error zeta (union bound)."""
    return zeta / family


@dataclass
class DirectValidation:
    failures: int
    n: int
    alpha: float
    err: float
    upper: float
    floor: float
    event: str = "p_T"

    @property
    def passed(self) -> bool:
        return self.upper <= self.alpha

    def bound(self, T: Optional[int] = None) -> Bound:
        if self.event == "q_beta":
            name, hyp = "q_beta upper bound (validated)", "N fresh independent episodes under the geometric clock (each ends with probability 1-beta per period)"
        else:
            name, hyp = ("p_T upper bound (validated)" if T is None else f"p_{T} upper bound (validated)"), "N fresh independent trajectories of the fixed candidate through action T"
        return Bound(name, self.upper, cite("validation") + ", exact binomial test",
                     [hyp,
                      f"error {self.err:.3g} allocated to this statement (family allocation)"],
                     dict(failures=self.failures, n=self.n, alpha=self.alpha, err=self.err, floor=self.floor),
                     note=("passes" if self.passed else "does not pass") + f" the tolerance {self.alpha:g}; "
                          f"resolution floor {self.floor:.3g} (zero failures certify no less). Failure to certify does not prove violation; "
                          "a passed test at one horizon does not certify a longer one")


def validate_direct(failures: int, n: int, alpha: float, zeta: float, family: int = 1) -> DirectValidation:
    err = allocate_error(zeta, family)
    return DirectValidation(int(failures), int(n), float(alpha), err, binomial_upper(failures, n, err), resolution_floor(n, err))


# ---------------------------------------------------------------------------------------------
# Value-based validation
# ---------------------------------------------------------------------------------------------
def hoeffding_value_lower(mean_return_H: float, H: int, N: int, zeta: float, m: float, ubar: float, beta: float) -> float:
    """V_m >= V_hat_H - (m+ubar)(1-beta^H)/(1-beta) sqrt(log(1/zeta)/(2N)) - m beta^H/(1-beta)  (Section 5.2, Hoeffding lower bound):
    the mean of N returns truncated after H actions, less the Hoeffding term and the omitted tail."""
    S_H = discounted_sum_to(beta, H)
    return float(mean_return_H - (m + ubar) * S_H * math.sqrt(math.log(1.0 / zeta) / (2.0 * N)) - m * beta ** H / (1.0 - beta))


def risk_from_validation_value(mean_return_H: float, H: int, N: int, zeta: float, m: float, ubar: float, beta: float, T: int) -> Bound:
    """p_T <= min{1, (ubar - (1-beta) V_lower) / ((m+ubar) beta^T)} with the Hoeffding lower bound (Section 5.2, value-based validation)."""
    V_low = hoeffding_value_lower(mean_return_H, H, N, zeta, m, ubar, beta)
    b = risk_from_value(V_low, m, ubar, beta, T)
    b.source = cite("validation") + ", value-based test with " + cite("value_certificate")
    b.hypotheses = ["returns are the clipped rewards r_m in [-m, ubar], truncated after H actions, N fresh trajectories",
                    "ubar is a valid upper bound on the one-step utility"]
    b.inputs.update(V_lower=V_low, mean_return_H=mean_return_H, H=H, N=N, zeta=zeta)
    return b


def value_resolution_check(beta: float, H: int, T: int, N: int, zeta: float, alpha: float) -> dict:
    """Before training: the value-based test can certify p_T <= alpha < 1 only if
    beta^(H-T) + (1-beta^H)/beta^T sqrt(log(1/zeta)/(2N)) <= alpha  (Section 5.2, resolution floor).
    Increasing m cannot repair a failure of this check."""
    lhs = beta ** (H - T) + (1.0 - beta ** H) / beta ** T * math.sqrt(math.log(1.0 / zeta) / (2.0 * N))
    return dict(attainable=lhs <= alpha, floor=lhs, alpha=alpha,
                tail_term=beta ** (H - T), statistical_term=(1.0 - beta ** H) / beta ** T * math.sqrt(math.log(1.0 / zeta) / (2.0 * N)))


def physical_horizon_risk(V_mH: float, H: int, m: float, ubar: float, beta: float, T: int,
                          N: Optional[int] = None, zeta: Optional[float] = None) -> Bound:
    """Task that ends after H actions, T < H:
    p_T <= min{1, (ubar S_H - V_m^(H)) / ((m+ubar) beta^T (1-beta^(H-T))/(1-beta))}  (Section 5.2, physical horizon).
    With N and zeta, V_mH is the mean complete return and the Hoeffding term (m+ubar) S_H sqrt(log(1/zeta)/(2N))
    is subtracted first (no omitted tail)."""
    S_H = discounted_sum_to(beta, H)
    V = V_mH
    if N is not None and zeta is not None:
        V = V_mH - (m + ubar) * S_H * math.sqrt(math.log(1.0 / zeta) / (2.0 * N))
    den = (m + ubar) * beta ** T * (1.0 - beta ** (H - T)) / (1.0 - beta)
    return Bound(f"p_{T} upper bound (physical horizon H={H})", clip01((ubar * S_H - V) / den),
                 cite("validation") + ", physical-horizon value certificate",
                 ["failure at t is charged the remaining discounted total -m(1-beta^(H-t))/(1-beta)", "ubar bounds the utility"],
                 dict(V_mH=V_mH, V_used=V, H=H, m=m, ubar=ubar, beta=beta, T=T, N=N, zeta=zeta))


def hoeffding_lower(mean: float, N: int, zeta: float, low: float, high: float) -> float:
    """One-sided Hoeffding lower confidence bound at error zeta for the mean of N independent returns in [low, high]."""
    return float(mean - (high - low) * math.sqrt(math.log(1.0 / zeta) / (2.0 * N)))


def lump_sanction_risk(V_H: float, H: int, nu: float, ubar: float, beta: float, T: int, N: Optional[int] = None,
                       zeta: Optional[float] = None, u_minus: float = 0.0) -> Bound:
    """A task that ends after H actions with a flat penalty nu charged once at the failing action (a lump sanction, the
    frozen-multiplier form): since ubar S_H - V^(H) >= nu E[beta^tau 1{tau <= T}],
        p_T <= min{1, (ubar S_H - V^(H)) / (nu beta^T)}   (Section 6.6).
    With N and zeta, V_H is the mean complete return and is first lowered by the Hoeffding term of the return's own range,
    (ubar + u_minus) S_H + nu: a penalty charged once widens the range by nu, not by a per-period price times S_H."""
    S_H = discounted_sum_to(beta, H)
    V = V_H
    if N is not None and zeta is not None:
        V = hoeffding_lower(V_H, N, zeta, -u_minus * S_H - nu, ubar * S_H)
    return Bound(f"p_{T} upper bound (lump sanction, H={H})", clip01((ubar * S_H - V) / (nu * beta ** T)),
                 cite("application") + ", value certificate with a flat penalty",
                 ["the penalty nu is charged once, at the failing action; the reward before failure is the utility, unclipped",
                  "-u_minus <= u <= ubar"],
                 dict(V_H=V_H, V_used=V, H=H, nu=nu, ubar=ubar, beta=beta, T=T, N=N, zeta=zeta, u_minus=u_minus),
                 note="informative when nu beta^T is large against the utility at stake; direct validation is often sharper")


__all__ = ["binomial_upper", "resolution_floor", "episodes_for_tolerance", "allocate_error", "DirectValidation",
           "validate_direct", "hoeffding_value_lower", "risk_from_validation_value", "value_resolution_check",
           "physical_horizon_risk", "hoeffding_lower", "lump_sanction_risk"]
