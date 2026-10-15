import math
import numpy as np
from barrierpg import convergence as K
from barrierpg import frontier as F


def test_curvature_constants_reproduce_the_paper():
    # The paper's solvable flow: beta = 0.9, m = 1, ubar = 1, plateau feature with the suprema of
    # chi' and chi''; the paper reports C_m = 1940 and kappa = 1.05 C_m = 2037 (lambda_W = 1).
    # Reconstruct c_psi from C_m: c_psi = C_m / (m + ubar) = 970 = L2/0.01 + L1^2/0.001.
    fc = K.FeatureConstants(L0=1.0, L1=0.3, L2=2.0)
    c = K.curvature_constants(fc, beta=0.9, m=1.0, ubar=1.0)
    assert abs(c["c_psi"] - (2.0 / 0.01 + 0.09 / 0.001)) < 1e-9 and abs(c["C_m"] - 2 * c["c_psi"]) < 1e-9
    CA = 1.0 * (1 + 2.7) * 2.0 / 0.01 + ((1 + 2) / 0.1 + 2 * 0.9 * 1.0 * 2.1 / 0.001) * 0.09
    assert abs(c["C_A"] - CA) < 1e-9
    cb = K.curvature_budget(fc, 0.9, 1.0, 1.0, kappa=1.05 * c["C_m"], lambda_H=1.0)
    assert cb.certified and abs(cb.kappa_certified - 1.05 * c["C_m"]) < 1e-9
    assert not K.curvature_budget(fc, 0.9, 1.0, 1.0, kappa=1.0, lambda_H=1.0).certified
    assert abs(K.gaussian_prior_curvature(2.0) - 0.25) < 1e-15 and K.tv_constant("kl+chi2", 1.0) == 2 / 3
    assert K.time_to_fraction(97.0, 1e-3) == math.log(1e3) / 194
    assert K.learning_time_for_slack(E0=10.0, D=1.0, Lambda=2.0) == math.log(10) / 4 and K.learning_time_for_slack(10.0, -1.0, 2.0) == math.inf
    assert abs(K.risk_slack_from_prior(1.0, 0, 2.0, 12.2, 10.0) - 0.5 * math.sqrt(2 * 12.2 / 10)) < 1e-12
    assert abs(K.kappa_for_tolerance(1.0, 0, 2.0, 12.2, 0.05, 0.01) - 2 * 12.2 / (4 * 0.04 ** 2)) < 1e-9
    assert abs(K.action_slack_from_prior(12.2, 2.0, 0.9, 0) - math.sqrt(12.2 / 4)) < 1e-12


def test_tanh_feature_constants():
    fc = K.FeatureConstants.tanh_unit(scale=2.0, input_bound=3.0)
    assert fc.L0 == 2.0 and fc.L1 == 6.0 and abs(fc.L2 - 2 * 9 * 4 / (3 * math.sqrt(3))) < 1e-12
    # sup |tanh''| = 4/(3 sqrt 3) numerically
    x = np.linspace(-5, 5, 200001); t = np.tanh(x); d2 = -2 * t * (1 - t ** 2)
    assert abs(np.max(np.abs(d2)) - 4 / (3 * math.sqrt(3))) < 1e-6


def test_trace_diagnostics():
    l = np.geomspace(1e-4, 10, 200); gap = 5.0 * np.exp(-3.0 * l)
    assert abs(K.fit_exponential_rate(l, gap, 1e-8, 1e-2) - 3.0) < 1e-6
    rng = np.random.default_rng(0)
    flat = 1.0 + 0.01 * rng.standard_normal(100)
    assert not K.trend_test(flat, 40).trend_detected
    rising = np.linspace(0, 1, 100) + 0.01 * rng.standard_normal(100)
    assert K.trend_test(rising, 40).trend_detected
    # multiplier traces
    settled = [dict(multiplier=5.0 + 0.1 * rng.standard_normal(), failure_rate=0.05 + 0.01 * rng.standard_normal(), episodes=400) for _ in range(60)]
    assert K.multiplier_diagnosis(settled, 0.05, 0.95).status == "stable"
    runaway = [dict(multiplier=10.0 * k, failure_rate=1.0, episodes=50) for k in range(60)]
    assert K.multiplier_diagnosis(runaway, 0.05, 0.95).status == "increasing"
    slack = [dict(multiplier=0.0, failure_rate=0.01, episodes=400) for _ in range(60)]
    assert K.multiplier_diagnosis(slack, 0.05, 0.95).status == "inactive"
    drifting = [dict(multiplier=k * 0.2, failure_rate=0.3, episodes=400) for k in range(60)]
    assert K.multiplier_diagnosis(drifting, 0.05, 0.95).status == "trending"
    assert K.multiplier_diagnosis(settled[:3], 0.05, 0.95).status == "short"
    sd = K.seed_dispersion([1.0, 2.0, 3.0]); assert sd.mean == 2.0 and sd.n == 3



def _dual_ascent_trace(seed, nu0, step, n=240, nu_star=34.0, alpha=0.02, episodes=400):
    """Dual ascent against a learner whose risk falls with the multiplier and equals alpha at nu_star."""
    rng = np.random.default_rng(seed); nu = nu0; trace = []
    for _ in range(n):
        q = min(1.0, alpha * math.exp(-(nu - nu_star) / 25.0))
        rate = rng.binomial(episodes, q) / episodes
        nu = max(0.0, nu + step * (rate - alpha))
        trace.append(dict(multiplier=nu, failure_rate=rate, episodes=episodes))
    return trace


def test_multiplier_drift_is_tested_on_its_updates():
    # The multiplier is a running sum of its updates. A slope fitted to its levels reports a trend in most windows
    # after convergence; the drift of the updates does not, and still detects a multiplier that is climbing.
    converged = [K.multiplier_diagnosis(_dual_ascent_trace(s, 34.0, 17.0), 0.02, 0.95, 40).status for s in range(40)]
    assert converged.count("stable") >= 34
    climbing = [K.multiplier_diagnosis(_dual_ascent_trace(s, 0.0, 2.0), 0.02, 0.95, 40).status for s in range(40)]
    assert climbing.count("trending") == 40


def test_frontier_checks():
    from barrierpg.examples.wear import EXACT_FRONTIER
    runs = [F.Run(m=m, q_beta=q, U=U, label="exact") for m, (q, U) in EXACT_FRONTIER.items() if m > 0]
    chk = F.check_runs(runs, beta=0.95, sup_U=5.4343, delta=1e-4)
    assert chk.consistent, chk.explain()
    assert not chk.dominated
    bad = runs + [F.Run(m=1.0, q_beta=0.03, U=4.0, label="poor learner")]
    chk2 = F.check_runs(bad, beta=0.95, sup_U=5.4343, delta=0.0)
    assert any("poor learner" in d for d in chk2.dominated)
    viol = [F.Run(m=10.0, q_beta=0.2, U=3.0)]
    assert F.check_runs(viol, beta=0.95, sup_U=5.4343).law_violations
    pts = F.frontier_points(runs)
    assert len(pts) == len(runs) and abs(F.interpolate_frontier(runs, 0.01453) - 4.6131) < 1e-9
    assert abs(F.complementarity_gap(0.05, 0.01, 1.0, 0.95) - 0.8) < 1e-12
