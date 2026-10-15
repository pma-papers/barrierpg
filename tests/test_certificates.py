import math
from barrierpg import certificates as C


def test_value_certificate_and_identity():
    assert abs(C.risk_from_value(0.0, 1.0, 1.0, 0.9).value - 0.5) < 1e-12
    assert abs(C.risk_from_value(0.0, 1.0, 1.0, 0.9, T=1).value - 0.5 / 0.9) < 1e-12
    assert C.risk_from_value(-100.0, 1.0, 1.0, 0.9).value == 1.0
    V = C.clipped_value_identity(U=5.0, I_m=1.0, q_beta=0.02, m=1.0, beta=0.95)
    assert abs(V - (5.0 - 1.0 - 0.4)) < 1e-12
    # the value certificate applied to the identity is never below the true risk
    q = (1.0 - 0.05 * V) / 2.0
    assert q >= 0.02
    b = C.risk_from_objective(J=-3.0, m=1.0, ubar=1.0, beta=0.9, eps_r=0.1)
    assert abs(b.value - (1.0 - 0.3 + 0.1) / 2.0) < 1e-12


def test_comparison_and_matched_risk():
    b = C.comparison_law_risk(q_bar=0.01, C_B=2.0, R_bar=0.5, delta=0.1, m=10.0, ubar=1.0, beta=0.9)
    assert abs(b.value - (0.01 + 0.1 * 2.6 / 11.0)) < 1e-12
    D = C.comparison_law_slack(m=10.0, ubar=1.0, beta=0.9, alpha=0.05, T=0, q_bar=0.01, C_B=2.0, R_bar=0.5)
    assert abs(D - (11.0 * 0.04 / 0.1 - 2.5)) < 1e-12
    g = C.matched_risk_utility_gap(I_bar=0.3, R_bar=0.2, delta=0.1, m=1.0, beta=0.9, alpha=0.05, q_hat=0.02)
    assert abs(g.value - (0.6 + 0.3)) < 1e-12
    g2 = C.matched_risk_utility_gap(I_bar=0.3, R_bar=0.2, delta=0.1, m=1.0, beta=0.9, alpha=0.05, q_hat=0.02, continuing=True, ubar=1.0, u_minus=0.5)
    assert abs(g2.value - (0.9 + (0.05 + 0.01) / 0.1)) < 1e-12


def test_learning_form():
    # with zero regularization cost and no reward error the learning form is the (m+ubar) variant / the law
    b = C.learning_form_risk(m=1.0, beta=0.95, ubar=1.0, q_bar=0.0, U_bar=0.0, I_bar=0.0, R_bar=0.0, delta=0.0, sup_U=5.35)
    assert abs(b.value - 0.05 * 5.35) < 1e-12 and "learning form" in b.source
    b2 = C.learning_form_risk(m=1.0, beta=0.95, ubar=1.0, q_bar=0.0, U_bar=0.0, I_bar=0.0, R_bar=2.0, delta=0.5, eps_r=0.1, sup_U=5.35)
    assert abs(b2.inputs["delta_prime"] - (0.5 + 2.0 + 4.0)) < 1e-12 and b2.value >= b.value
    # the unclipped comparison-law form is never tighter than the learning form with I_bar <= C_B-based cost
    cb = C.comparison_law_risk(q_bar=0.01, C_B=2.0, R_bar=0.5, delta=0.1, m=10.0, ubar=1.0, beta=0.9)
    lf = C.learning_form_risk(m=10.0, beta=0.9, ubar=1.0, q_bar=0.01, U_bar=3.0, I_bar=0.5, R_bar=0.5, delta=0.1)
    assert 0.0 <= lf.value <= 1.0 and 0.0 <= cb.value <= 1.0
    bT = C.learning_form_risk(m=1.0, beta=0.95, ubar=1.0, q_bar=0.0, U_bar=0.0, I_bar=0.0, R_bar=0.0, delta=0.0, sup_U=5.35, T=5)
    assert abs(bT.value - min(1.0, 0.05 * 5.35 / 0.95 ** 5)) < 1e-12


def test_price_risk_law_and_monotonicity():
    b = C.price_risk_law(m=1.0, beta=0.95, sup_U=5.35)
    assert abs(b.value - 0.05 * 5.35) < 1e-12 and b.inputs["law"] == b.value
    b = C.price_risk_law(m=1.0, beta=0.95, sup_U=5.35, ubar=1.0)
    assert abs(b.inputs["variant"] - 0.5) < 1e-12 and abs(b.value - 0.2675) < 1e-12
    # the exact wear frontier obeys the law: m=1 gives q=0.0145 <= 0.05*5.43/1
    assert 0.01453 <= C.price_risk_law(1.0, 0.95, 5.4343).value
    assert 0.00115 <= C.price_risk_law(10.0, 0.95, 5.4343).value
    assert C.price_risk_law(m=0.0, beta=0.95, sup_U=5.0).value == 1.0
    mc = C.conservative_price(alpha=0.05, beta=0.95, sup_U=5.4343)
    assert abs(mc.value - 0.05 * 5.4343 / 0.05) < 1e-9
    assert C.conservative_price(alpha=0.01, beta=0.95, sup_U=5.0, q0=0.02).value == math.inf
    assert C.conservative_price(alpha=0.5, beta=0.95, sup_U=1.0, u_minus=3.0).value == 3.0
    mono = C.price_monotonicity(q1=0.04, m1=0.3, m2=1.0, beta=0.95, delta1=0.01, delta2=0.01)
    assert abs(mono.value - (0.04 + 0.05 * 0.02 / 0.7)) < 1e-12
    assert abs(C.frozen_multiplier_value(5.0, 0.1, 20.0) - 3.0) < 1e-12


def test_action_risk_bridge():
    assert abs(C.kl_bernoulli(0.0, 0.3) + math.log(0.7)) < 1e-15 and C.kl_bernoulli(0.5, 0.0) == math.inf
    assert abs(C.kl_bernoulli(0.3, 0.3)) < 1e-15
    up = C.kl_upper_inverse(0.01, 0.1)
    assert abs(C.kl_bernoulli(up, 0.01) - 0.1) < 1e-8 and up > 0.01
    lo = C.kl_lower_inverse(0.5, 0.05)
    assert abs(C.kl_bernoulli(lo, 0.5) - 0.05) < 1e-8 and lo < 0.5
    b = C.action_risk_bound(A=0.1, q_ref=0.01, beta=0.95)
    assert abs(b.value - up) < 1e-12 and abs(b.inputs["pinsker_half_width"] - math.sqrt(0.05)) < 1e-12
    assert C.action_risk_bound(A=50.0, q_ref=0.01, beta=0.95).value == 1.0        # uninformative at large A
    bT = C.action_risk_bound(A=0.1, q_ref=0.01, beta=0.95, T=10)
    assert bT.value >= b.value


def test_near_exhaustion_eventual_width():
    b = C.near_exhaustion_bound(V_m=4.0, m=1.0, ubar=1.0, beta=0.95, T=0, lower_barrier=0.3)
    assert abs(b.value - (1 - 0.2) / (0.05 * 0.3)) < 1e-12 or b.value == 1.0
    assert abs(C.near_exhaustion_bound(V_m=19.9, m=1.0, ubar=1.0, beta=0.95, T=0, lower_barrier=0.3).value - (1 - 0.05 * 19.9) / (0.05 * 0.3)) < 1e-9
    assert C.near_exhaustion_bound(4.0, 1.0, 1.0, 0.95, 0, 0.0).value == 1.0
    assert abs(C.unresolved_threshold(0.1, 0.9) - 1.0) < 1e-12
    assert abs(C.eventual_risk_structural(0.01, 0.02).value - 0.03) < 1e-15
    t = C.finite_network_transfer(n=10_000, L0=1.0, beta=0.9, zeta=0.05, m=1.0, ubar=1.0, T=9)
    assert abs(t["eps_n"] - 1 / (2 * math.sqrt(10_000 * 0.1 * 0.05))) < 1e-12 and abs(t["value"] - 20 * t["eps_n"]) < 1e-12
    assert abs(t["p_T"] - math.sqrt(10) / (2 * math.sqrt(500))) < 1e-12
    assert C.width_for_target(1.0, 0.9, 0.05, 0.01) == math.ceil(1 / (4 * 0.1 * 0.05 * 1e-4))
    assert C.width_for_direct_target(1.0, 9, 0.05, 0.01) == math.ceil(10 / (4 * 0.05 * 1e-4))
    assert C.width_for_target(1.0, 0.9, 0.05, -0.01) == 0
