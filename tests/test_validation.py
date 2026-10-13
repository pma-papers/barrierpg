import math
from bpg import validation as V


def test_binomial_and_floor():
    assert abs(V.binomial_upper(0, 1000, 0.05) - (1 - 0.05 ** (1 / 1000))) < 1e-12
    assert V.binomial_upper(10, 10, 0.05) == 1.0
    up = V.binomial_upper(3, 20000, 0.05 / 120)
    assert 0.0001 < up < 0.05
    assert abs(V.resolution_floor(20000, 0.05 / 120) - (1 - (0.05 / 120) ** (1 / 20000))) < 1e-15
    n = V.episodes_for_tolerance(0.05, 0.05)
    assert n == math.ceil(math.log(0.05) / math.log(0.95)) and V.resolution_floor(n, 0.05) <= 0.05
    n2 = V.episodes_for_tolerance(0.05, 0.05, expected_failure_rate=0.02)
    assert n2 >= n and V.binomial_upper(round(0.02 * n2), n2, 0.05) <= 0.05
    d = V.validate_direct(0, 50, 0.05, 0.05)
    assert not d.passed and d.floor > 0.05
    d = V.validate_direct(3, 20000, 0.05, 0.05, family=120)
    assert d.passed and d.upper < 0.05 and "passes" in d.bound(19).note


def test_value_validation():
    lo = V.hoeffding_value_lower(mean_return_H=3.0, H=100, N=10_000, zeta=0.05, m=1.0, ubar=1.0, beta=0.95)
    S = (1 - 0.95 ** 100) / 0.05
    assert abs(lo - (3.0 - 2 * S * math.sqrt(math.log(20) / 20000) - 0.95 ** 100 / 0.05)) < 1e-12
    b = V.risk_from_validation_value(3.0, 100, 10_000, 0.05, 1.0, 1.0, 0.95, T=19)
    assert abs(b.value - min(1.0, (1 - 0.05 * lo) / (2 * 0.95 ** 19))) < 1e-12
    chk = V.value_resolution_check(beta=0.95, H=300, T=19, N=20000, zeta=0.05, alpha=0.05)
    assert chk["floor"] > chk["tail_term"] and isinstance(chk["attainable"], bool)
    chk2 = V.value_resolution_check(beta=0.95, H=25, T=19, N=20000, zeta=0.05, alpha=0.05)
    assert not chk2["attainable"]                                        # beta^(H-T) = 0.95^6 > 0.05
    p = V.physical_horizon_risk(V_mH=10.0, H=100, m=1.0, ubar=1.0, beta=0.95, T=19, N=1000, zeta=0.05)
    assert 0.0 <= p.value <= 1.0 and p.inputs["V_used"] < 10.0
