"""A deployment report must not silently assume an unknown optimizer error is zero."""
import numpy as np
from bpg import BudgetSpec, Requirement, Stage, interpret
from bpg.evaluate import Evaluation


def candidate():
    n = 100
    zeros = np.zeros(n)
    return Evaluation(n=n, seed=1, T=0, beta=.9, m=1., failures=0,
        failures_any=0, p_T=0., q_beta=0., U=0., U_se=0., I_m=0.,
        V_m=0., V_m_se=0., mean_length=1., near={.05: 0.},
        min_reserve_quantiles={.05: 1.}, geometric_stop=False, event="p_T",
        tau=np.full(n, -1), U_episodes=zeros, I_episodes=zeros,
        V_episodes=zeros, clipped_returns=zeros, failed=np.zeros(n, dtype=bool),
        min_reserve=np.ones(n))


def test_unknown_error_does_not_generate_price_comparison():
    spec = BudgetSpec(budgets=[1.], beta=.9, ubar=1.)
    req = Requirement(T=0, alpha=.05, zeta=.05)
    for stage in (Stage(m=1.), Stage(m=1., penalty="multiplier", multiplier=10.)):
        rep = interpret(spec, req, stage, candidate(), sup_U=10., delta=None,
                        comparator=dict(U0=0., I0=0., q0=0., C_B=0., R=0.))
        assert rep.certified  # a valid finite-horizon decision requires no optimizer theorem
        assert "no optimality or convergence claim" in rep.text()
        assert not any("Proposition 3.3(ii)" in b.source for b in rep.bounds)
        assert not any("optimal at the risk it attains" in x for x in rep.reading)


def test_known_error_keeps_the_conditional_price_comparison():
    rep = interpret(BudgetSpec(budgets=[1.], beta=.9, ubar=1.),
                    Requirement(T=0, alpha=.05, zeta=.05), Stage(m=1.),
                    candidate(), sup_U=10., delta=.2)
    assert any("Proposition 3.3(ii)" in b.source for b in rep.bounds)
    assert any("delta = 0.2" in x for x in rep.reading)
