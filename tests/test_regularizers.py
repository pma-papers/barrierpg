from bpg import BudgetSpec, Requirement, Stage, plan
from bpg.regularizers import advise_regularizers, regularizer_effects
from bpg.convergence import FeatureConstants


def test_advice_by_condition():
    spec = BudgetSpec(budgets=[1.0], beta=0.95, ubar=1.0)
    a = advise_regularizers(spec, Stage(m=0.03, eta=0.1), action_reference_trusted=True, price_trusted=False, incumbent_available=True, prior_sigma=1.0)
    mech = [r.mechanism for r in a.recommendations]
    assert "action anchor" in mech and "parameter anchor" in mech and "Gaussian prior width" in mech and "strength kappa" in mech
    assert any("eta > 0" in r.setting for r in a.recommendations if r.mechanism == "action anchor")
    assert any("inherits the prior policy" in c for c in a.cautions)
    b = advise_regularizers(spec, Stage(m=1.0), action_reference_trusted=True, price_trusted=True, model_trusted=True, risk_free_policy_available=True,
                            starting_from_finite_network=False, want_unit_spread=True)
    assert any(r.mechanism == "action anchor" and "not needed" in r.setting for r in b.recommendations)
    assert any(r.mechanism == "prior" and "risk-free" in r.setting for r in b.recommendations)
    assert any(r.mechanism == "integrand" and "power integrand" in r.setting for r in b.recommendations)
    c = advise_regularizers(spec, Stage(m=1.0, eta=0.1, kappa=1.0), features=FeatureConstants.tanh_unit(1.0, 2.0), lambda_H=1.0)
    assert c.curvature is not None and any(r.mechanism == "joint effect" for r in c.recommendations)
    assert any(r.mechanism == "prior" and "centering" in r.setting for r in c.recommendations)
    txt = c.explain(); assert "Theorem 4.3" in txt
    p = plan(spec, Requirement(T=19, alpha=0.05), sup_U=5.4, stage=Stage(m=1.0, kappa=1.0),
             regularizers=dict(incumbent_available=True, prior_sigma=0.5))
    assert p.regularizers is not None and "Regularizers" in p.explain()


def test_effects_readings():
    spec = BudgetSpec(budgets=[1.0], beta=0.95, ubar=1.0)
    cand = dict(q_beta=0.06, U=2.85, q_se=0.002, U_se=0.02)
    prior = dict(q_beta=0.06, U=2.85, q_se=0.002, U_se=0.02)
    e = regularizer_effects(spec, Stage(m=0.03, eta=0.1, kappa=100.0), cand, action_entropy=93.0, reference_risk=0.011,
                            parameter_distance=0.003, dispersion_final=0.37, dispersion_prior=0.37, dispersion_anchored=True,
                            prior_policy=prior, path=[dict(kappa=1, q_beta=0.3, U=5.0), dict(kappa=10, q_beta=0.15, U=4.0), dict(kappa=100, q_beta=0.06, U=2.85)],
                            A_bar=0.5, H_bar=0.01, centered=False)
    assert e.action["informative"] is False and any("uninformative" in f for f in e.flags)
    assert e.inheritance["at_prior"] and e.parameter["dispersion_escape"] is False
    assert e.bias["R"] == 0.1 * 0.5 + 100.0 * 0.01 and e.joint is not None and len(e.path) == 3
    # mean-only prior: dispersion escape
    e2 = regularizer_effects(spec, Stage(m=0.03, kappa=100.0), dict(q_beta=0.34, U=4.0), parameter_distance=0.002,
                             dispersion_final=7.5, dispersion_prior=0.37, dispersion_anchored=False)
    assert e2.parameter["dispersion_escape"] and any("dispersion escape" in f for f in e2.flags)
    # informative action bound
    e3 = regularizer_effects(spec, Stage(m=0.03, eta=1.0), dict(q_beta=0.012, U=2.8), action_entropy=0.01, reference_risk=0.011)
    assert e3.action["informative"] and e3.action["bound"] >= 0.012
    assert "Regularizer effects" in e.explain()
