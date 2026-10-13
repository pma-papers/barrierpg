import numpy as np
from bpg import BudgetSpec, Requirement, Stage, evaluate, interpret, plan
from bpg.examples.wear import EXACT_FRONTIER, INCUMBENT, IncumbentRule, WearModel, full_rate_policy, idle_policy, make_wear_env
from bpg.evaluate import stopped_action_relative_entropy
from bpg.convergence import FeatureConstants


def test_idle_and_full_rate():
    model = WearModel(); spec = model.spec(); stage = Stage(m=1.0)
    ev = evaluate(lambda: make_wear_env(stage, model), idle_policy, 500, spec=spec, stage=stage, seed=1)
    assert ev.failures_any == 0 and ev.U == 0.0 and ev.q_beta == 0.0 and ev.geometric_stop
    ev = evaluate(lambda: make_wear_env(stage, model), full_rate_policy, 4000, spec=spec, stage=stage, seed=2)
    # the exact unpriced optimum runs at full rate until breakdown: q = 0.592, U = 5.43 (grid values)
    assert abs(ev.q_beta - EXACT_FRONTIER[0.0][0]) < 0.04 and abs(ev.U - EXACT_FRONTIER[0.0][1]) < 0.3
    assert abs(ev.V_m - (ev.U - ev.I_m - 1.0 * ev.q_beta / 0.05)) < 1e-9


def test_incumbent_matches_paper():
    model = WearModel(); spec = model.spec(); stage = Stage(m=0.03)
    rule = IncumbentRule(seed=5)
    ev = evaluate(lambda: make_wear_env(stage, model), rule, 20000, spec=spec, stage=stage, seed=3)
    assert abs(ev.q_beta - INCUMBENT["q_beta"]) < 0.004 and abs(ev.U - INCUMBENT["U"]) < 0.1
    assert ev.event == "q_beta" and ev.p_T != ev.p_T          # censored under the geometric clock
    val = ev.validate(0.05, 0.05, family=120)
    assert val.passed and val.event == "q_beta"
    # stopped action relative entropy of the rule to itself is zero
    A = stopped_action_relative_entropy(lambda: make_wear_env(stage, model), lambda o: (rule(o), 0.0), 50, beta=0.95, geometric_stop=True)
    assert A["A"] == 0.0


def test_robustness_and_near():
    model = WearModel(); spec = model.spec(); stage = Stage(m=1.0)
    rule = IncumbentRule(seed=7)
    nominal = evaluate(lambda: make_wear_env(stage, model), rule, 3000, spec=spec, stage=stage, seed=4)
    reduced = evaluate(lambda: make_wear_env(stage, model, true_budget_scale=0.8), rule, 3000, spec=spec, stage=stage, seed=4)
    assert reduced.q_beta >= nominal.q_beta
    assert 0.0 <= nominal.near[0.05] <= 1.0 and nominal.near[0.2] >= nominal.near[0.05]


def test_plan_and_report_run():
    model = WearModel(); spec = model.spec()
    req = Requirement(T=19, alpha=0.05, zeta=0.05)
    p = plan(spec, req, sup_U=5.44, expected_failure_rate=0.01, validation_H=300, validation_N=20000,
             features=FeatureConstants.tanh_unit(1.0, 2.0), lambda_H=1.0, allowance_error_known=False)
    txt = p.explain()
    assert "Price" in txt and p.validation_episodes > 0 and p.curvature is not None
    stage = p.stage
    rule = IncumbentRule(seed=9)
    ev = evaluate(lambda: make_wear_env(stage, model), rule, 2000, spec=spec, stage=stage, seed=11, T=19)
    rep = interpret(spec, req, stage, ev, sup_U=5.44, delta=None, lower_barrier=None)
    t = rep.text()
    assert "Verdict" in t and "Not established" in t and rep.best_bound() is not None
    assert rep.validation is not None and rep.validation.event == "q_beta"
    # without the geometric clock p_T is estimated directly
    env2 = lambda: make_wear_env(stage, model, geometric_stop=False)
    ev2 = evaluate(env2, rule, 300, spec=spec, stage=stage, seed=12, T=19, max_steps=400)
    assert ev2.event == "p_T" and 0.0 <= ev2.p_T <= 1.0
