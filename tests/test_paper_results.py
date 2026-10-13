"""Checks of the revision-62 additions against the paper's numbers and hand computations (numpy only)."""
import math
import numpy as np

from bpg import BudgetSpec, Requirement, Stage, cite, plan, evaluate, evaluation_from_episodes, RunningMaxCost, BudgetWrapper
from bpg import certificates as C
from bpg import convergence as K
from bpg import validation as V
from bpg.frontier import price_sequence_check
from bpg.paper import RESULTS, results_table
from bpg.regularizers import regularizer_effects, advise_regularizers
from bpg.roles import RegularizerMix, stationary_spread, kappa1_for_spread, adam_stationary_spread
from bpg.examples.plateau import PlateauModel, simulate_particles, feature_bounds, PAPER_ROLES_TABLE
from bpg.examples.wear import WearModel, IncumbentRule, make_wear_env


def test_registry():
    assert cite("fixed_price", "ii").startswith("Proposition 3.3(ii)") and "price-risk law" in cite("fixed_price", "ii")
    assert cite("flow").startswith("Theorem 4.3") and cite("finite_training").startswith("Proposition 4.7")
    assert cite("action_risk").startswith("Proposition 3.4") and cite("learning_error").startswith("Corollary 4.4")
    assert "supplementary" in cite("near_exhaustion")
    assert results_table().count("\n") == len(RESULTS) + 1


def test_kernel_constants_match_softmax_closed_forms():
    for fc in (K.FeatureConstants(1.0, 0.3, 2.0), K.FeatureConstants.tanh_unit(2.0, 3.0), K.FeatureConstants(0.5, 1.7, 0.2)):
        for beta in (0.5, 0.9, 0.99):
            d = 1 - beta
            c = K.curvature_constants(fc, beta, m=1.3, ubar=0.7)
            c_psi = fc.L2 / d ** 2 + fc.L1 ** 2 / d ** 3
            C_A = fc.L0 * (1 + 3 * beta) * fc.L2 / d ** 2 + ((1 + 2 * fc.L0) / d + 2 * beta * fc.L0 * (3 - beta) / d ** 3) * fc.L1 ** 2
            assert abs(c["c_psi"] - c_psi) < 1e-9 * c_psi and abs(c["C_m"] - 2.0 * c_psi) < 1e-9 * c_psi
            assert abs(c["C_A"] - C_A) < 1e-9 * C_A
            assert c == K.curvature_constants(fc.kernel(), beta, 1.3, 0.7)


def test_plateau_model_reproduces_the_paper():
    m = PlateauModel()
    st = m.stats(0.0)
    assert abs(st["V_m"] - (-2.2533)) < 5e-4 and abs(st["q_beta"] - 0.1) < 1e-12 and abs(st["identity_residual"]) < 1e-9
    fb = feature_bounds()
    c = K.curvature_constants(K.FeatureConstants(**fb), beta=0.9, m=1.0, ubar=1.0)
    assert abs(c["C_m"] - 1940) < 1.0 and abs(K.certified_kappa(1.0, c["C_m"]) - 2037) < 1.0
    assert m.dV(0.0) > 0 and m.stats(1.0)["V_m"] > st["V_m"]                     # tilting toward high rates improves V_m


def test_roles_on_the_solvable_flow():
    m = PlateauModel(n_x=60, n_s=60)
    pot = simulate_particles(m, 0.0, 1.0, n=500, seed=1)
    ent = simulate_particles(m, 0.1, 0.0, n=500, seed=1)
    mix = simulate_particles(m, 0.1, 10.0, n=500, seed=1)
    plateau = m.stats(0.0)["V_m"]
    assert pot.spread[-1] < 1e-3 and abs(pot.V_m[-1] - plateau) < 1e-9             # a potential alone collapses, value never moves
    assert ent.V_m[-1] > plateau + 0.2                                               # noise escapes the plateau
    assert abs(mix.spread[-1] - stationary_spread(0.1, 10.0)) < 0.02                 # spread sqrt(k1/(k1+k2)) = 0.0995
    for name, k1, k2, V, spread, pred in PAPER_ROLES_TABLE:
        assert abs(stationary_spread(k1, k2) - pred) < 2e-3 or k2 == 0


def test_regularizer_mix():
    mix = RegularizerMix(kappa1=0.101, kappa2=10.0)
    assert abs(mix.stationary_spread() - 0.1) < 1e-3 and abs(mix.curvature - 10.101) < 1e-12
    assert RegularizerMix(kappa2=1.0).collapses and not mix.collapses
    for s, k1 in ((0.01, 0.0010), (0.03, 0.0090), (0.1, 0.1010), (0.3, 0.9890)):   # the paper's protocol values
        assert abs(kappa1_for_spread(s, 10.0) - k1) < 5e-4
    assert abs(adam_stationary_spread(0.1) - 0.1 * math.sqrt(2)) < 1e-12
    assert abs(mix.noise_std_per_step(3e-4) - math.sqrt(2 * 0.101 * 3e-4)) < 1e-15
    assert "Online Remark F.2" in RegularizerMix(kappa2=1.0).explain()


def test_new_certificates():
    b = C.matched_risk_signed(U_bar=3.0, I_bar=0.2, R_bar=0.1, delta=0.05, m=1.0, beta=0.9, q_hat=0.01, q_bar=0.03)
    assert abs(b.value - (3.0 - 0.2 - 0.1 - 0.05 + 10 * (0.01 - 0.03))) < 1e-12 and b.inputs["risk_price_term"] < 0
    nu = C.conservative_multiplier(alpha=0.05, sup_U=5.0, U0=1.0, q0=0.01)
    assert abs(nu.value - 4.0 / 0.04) < 1e-9 and C.conservative_multiplier(0.01, 5.0, q0=0.02).value == math.inf
    fl = C.risk_floor(p_T_reference=0.1, L0=0.5, T=3)
    assert abs(fl.value - math.exp(-4.0) * 0.1) < 1e-15
    assert not C.prescribed_level_attainable(0.001, fl.value)["attainable"]
    # Corollary 4.4 slack: the larger of the two forms
    d = C.learning_slack(m=1.0, ubar=1.0, beta=0.95, alpha=0.05, T=0, q_bar=0.0, U_bar=2.0, I_bar=0.1, R_bar=0.2, sup_U=5.0)
    d1 = 1.0 * 0.05 / 0.05 - (5.0 - 2.0 + 0.1 + 0.2); d2 = 2.0 * 0.05 / 0.05 - (1.0 / 0.05 - 2.0 + 0.1 + 0.2)
    assert abs(d - max(d1, d2)) < 1e-12
    lr = K.learning_risk_bound(0.5, m=1.0, beta=0.95, ubar=1.0, q_bar=0.0, U_bar=2.0, I_bar=0.1, R_bar=0.2, E0=3.0, Lambda=2.0, sup_U=5.0)
    lf = C.learning_form_risk(1.0, 0.95, 1.0, 0.0, 2.0, 0.1, 0.2, 3.0 * math.exp(-2.0), sup_U=5.0)
    assert abs(lr.value - lf.value) < 1e-12 and lr.source.startswith("Corollary 4.4")
    ft = K.finite_training_transfer(0.3, Lambda=-1.0, ell=2.0, m=1.0, ubar=1.0, beta=0.9, B1=0.5)
    assert abs(ft["W2"] - 0.3 * math.exp(2.0)) < 1e-12 and abs(ft["risk"] - 0.5 / 0.2 * ft["W2"]) < 1e-12
    assert abs(ft["value"] - 2.0 * 0.5 / (2 * 0.01) * ft["W2"]) < 1e-9


def test_lump_sanction_certificate():
    # one-week toy: ubar = 0, utility = -cost, nu = F; V = -E[cost] - F p  =>  p <= -V/F
    b = V.lump_sanction_risk(V_H=-1.5, H=52, nu=50.0, ubar=0.0, beta=1 - 1e-6, T=51)
    assert abs(b.value - 1.5 / (50.0 * (1 - 1e-6) ** 51)) < 1e-9
    bh = V.lump_sanction_risk(V_H=-1.5, H=52, nu=50.0, ubar=0.0, beta=1 - 1e-6, T=51, N=20000, zeta=0.05 / 48, u_minus=0.1)
    S = (1 - (1 - 1e-6) ** 52) / 1e-6
    lo = -1.5 - (0.1 * S + 50.0) * math.sqrt(math.log(48 / 0.05) / 40000)
    assert abs(bh.inputs["V_used"] - lo) < 1e-9 and bh.value > b.value


def test_price_sequence_on_the_application():
    F = [25, 50, 100, 200, 400]
    ppo = price_sequence_check(F, [0.79, 1.51, 3.77, 7.55, 12.07])         # the paper's PPO shortfalls
    sac = price_sequence_check(F, [0.93, 1.59, 1.35, 3.43, 2.78])          # and SAC's
    assert not ppo.shrinking and sac.shrinking and ppo.loglog_slope > 0.9
    assert "not the remedy" in ppo.explanation and "Online Proposition E.4" in sac.explanation
    empty = price_sequence_check([0.5, 2.0, 8.0], [0.0, 0.0, 0.0])
    assert not empty.informative and "says nothing" in empty.explanation


def test_running_max_cost_and_signed_use():
    class Walk:
        def __init__(self, steps): self.steps = steps
        def reset(self, *, seed=None, options=None): self.t = 0; return np.zeros(1, np.float32), {}
        def step(self, a):
            d = self.steps[self.t]; self.t += 1
            return np.zeros(1, np.float32), 0.0, False, False, dict(d=d)
    cost = RunningMaxCost(lambda o, a, o2, r, i: [i["d"]])
    spec = BudgetSpec(budgets=[1.0], beta=0.99, ubar=0.0, horizon=10)
    w = BudgetWrapper(Walk([0.6, -0.5, 0.6, 0.3]), cost, spec, Stage(m=1.0))
    w.reset()
    infos = [w.step(0)[4] for _ in range(3)]
    assert [round(float(i["cost"][0]), 10) for i in infos] == [0.6, 0.0, 0.1]            # running max 0.6, 0.6, 0.7
    _, _, term, _, info = w.step(0)                                                     # cumulative 1.0, running max 1.0: equality is not failure
    assert not info["failure"]
    cost.reset(); assert cost.M[0] == 0.0


def test_evaluation_from_episodes_matches_evaluate():
    model = WearModel(); spec = model.spec(); stage = Stage(m=1.0)
    rule = IncumbentRule(seed=3)
    ev = evaluate(lambda: make_wear_env(stage, model), rule, 400, spec=spec, stage=stage, seed=5, T=19)
    ev2 = evaluation_from_episodes(ev.failed, ev.tau, ev.U_episodes, ev.I_episodes, spec=spec, stage=stage, T=19,
                                   geometric_stop=True, min_reserve=ev.min_reserve)
    assert ev2.q_beta == ev.q_beta and ev2.failures == ev.failures and abs(ev2.V_m - ev.V_m) < 1e-12 and ev2.event == "q_beta"


def test_plan_conditions_and_multiplier_stage():
    spec = BudgetSpec(budgets=[31.7], beta=1 - 1e-6, ubar=0.0, u_minus=4.78, horizon=52)
    req = Requirement(T=51, alpha=0.05, zeta=0.05, family=48)
    st = Stage(m=0.0, penalty="multiplier", multiplier=50.0, clip_utility=False)
    p = plan(spec, req, sup_U=0.0, comparator=dict(U0=-5.27, I0=0.0, q0=0.0131), stage=st, structured_solution_available=True)
    assert p.price_bound.name.startswith("flat penalty") and p.another_method_preferable
    txt = p.explain()
    assert "flat failure penalty nu = 50" in txt and "charged once" in txt and "[APPLIES] the problem can be solved by a dynamic program" in txt
    assert "Section 1.2" in txt
    p2 = plan(BudgetSpec(budgets=[22.0], beta=0.999, ubar=100.0, u_minus=100.0, horizon=1000), Requirement(T=999, alpha=0.05),
              stage=Stage(m=2.0, clip_utility=False), task_stakes=100.0)
    assert any("ten times" in w for w in p2.warnings)
    p3 = plan(BudgetSpec(budgets=[1.0], beta=0.95, ubar=1.0), Requirement(T=19, alpha=0.0), sup_U=5.4)
    assert p3.conditions[0].applies


def test_effects_noise_and_inheritance():
    spec = BudgetSpec(budgets=[1.0], beta=0.95, ubar=1.0)
    st = Stage(m=0.03, kappa=10.0)
    e = regularizer_effects(spec, st, dict(q_beta=0.14, U=3.08, q_se=0.01, U_se=0.05, V_m=2.0),
                            prior_policy=dict(q_beta=0.0103, U=2.73, q_se=0.001, U_se=0.01, V_m=2.5),
                            noise_kappa1=0.101, anchor_kappa2=10.0, spread=0.13, dispersion_final=0.10)
    assert abs(e.noise["nominal_gradient_spread"] - 0.1) < 1e-3 and any("comparable to the action dispersion" in f for f in e.flags)
    assert any("clipped value is below the prior" in f for f in e.flags)
    e2 = regularizer_effects(spec, st, dict(q_beta=0.506, U=5.24), noise_kappa1=0.101, spread=1.5)
    assert any("abandons the prior" in f for f in e2.flags)
    a = advise_regularizers(spec, st, stalls_where_gradient_vanishes=True)
    assert "noise" in a.mechanisms() and "Online Remark F.2" in a.explain()
