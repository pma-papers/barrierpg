import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from barrierpg import (
    BudgetSpec, BudgetWrapper, InverseBarrier, LogBarrier, ReserveHinge, Stage,
    Requirement, categorical_action_divergence, cpo_step, cpo_update, evaluate, interpret, omnisafe_cpo,
    parameter_law_divergence, parameter_potential, regularized_utility,
    select_scenario_policy, upper_cvar,
)


def test_dense_cpo_matches_direct_qp():
    step = cpo_step([2, 1], [1, 0], np.eye(2), -0.6, 0.5)
    assert step.status == "active"
    np.testing.assert_allclose(step.step, [0.6, 0.8])
    rng = np.random.default_rng(94)
    for _ in range(24):
        reward, cost = rng.normal(size=(2, 3))
        matrix = rng.normal(size=(3, 3))
        metric = matrix.T @ matrix + np.eye(3)
        kl = 0.4
        gap = rng.uniform(-0.95, 0.95) * np.sqrt(2 * kl * (cost @ np.linalg.solve(metric, cost)))
        initial = -gap * np.linalg.solve(metric, cost) / (cost @ np.linalg.solve(metric, cost))
        reference = minimize(
            lambda w: -reward @ w, initial, jac=lambda w: -reward, method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": lambda w: -gap - cost @ w, "jac": lambda w: -cost},
                {"type": "ineq", "fun": lambda w: kl - w @ metric @ w / 2, "jac": lambda w: -metric @ w},
            ],
            options={"ftol": 1e-11},
        )
        assert reference.success or reference.status == 8, reference.message
        assert gap + cost @ reference.x <= 1e-8
        assert reference.x @ metric @ reference.x / 2 <= kl + 1e-8
        proposal = cpo_step(reward, cost, metric, gap, kl)
        assert proposal.predicted_cost_gap < 1e-12
        assert proposal.step @ metric @ proposal.step / 2 <= kl + 1e-12
        assert reward @ proposal.step == pytest.approx(-reference.fun, abs=1e-8)


def test_cpo_recovery_and_rechecks_nonlinear_cost():
    metric = np.eye(2)
    assert cpo_step([2, 1], [0, 0], metric, 1, 0.5).status == "infeasible"
    with pytest.raises(ValueError, match="positive definite"):
        cpo_step([2, 1], [1, 0], np.zeros((2, 2)), 1, 0.5)
    recovery = cpo_update([2, 1], [1, 0], metric, 2, 0.5,
                          lambda w: (w[0], 2 + w[0], 0.49 * (w @ w)))
    assert recovery.status == "recovery"
    np.testing.assert_allclose(recovery.step, [-1, 0])

    update = cpo_update([2, 1], [1, 0], metric, -0.6, 0.5,
                        lambda w: (2*w[0] + w[1], -0.6 + w[0] + w[1]**2, w @ w / 2))
    assert update.status == "accepted"
    np.testing.assert_allclose(update.step, [0.3, 0.4])
    rejected = cpo_update([2, 1], [1, 0], metric, -0.6, 0.5,
                          lambda w: (100, 0.01, 0))
    assert rejected.status == "rejected"
    np.testing.assert_array_equal(rejected.step, [0, 0])


def test_omnisafe_adapter_preserves_caller_settings(monkeypatch):
    def agent(method, env, *, custom_cfgs):
        return method, env, custom_cfgs

    monkeypatch.setitem(sys.modules, "omnisafe", SimpleNamespace(Agent=agent))
    cfg = {"algo_cfgs": {"gamma": 0.99, "cost_limit": 2}}
    assert omnisafe_cpo("Example-v0", cost_limit=4, custom_cfgs=cfg) == (
        "CPO", "Example-v0", {"algo_cfgs": {"gamma": 0.99, "cost_limit": 4.0}},
    )
    assert cfg["algo_cfgs"]["cost_limit"] == 2


def test_upper_cvar_matches_highs_epigraph():
    assert upper_cvar([0, 2, 5], 0.1, [0.8, 0.15, 0.05]) == pytest.approx(3.5)
    assert upper_cvar([10, -1], 0.5, [0, 1]) == -1
    rng = np.random.default_rng(18)
    for tail_mass in (0.013, 0.1, 0.5, 1):
        for _ in range(6):
            costs = rng.integers(-3, 10, size=8)
            weights = rng.dirichlet(np.ones(8))
            n = costs.size
            ref = linprog(
                np.r_[1.0, weights / tail_mass], A_ub=-np.c_[np.ones(n), np.eye(n)],
                b_ub=-costs, bounds=[(None, None)] + [(0, None)] * n, method="highs",
            )
            assert ref.success
            assert upper_cvar(costs, tail_mass, weights) == pytest.approx(ref.fun)


def test_scenario_policy_switches_by_requirement_and_objective():
    utility = np.array([1.0, 1.1])
    consumption = [[1, 1], [0, 2]]
    assert select_scenario_policy(utility, consumption, cost_limit=1, formulation="mean").index == 1
    assert select_scenario_policy(utility, consumption, cost_limit=1, formulation="cvar", tail_mass=0.5).index == 0
    assert select_scenario_policy(utility, consumption, cost_limit=0, formulation="chance", budget=1).index == 0
    assert select_scenario_policy(utility, consumption, cost_limit=0.9, formulation="pathwise").index is None
    adjusted = regularized_utility(utility, action_penalty=np.array([0.0, 0.3]), eta=1.0)
    assert select_scenario_policy(adjusted, consumption, cost_limit=2, formulation="cvar", tail_mass=0.5).index == 0


def test_reserve_shapes_at_activation_and_boundary():
    log = LogBarrier(0.3, [2, 4])
    inverse = InverseBarrier(0.3, [2, 4])
    linear = ReserveHinge(0.3, [2, 4], fraction=0.5)
    quadratic = ReserveHinge(0.3, [2, 4], fraction=0.5, power=2)
    for barrier in [log, inverse, linear, quadratic]:
        assert barrier([0.2, 4]) > barrier([1, 4])
        assert barrier.lower_level([0.2, 4]) == pytest.approx(barrier([math.inf, 4]))
    assert math.isinf(log([0, 4])) and math.isinf(inverse([0, 4]))
    assert linear([0, 4]) == pytest.approx(0.15)
    assert quadratic([0, 4]) == pytest.approx(0.075)


@pytest.mark.parametrize("cost, expected", [(0.75, 0.75), (1.0, 0.5), (1.01, -1.0)])
def test_wrapper_uses_finite_hinge_at_zero_without_changing_failure(cost, expected):
    class OneStep:
        def reset(self, *, seed=None, options=None):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 1.0, True, False, {}

    spec = BudgetSpec([1], beta=0.9, ubar=1, horizon=1)
    env = BudgetWrapper(OneStep(), lambda *_: [cost], spec, Stage(m=1),
                        barrier=ReserveHinge(1, [1], fraction=0.5))
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(expected)
    assert info["failure"] == (cost > 1)

    # Stage.theta is the default constructor's strength, so an explicit hinge
    # with default Stage settings must not be mistaken for failure-only.
    stage = Stage(m=1, clip_utility=False)
    evaluation = evaluate(
        lambda: BudgetWrapper(OneStep(), lambda *_: [cost], spec, stage,
                              barrier=ReserveHinge(1, [1], fraction=0.5)),
        lambda _: 0, 2, spec=spec, stage=stage, T=0,
    )
    report = interpret(spec, Requirement(T=0, alpha=0.1), stage, evaluation)
    assert evaluation.failure_only is False
    assert len(report.bounds) == 1


def test_generalized_convex_terms_and_torch_gradients():
    policy, prior = [0.75, 0.25], [0.5, 0.5]
    kl = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert categorical_action_divergence(policy, prior) == pytest.approx(kl)
    assert categorical_action_divergence([1, 0], [0, 1], form="hellinger") == pytest.approx(1)
    assert categorical_action_divergence(policy, prior, form="squared") == pytest.approx(0.0625)
    assert parameter_law_divergence(policy, prior, weight=2) == pytest.approx(kl + 0.5)
    assert parameter_law_divergence(policy, prior, weight=2, power=3) == pytest.approx(kl + 0.75)
    with pytest.raises(ValueError, match="support"):
        categorical_action_divergence(policy, [0, 1])

    torch = pytest.importorskip("torch")
    logit = torch.tensor(math.log(3), dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(logit)
    action = torch.stack((p, 1 - p))
    for form in ("kl", "hellinger", "squared"):
        div = categorical_action_divergence(action, prior, form=form)
        derivative, = torch.autograd.grad(div, logit, retain_graph=True)
        assert float(derivative) > 0
    div = parameter_law_divergence(action, prior, weight=2)
    derivative, = torch.autograd.grad(div, logit)
    assert float(derivative) == pytest.approx(0.1875 * (math.log(3) + 4))

    delta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    for form, opts, slope in (("quadratic", {}, 2), ("pseudo_huber", {"scale": 1}, 2 / math.sqrt(5)),
                              ("elastic_net", {"l1": 0.5}, 2.5)):
        derivative, = torch.autograd.grad(parameter_potential(delta, form=form, **opts), delta)
        assert float(derivative.item()) == pytest.approx(slope)
