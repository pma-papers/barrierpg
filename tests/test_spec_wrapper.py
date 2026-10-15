import math
import numpy as np
from barrierpg.spec import BudgetSpec, Requirement, Stage, discounted_total, failure_tail, price_from_multiplier, price_from_multiplier_physical
from barrierpg.barrier import LogBarrier
from barrierpg.wrapper import BudgetWrapper


class Counter:
    """Deterministic test env: obs = (t,), reward = 1, step cost given by a sequence."""
    def __init__(self, costs):
        self.costs = list(costs)
    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.array([0.0], dtype=np.float32), {}
    def step(self, a):
        g = self.costs[self.t] if self.t < len(self.costs) else 0.0
        self.t += 1
        return np.array([float(self.t)], dtype=np.float32), 1.0, False, False, dict(g=g)


def test_spec_and_conversions():
    s = BudgetSpec(budgets=[1.0, 2.0], beta=0.9, ubar=1.0)
    assert s.K == 2 and s.gammas == (1.0, 1.0)
    assert abs(discounted_total(0.5, 0.9) - 5.0) < 1e-12
    assert abs(price_from_multiplier(5.0, 0.9) - 0.5) < 1e-12
    assert abs(failure_tail(1.0, 0.5, 0, horizon=3) - (1 - 0.125) / 0.5) < 1e-12
    assert failure_tail(1.0, 0.5, 3, horizon=3) == 0.0
    nu = failure_tail(0.5, 0.999, 300, 1000)
    assert abs(price_from_multiplier_physical(nu, 0.999, 1000, 300) - 0.5) < 1e-9
    r = Requirement(T=19, alpha=0.05, zeta=0.05, family=120)
    assert abs(r.error_per_statement - 0.05 / 120) < 1e-15
    try:
        Stage(m=-1.0); raise AssertionError
    except ValueError:
        pass


def test_log_barrier():
    B = LogBarrier(0.5, [4.0])
    assert abs(B([3.0]) - 0.5 * math.log1p(4 / 3)) < 1e-12 and B([0.0]) == math.inf and B([-1.0]) == math.inf
    assert LogBarrier(0.0, [4.0])([0.0]) == math.inf and LogBarrier(0.0, [4.0])([0.5]) == 0.0
    assert abs(B.lower_level([0.5]) - 0.5 * math.log1p(8.0)) < 1e-12


def test_wrapper_absorbing_and_identity():
    spec = BudgetSpec(budgets=[1.0], beta=0.9, ubar=1.0)
    stage = Stage(m=2.0, theta=0.5)
    w = BudgetWrapper(Counter([0.4, 0.4, 0.4]), lambda o, a, o2, r, i: [i["g"]], spec, stage, observe="fraction")
    o, _ = w.reset(seed=0)
    assert o[-1] == 1.0
    B = LogBarrier(0.5, [1.0])
    o, r, term, trunc, info = w.step(0)
    assert abs(r - max(-2.0, 1.0 - B([0.6]))) < 1e-12 and not term and abs(o[-1] - 0.6) < 1e-6
    assert abs(info["interior_loss"] - min(B([0.6]), 1.0 + 2.0)) < 1e-12
    o, r, term, trunc, info = w.step(0)          # z = 0.2
    o, r, term, trunc, info = w.step(0)          # z = -0.2: failure at tau = 2
    assert term and info["failure"] and abs(r + 2.0 / 0.1) < 1e-9
    ep = info["budget_episode"]
    assert ep["tau"] == 2 and ep["failed"] and ep["length"] == 3
    # V_m identity on this path: sum beta^t (u - interior) for t < tau, minus beta^tau m/(1-beta)
    expect = sum(0.9 ** t * (1.0 - min(B([z]), 3.0)) for t, z in enumerate([0.6, 0.2])) - 0.81 * 20.0
    assert abs(ep["clipped_return"] - expect) < 1e-9
    assert abs(ep["U"] - (1 + 0.9)) < 1e-12


def test_wrapper_modes():
    spec = BudgetSpec(budgets=[1.0], beta=0.9, ubar=1.0, horizon=5)
    w = BudgetWrapper(Counter([2.0]), lambda o, a, o2, r, i: [i["g"]], spec, Stage(m=1.0))
    w.reset(); _, r, term, _, info = w.step(0)
    assert term and abs(r + (1 - 0.9 ** 5) / 0.1) < 1e-12          # physical-horizon tail at t = 0
    w = BudgetWrapper(Counter([2.0]), lambda o, a, o2, r, i: [i["g"]], spec, Stage(m=1.0, penalty="multiplier", multiplier=7.0))
    w.reset(); _, r, term, _, info = w.step(0)
    assert abs(r + 7.0) < 1e-12
    w.set_multiplier(3.0); w.reset(); _, r, _, _, _ = w.step(0); assert abs(r + 3.0) < 1e-12
    # equality is not failure
    w = BudgetWrapper(Counter([1.0]), lambda o, a, o2, r, i: [i["g"]], BudgetSpec(budgets=[1.0], beta=0.9, ubar=1.0), Stage(m=1.0))
    w.reset(); _, r, term, _, info = w.step(0)
    assert not info["failure"] and r == -1.0        # z = 0: barrier infinite, clipped to -m (theta = 0 gives B(0) = inf)
    # horizon termination without failure
    w = BudgetWrapper(Counter([0.0] * 10), lambda o, a, o2, r, i: [0.0], spec, Stage(m=1.0))
    w.reset()
    for k in range(5):
        _, _, term, _, info = w.step(0)
    assert term and not info["failure"] and info["budget_episode"]["tau"] is None
    # cost discount and true-budget scale
    spec2 = BudgetSpec(budgets=[1.0], beta=0.9, ubar=1.0, gammas=[0.5])
    w = BudgetWrapper(Counter([0.5, 0.5]), lambda o, a, o2, r, i: [i["g"]], spec2, Stage(m=1.0), true_budget_scale=0.9)
    w.reset(); _, _, term, _, info = w.step(0)
    assert not term and abs(info["reserve"][0] - 1.0) < 1e-12          # (1 - 0.5)/0.5 = 1 nominal
    _, _, term, _, info = w.step(0)                                        # cum = 0.5 + 0.25 = 0.75 <= 0.9: survives
    assert not term
    w = BudgetWrapper(Counter([0.5, 0.5]), lambda o, a, o2, r, i: [i["g"]], spec2, Stage(m=1.0), true_budget_scale=0.7)
    w.reset(); w.step(0); _, _, term, _, info = w.step(0)
    assert term and info["failure"]                                        # 0.75 > 0.7 while the nominal reserve is 1.0
