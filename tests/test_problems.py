"""The tutorial problems: budgets, failure events and observations (the retailer needs numpy only; the lander and
the robot are skipped when Box2D or MuJoCo is missing; navigation needs Safety-Gymnasium in its own environment)."""
import numpy as np

from bpg import Requirement, Stage, evaluate, interpret


def test_retailer_cash_is_the_budget():
    from bpg.problems.inventory import Retailer, InventoryCore, make_inventory_env, BaseStockRule
    r = Retailer()
    stage = Stage(m=0.0, penalty="multiplier", multiplier=5000.0, clip_utility=False)
    env = make_inventory_env(stage, r)
    # ordering the most every week exhausts the cash: the wrapper fails in the first week with negative cash
    obs, _ = env.reset(seed=0)
    core_cash = r.cash
    for t in range(r.weeks):
        obs, reward, term, trunc, info = env.step(np.ones(r.K, dtype=np.float32))
        core_cash = info["cash"]
        if term or trunc:
            break
    assert info["failure"] and core_cash < 0 and reward == -5000.0
    # the observation carries the cash and the wrapper's reserve; the rule reads the cash
    core = InventoryCore(r)
    o, _ = core.reset(seed=1)
    assert o.shape == (core.n_obs,) and abs(o[-1] - 1.0) < 1e-9
    # the incumbent rule rarely goes insolvent and earns a profit
    ev = evaluate(lambda: make_inventory_env(stage, r), BaseStockRule(r, 2.0, 300.0).deterministic, 300, spec=r.spec(),
                  stage=stage, T=r.weeks - 1, seed=3)
    assert ev.p_T < 0.03 and ev.U > 5000


def test_retailer_rule_with_noise_is_riskier():
    from bpg.problems.inventory import Retailer, make_inventory_env, BaseStockRule
    r = Retailer()
    stage = Stage(m=0.0, penalty="multiplier", multiplier=0.0, clip_utility=False)
    exact = evaluate(lambda: make_inventory_env(stage, r), BaseStockRule(r, 2.0, 300.0).deterministic, 400, spec=r.spec(),
                     stage=stage, T=r.weeks - 1, seed=5)
    noisy = evaluate(lambda: make_inventory_env(stage, r), BaseStockRule(r, 2.0, 300.0, std=0.1, seed=1), 400, spec=r.spec(),
                     stage=stage, T=r.weeks - 1, seed=5)
    assert noisy.p_T > exact.p_T


def test_observe_time_appends_the_elapsed_fraction():
    from bpg import BudgetSpec, BudgetWrapper
    from bpg.problems.inventory import InventoryCore, cash_outflow
    from bpg import RunningMaxCost
    spec = BudgetSpec(budgets=[2500.0], beta=0.999, ubar=1.0, horizon=52)
    env = BudgetWrapper(InventoryCore(), RunningMaxCost(cash_outflow), spec, Stage(m=0.0), observe_time=True)
    o, _ = env.reset(seed=0)
    assert abs(o[-1]) < 1e-9 and abs(o[-2] - 1.0) < 1e-9
    for _ in range(13):
        o, *_ = env.step(np.zeros(3, dtype=np.float32) - 1.0)
    assert abs(o[-1] - 0.25) < 1e-9
    try:
        BudgetWrapper(InventoryCore(), RunningMaxCost(cash_outflow), BudgetSpec(budgets=[1.0], beta=0.9, ubar=1.0), Stage(m=0.0),
                      observe_time=True)
        raise AssertionError("observe_time without a horizon must fail")
    except ValueError:
        pass


def test_report_summary_is_plain():
    from bpg.problems.inventory import Retailer, make_inventory_env, BaseStockRule
    r = Retailer()
    stage = Stage(m=0.0, penalty="multiplier", multiplier=5000.0, clip_utility=False)
    req = Requirement(T=r.weeks - 1, alpha=0.02, zeta=0.05, family=4)
    ev = evaluate(lambda: make_inventory_env(stage, r), BaseStockRule(r, 2.0, 300.0).deterministic, 300, spec=r.spec(),
                  stage=stage, T=r.weeks - 1, seed=7)
    text = interpret(r.spec(), req, stage, ev, candidate_name="the rule").summary(resource="cash", period="weeks")
    assert text.startswith("The rule was run on 300 fresh episodes") and "within 52 weeks" in text
    assert "shared among 4 candidates" in text and "What this test does not show" in text
    for word in ("q_beta", "beta^", "Proposition", "theorem", "population flow"):
        assert word not in text


def test_lander_and_robot_budgets():
    try:
        import Box2D  # noqa: F401
        from bpg.problems.lander import make_lander_env, fuel_use, Tank
        env = make_lander_env(Stage(m=0.5, clip_utility=False), Tank(capacity=1.0))
        o, _ = env.reset(seed=0)
        for t in range(200):
            o, rwd, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))     # full main engine: 0.30 per frame
            if term or trunc:
                break
        assert info["failure"] and t == 3 and abs(fuel_use([1.0, 0.0]) - 0.30) < 1e-12
    except ImportError:
        print("Box2D missing; lander skipped")
    try:
        import mujoco  # noqa: F401
        from bpg.problems.energy import make_robot_env, Battery
        env = make_robot_env(Stage(m=1.0, clip_utility=False), Battery(charge=30.0, efficiency_sd=0.0))
        o, _ = env.reset(seed=0)
        for t in range(100):
            o, rwd, term, trunc, info = env.step(np.ones(6, dtype=np.float32))      # 0.5 + 6 per step: dry after 5 steps
            if term or trunc:
                break
        assert info["failure"] and t == 4 and o.shape == (19,)
    except ImportError:
        print("MuJoCo missing; robot skipped")


def test_zero_reserve_is_finite_under_the_unclipped_form():
    # whole-unit costs can bring the reserve to exactly zero, where the barrier is infinite; the unclipped form must
    # charge nothing there when only the failure is priced, and a finite amount with an interior barrier
    from bpg import BudgetSpec, BudgetWrapper
    from bpg.examples.wear import WearCore
    spec = BudgetSpec(budgets=[3.0], beta=0.99, ubar=1.0, horizon=10)
    for stage, expect in ((Stage(m=0.5, clip_utility=False), None), (Stage(m=0.5, theta=0.2, clip_utility=False), -0.5)):
        env = BudgetWrapper(WearCore(), lambda o, a, o2, r, i: [1.0], spec, stage)
        env.reset(seed=0)
        for _ in range(3):
            o, r, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
        assert not info["failure"] and np.isfinite(r) and info["reserve"][0] == 0.0
        if expect is None:
            assert info["interior_loss"] == 0.0
        else:
            assert abs(r - expect) < 1e-9


def test_half_cheetah_without_settotalmass_matches_the_stock_environment():
    try:
        import mujoco  # noqa: F401
        import gymnasium as gym
    except ImportError:
        print("MuJoCo missing; skipped"); return
    import os
    import tempfile
    from bpg.problems.energy import half_cheetah
    # the stock model warns by design, and MuJoCo writes the warning to MUJOCO_LOG.TXT in the working directory:
    # build it in a temporary one so that the test leaves nothing behind
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            a = gym.make("HalfCheetah-v5")
        finally:
            os.chdir(here)
    b = half_cheetah()
    assert np.array_equal(a.unwrapped.model.body_mass, b.unwrapped.model.body_mass)
    assert np.array_equal(a.unwrapped.model.body_inertia, b.unwrapped.model.body_inertia)
    oa, _ = a.reset(seed=1); ob, _ = b.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(200):
        act = rng.uniform(-1, 1, 6)
        oa, ra, *_ = a.step(act); ob, rb, *_ = b.step(act)
        assert np.array_equal(oa, ob) and ra == rb
