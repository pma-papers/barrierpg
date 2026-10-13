"""Needs gymnasium + stable_baselines3 (+ Box2D for the lunar reproduction): run with numerics-v38/.venv."""
import numpy as np


def test_wrapper_reproduces_fuellander():
    import gymnasium as gym
    import sys
    from pathlib import Path
    from bpg import BudgetSpec, BudgetWrapper, Stage
    try:
        import Box2D  # noqa: F401
    except ImportError:
        print("Box2D not installed; skipped"); return
    root = Path(__file__).resolve().parents[2] / "barrier-policy-gradients" / "numerics-v39"
    if not (root / "lunar.py").exists():
        print("lunar.py not found; skipped"); return
    sys.path.insert(0, str(root))
    from lunar import FuelLander, fuel_use, H
    for seed, m, budget in ((1, 0.5, 22.0), (2, 0.0, 20.0), (4, 0.5, 5.0)):
        ref = FuelLander(budget, "failure", m, 0.999)
        spec = BudgetSpec(budgets=[budget], beta=0.999, ubar=100.0, u_minus=100.0, horizon=H)
        new = BudgetWrapper(gym.make("LunarLanderContinuous-v3"), lambda o, a, o2, r, i: [fuel_use(a)], spec, Stage(m=m, clip_utility=False))
        o1, _ = ref.reset(seed=seed); o2, _ = new.reset(seed=seed)
        rng = np.random.default_rng(seed)
        assert np.allclose(o1, o2)
        for t in range(1000):
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            o1, r1, d1, tr1, i1 = ref.step(a); o2, r2, d2, tr2, i2 = new.step(a)
            assert np.allclose(o1, o2) and abs(r1 - r2) < 1e-9 and d1 == d2 and i1["failure"] == i2["failure"], t
            if d1 or tr1:
                break


def test_ppo_smoke_with_callbacks():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from bpg import Stage, evaluate, sb3_policy
    from bpg.callbacks import BudgetMonitor, DualAscent, ParameterAnchor
    from bpg.examples.wear import WearModel, make_wear_env
    model = WearModel(); spec = model.spec()
    stage = Stage(m=1.0, penalty="multiplier")
    venv = DummyVecEnv([lambda: make_wear_env(stage, model) for _ in range(4)])
    sb3 = PPO("MlpPolicy", venv, seed=0, n_steps=64, batch_size=64, n_epochs=2, gamma=1.0, device="cpu", verbose=0, policy_kwargs=dict(net_arch=[16, 16]))
    anchor = ParameterAnchor(sb3, kappa_p=1.0).install()
    dual = DualAscent(alpha=0.05, step=1.0); mon = BudgetMonitor(spec.beta, stage.m, penalty="multiplier")
    sb3.learn(total_timesteps=64 * 4 * 6, callback=[dual, mon])
    assert len(dual.trace) == 6 and len(mon.trace) == 6 and anchor.distance() >= 0.0
    d = dual.diagnose(spec.beta); assert d.status in ("short", "settled", "drifting", "runaway", "slack")
    print(mon.explain(6))
    ev = evaluate(lambda: make_wear_env(stage.with_(multiplier=dual.multiplier), model), sb3_policy(sb3), 50, spec=spec, stage=stage.with_(multiplier=dual.multiplier), seed=0)
    assert ev.n == 50


def _small_ppo(stage, seed=0, n_envs=2, log_std_init=-1.0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from bpg.examples.wear import WearModel, make_wear_env
    model = WearModel()
    venv = DummyVecEnv([lambda: make_wear_env(stage, model) for _ in range(n_envs)])
    return PPO("MlpPolicy", venv, seed=seed, n_steps=64, batch_size=64, n_epochs=1, gamma=1.0, learning_rate=1e-3, device="cpu", verbose=0,
               policy_kwargs=dict(net_arch=[8, 8], log_std_init=log_std_init))


def test_parameter_noise_variance_and_composition():
    import torch
    from bpg import Stage
    from bpg.callbacks import ParameterAnchor, ParameterNoise
    ppo = _small_ppo(Stage(m=1.0))
    noise = ParameterNoise(ppo, kappa1=0.5, seed=0).install()
    opt = ppo.policy.optimizer
    for p in ppo.policy.parameters():
        p.grad = torch.zeros_like(p)
    before = [p.detach().clone() for p in noise.params]
    opt.step()                                              # zero gradient: Adam moves nothing, the noise moves everything
    diffs = torch.cat([(p.detach() - b).reshape(-1) for p, b in zip(noise.params, before)])
    expected = 2 * 0.5 * 1e-3
    assert abs(float(diffs.var()) / expected - 1.0) < 0.25 and noise.steps == 1
    # composed with an anchor in either order: the anchor's gradient is applied, then the noise
    anchor = ParameterAnchor(ppo, kappa_p=2.0).install()
    assert anchor.distance() == 0.0 and noise.spread() > 0
    ppo.learn(total_timesteps=64 * 2 * 3)
    assert noise.steps > 1 and anchor.spread() > 0


def test_fit_actor_mean_and_dispersion():
    import numpy as np
    from bpg import Stage
    from bpg.callbacks import fit_actor_mean, set_log_std, ParameterAnchor
    from bpg.examples.wear import IncumbentRule
    ppo = _small_ppo(Stage(m=0.03))
    rule = IncumbentRule(span=3.0, std=0.1)
    rng = np.random.default_rng(0)
    obs = np.column_stack([rng.uniform(-1, 1, 4000), np.ones(4000), rng.uniform(0, 1, 4000)]).astype(np.float32)
    target = lambda o: rule.mean_x(o[:, 2]).reshape(-1, 1)          # the rule's mean in the pre-clipping variable x in [-1, 1]
    mse = fit_actor_mean(ppo, target, obs, steps=1500, lr=3e-3)
    assert mse < 1e-3
    set_log_std(ppo, np.log(0.1))
    anchor = ParameterAnchor(ppo, kappa_p=10.0).install()          # theta_0 = the fitted actor with the incumbent's dispersion
    assert abs(anchor.effects_inputs()["dispersion_prior"] - 0.1) < 1e-6


def test_action_anchor_charges_the_failing_action():
    import numpy as np
    from bpg import Stage
    from bpg.callbacks import ActionAnchor
    ppo = _small_ppo(Stage(m=1.0), n_envs=1, log_std_init=0.0)
    seen = {}

    class Probe(ActionAnchor):
        def _on_step(self):
            before = self.locals["rewards"].copy()
            out = super()._on_step()
            for info, b, a in zip(self.locals["infos"], before, self.locals["rewards"]):
                if info.get("failure"):
                    seen.setdefault(self.include_failing_action, []).append(a - b)
            return out

    log_ref = lambda o, a: np.full(len(a), -10.0)                      # a far reference: every charge is negative
    for flag in (True, False):
        ppo.learn(total_timesteps=64 * 4, callback=Probe(1.0, log_ref, include_failing_action=flag))
    assert seen.get(True) and all(x < 0 for x in seen[True])           # failing steps are charged by default
    assert seen.get(False) and all(x == 0 for x in seen[False])        # and not with include_failing_action=False
