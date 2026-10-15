"""The convergence check of a training run, on synthetic per-batch records and on a short PPO run."""
import numpy as np

from barrierpg.convergence import training_convergence


def _trace(f, n=100, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(steps=1000 * (k + 1), failure_rate=max(0.0, 0.05 + noise * rng.standard_normal()),
                 U=f(k) + 0.5 * rng.standard_normal(), V_m=f(k) - 1.0 + 0.5 * rng.standard_normal(), log_std=-1.0)
            for k in range(n)]


def test_converged_and_rising_runs():
    flat = training_convergence(_trace(lambda k: 50.0))
    assert flat.status == "converged" and "converged." in flat.explain()
    rising = training_convergence(_trace(lambda k: 10.0 + 0.5 * k))
    assert rising.status == "not converged" and any(c.key == "U" and c.status == "rising" for c in rising.moving)
    assert "Train longer" in rising.explain()
    assert training_convergence(_trace(lambda k: 50.0, n=15)).status == "too short"   # windows of 10 batches at least


def test_monitor_convergence_and_report():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("Stable-Baselines3 missing; skipped"); return
    from barrierpg import Requirement, Stage, evaluate, interpret
    from barrierpg.callbacks import BudgetMonitor
    from barrierpg.examples.wear import WearModel, make_wear_env
    model = WearModel(); spec = model.spec(); stage = Stage(m=1.0)
    venv = DummyVecEnv([lambda: make_wear_env(stage, model) for _ in range(4)])
    sb3 = PPO("MlpPolicy", venv, seed=0, n_steps=64, batch_size=64, n_epochs=2, gamma=1.0, device="cpu", verbose=0,
              policy_kwargs=dict(net_arch=[16, 16]))
    mon = BudgetMonitor(spec.beta, stage.m, check_every=5)
    sb3.learn(64 * 4 * 12, callback=[mon])
    c = mon.convergence()
    assert c.status in ("converged", "not converged", "too short") and len(mon.checks) == 2
    ev = evaluate(lambda: make_wear_env(stage, model), lambda o: sb3.predict(o)[0], 50, spec=spec, stage=stage, T=19, seed=0)
    text = interpret(spec, Requirement(T=19, alpha=0.05), stage, ev, training=c).summary()
    assert "Training" in text or "training record" in text
