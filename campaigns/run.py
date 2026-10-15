"""Train and evaluate one (problem, configuration, seed) and write its record.

    python campaigns/run.py retailer lump-5000 --seed 1
    python campaigns/run.py retailer lump-5000 --seed 1 --steps 200000 --eval 200 --out /tmp/pilot   (a short pilot)

Writes results/<problem>/<configuration>/seed<seed>/ with model.zip, vecnormalize.pkl (when observations are
normalized) and record.json: the configuration, the PPO settings, the training traces (per rollout), and the
per-episode results of the evaluation on fresh episodes (failure, failure time, stopped utility, interior loss,
length, lowest reserve, and the problem's own extras), from which bpg.evaluation_from_episodes rebuilds the
evaluation without simulating again.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import problems as P  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import bpg  # noqa: E402
from bpg.callbacks import ActionAnchor, BudgetMonitor, DualAscent, ParameterAnchor, ParameterNoise, fit_actor_mean, set_log_std  # noqa: E402

torch.set_num_threads(1)


def _jsonable(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return float(x)


def build(problem, config, seed, venv_stats=None):
    s = problem.ppo
    venv = DummyVecEnv([lambda: problem.make_env(config.stage, **config.env_kw) for _ in range(s.n_envs)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=s.norm_obs, norm_reward=s.norm_reward, clip_reward=1e9, clip_obs=1e3, gamma=s.gamma)
    if venv_stats is not None:
        venv.obs_rms, venv.ret_rms = venv_stats.obs_rms, venv_stats.ret_rms
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=s.n_steps, batch_size=s.batch_size, n_epochs=s.n_epochs,
                learning_rate=lambda p: s.lr * p, gamma=s.gamma, gae_lambda=s.gae_lambda, clip_range=s.clip_range,
                ent_coef=s.ent_coef, policy_kwargs=dict(net_arch=list(s.net_arch), log_std_init=s.log_std_init),
                device="cpu", verbose=0)
    return model, venv


def incumbent_observations(problem, config, n_episodes=400, seed=12345):
    """Observations for fitting the actor to the incumbent: those a noisy version of the incumbent visits, so that the
    fit also covers the states (low cash, low fuel) where the incumbent's safeguards act."""
    _, _, explorer = problem.incumbent()
    env = problem.make_env(config.stage, **config.env_kw)
    out = []
    for i in range(n_episodes):
        o, _ = env.reset(seed=seed + i)
        while True:
            out.append(o)
            o, r, te, tr, info = env.step(explorer(o))
            if te or tr:
                break
    return np.asarray(out, dtype=np.float32)


def evaluate_policy(problem, config, policy, n, seed, normalizer=None, **env_kw):
    """n fresh episodes; per-episode arrays for bpg.evaluation_from_episodes plus the problem's extras."""
    env = problem.make_env(config.stage, **{**config.env_kw, **env_kw})
    rows = []
    for i in range(n):
        o, _ = env.reset(seed=seed + i)
        infos = []
        while True:
            x = normalizer.normalize_obs(o) if normalizer is not None else o
            a = policy(x)
            o, r, te, tr, info = env.step(a)
            infos.append(info)
            if te or tr:
                break
        rec = info["budget_episode"]
        row = dict(failed=bool(rec["failed"]), tau=-1 if rec["tau"] is None else int(rec["tau"]), U=float(rec["U"]),
                   I=float(rec["I"]), length=int(rec["length"]), min_reserve=float(rec["min_reserve_fraction"]),
                   clipped=float(rec["clipped_return"]))
        if problem.summarize is not None:
            row.update(problem.summarize(infos))
        rows.append(row)
    return {k: [r[k] for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem"); ap.add_argument("config"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--eval", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    problem = P.get(args.problem)
    config = problem.configs[args.config]
    steps = args.steps or config.steps
    n_eval = args.eval or config.eval_episodes
    out = Path(args.out) / problem.name / args.config / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    init_stats = None
    if config.init:
        src = Path(args.out) / problem.name / config.init / f"seed{args.seed}"
        init_stats = VecNormalize.load(str(src / "vecnormalize.pkl"), DummyVecEnv([lambda: problem.make_env(config.stage, **config.env_kw)])) \
            if (src / "vecnormalize.pkl").exists() else None
    model, venv = build(problem, config, args.seed, init_stats)
    prior_mse = None
    if config.init:
        start = PPO.load(str(src / "model.zip"), device="cpu")
        model.policy.load_state_dict(start.policy.state_dict())
    if config.prior == "incumbent":
        mean, log_std, _ = problem.incumbent()
        obs = incumbent_observations(problem, config)
        prior_mse = fit_actor_mean(model, lambda o: np.stack([mean(x) for x in o]), obs, seed=args.seed, steps=15000)
        set_log_std(model, log_std)

    callbacks = []
    dual = None
    if config.dual:
        dual = DualAscent(alpha=config.dual["alpha"], step=config.dual["step"], multiplier0=config.dual.get("multiplier0", 0.0))
        callbacks.append(dual)
    st = config.stage
    monitor = BudgetMonitor(problem.spec.beta, st.m, horizon=problem.spec.horizon, penalty=st.penalty, dual=dual,
                            multiplier=(st.multiplier if st.penalty == "multiplier" and dual is None else None))
    callbacks.append(monitor)
    anchor = ParameterAnchor(model, config.anchor).install() if config.anchor else None
    action_anchor = None
    if config.action_anchor:
        reference = PPO.load(str(src / "model.zip"), device="cpu")

        def log_reference(obs, act):
            with torch.no_grad():
                _, logp, _ = reference.policy.evaluate_actions(torch.as_tensor(obs, dtype=torch.float32),
                                                              torch.as_tensor(act, dtype=torch.float32))
            return logp.numpy()
        action_anchor = ActionAnchor(config.action_anchor, log_reference)
        callbacks.insert(0, action_anchor)
    noise = ParameterNoise(model, config.noise, seed=args.seed).install() if config.noise else None

    # the starting policy is a candidate too: evaluate it before training when it is a prior or a trained policy
    def policy_of(m):
        return lambda x: m.predict(x, deterministic=False)[0]
    normalizer = venv if problem.ppo.norm_obs else None
    start_eval = None
    if config.prior or config.init:
        start_eval = evaluate_policy(problem, config, policy_of(model), min(n_eval, 1000), seed=7_000_000 + 1000 * args.seed,
                                     normalizer=normalizer)

    model.learn(total_timesteps=steps, callback=callbacks)
    train_wall = time.time() - t0
    model.save(str(out / "model.zip"))
    if problem.ppo.norm_obs:
        venv.save(str(out / "vecnormalize.pkl"))
    venv.training = False

    eval_config = config
    if dual is not None:
        from dataclasses import replace
        eval_config = replace(config, stage=config.stage.with_(multiplier=dual.multiplier))
    episodes = evaluate_policy(problem, eval_config, policy_of(model), n_eval, seed=9_000_000 + 10_000 * args.seed, normalizer=normalizer)
    extra_evals = [dict(env_kw=kw, episodes=evaluate_policy(problem, eval_config, policy_of(model), n_eval,
                                                           seed=9_500_000 + 10_000 * args.seed, normalizer=normalizer, **kw))
                   for kw in config.eval_kw]

    record = dict(
        problem=problem.name, config=args.config, seed=args.seed, steps=int(model.num_timesteps), note=config.note,
        stage=asdict(eval_config.stage), trained_stage=asdict(config.stage), ppo=asdict(problem.ppo),
        regularizers=dict(anchor=config.anchor, noise=config.noise, action_anchor=config.action_anchor, prior=config.prior,
                          init=config.init, prior_mse=prior_mse),
        dual=None if dual is None else dict(config.dual, final=dual.multiplier, trace=dual.trace,
                                            diagnosis=asdict(dual.diagnose(problem.spec.beta, window=40))),
        anchor_distance=None if anchor is None else anchor.distance(), noise_spread=None if noise is None else noise.spread(),
        action_log_ratio=None if action_anchor is None else action_anchor.mean_log_ratio,
        monitor=monitor.trace, T=problem.T, evaluation=episodes, start_evaluation=start_eval, extra_evaluations=extra_evals,
        train_wall=train_wall, total_wall=time.time() - t0,
        versions=dict(bpg=bpg.__version__, python=platform.python_version(), torch=torch.__version__,
                      sb3=__import__("stable_baselines3").__version__, gymnasium=__import__("gymnasium").__version__),
    )
    (out / "record.json").write_text(json.dumps(record, default=_jsonable))
    ev = np.asarray(episodes["failed"])
    print(f"{problem.name}/{args.config}/seed{args.seed}: {steps} steps in {train_wall / 60:.1f} min; "
          f"failure rate {ev.mean():.4f}, U {np.mean(episodes['U']):.4g}")


if __name__ == "__main__":
    main()
