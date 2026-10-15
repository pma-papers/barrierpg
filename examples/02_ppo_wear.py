"""Train PPO on the wear-allowance problem through the wrapper and callbacks, then certify.

Runtime: a Python with gymnasium, stable-baselines3 and torch (pip install -e ".[rl]").
Configurations: fixed price m (optionally with the interior barrier), adaptive multiplier at a tolerance,
the fixed price with a parameter anchor toward the initial actor, and the anchor with injected parameter
noise (the noise and pull roles of the paper's study of the roles of the regularizers). The defaults are the paper's PPO settings (32 environments,
minibatches of 1024, a linearly decaying learning rate) at a quarter of its budget: two million steps, about
90 seconds per run on one CPU core.

  python examples/02_ppo_wear.py --config fixed --m 1
  python examples/02_ppo_wear.py --config adaptive --alpha 0.02
  python examples/02_ppo_wear.py --config fixed --m 1 --theta 0.1
  python examples/02_ppo_wear.py --config noise --m 0.03 --kappa-p 10 --kappa1 0.1
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bpg import Requirement, Stage, evaluate, interpret, sb3_policy
from bpg.callbacks import BudgetMonitor, DualAscent, ParameterAnchor, ParameterNoise, dual_step_suggestion
from bpg.convergence import seed_dispersion
from bpg.examples.wear import EXACT_FRONTIER, WearModel, make_wear_env
from bpg.frontier import Run, interpolate_frontier
from bpg.regularizers import regularizer_effects
from bpg.spec import discounted_total

ap = argparse.ArgumentParser()
ap.add_argument("--config", choices=["fixed", "adaptive", "anchored", "noise"], default="fixed")
ap.add_argument("--m", type=float, default=1.0); ap.add_argument("--theta", type=float, default=0.0)
ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--dual-step", type=float, default=None)
ap.add_argument("--kappa-p", type=float, default=10.0); ap.add_argument("--kappa1", type=float, default=0.1)
ap.add_argument("--steps", type=int, default=2_000_000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-envs", type=int, default=32); ap.add_argument("--eval", type=int, default=5000)
args = ap.parse_args()

model = WearModel(); spec = model.spec(ubar=1.0)
req = Requirement(T=19, alpha=args.alpha, zeta=0.05)
if args.config == "adaptive":
    stage = Stage(m=args.m if args.theta > 0 else 0.0, theta=args.theta, penalty="multiplier")   # --m sizes the dual step
else:
    stage = Stage(m=args.m, theta=args.theta)

venv = DummyVecEnv([lambda: make_wear_env(stage, model, geometric_stop=True) for _ in range(args.n_envs)])
venv.seed(args.seed)
venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)   # algorithmic discount one: geometric stopping carries beta
ppo = PPO("MlpPolicy", venv, seed=args.seed, n_steps=256, batch_size=1024, n_epochs=10, learning_rate=lambda p: 3e-4 * p, gamma=1.0,
          gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0),
          device="cpu", verbose=0)
callbacks = []
if args.config == "adaptive":
    step = args.dual_step if args.dual_step is not None else dual_step_suggestion(discounted_total(args.m, spec.beta))
    dual = DualAscent(alpha=args.alpha, step=step); callbacks.append(dual)
    callbacks.insert(0, BudgetMonitor(spec.beta, stage.m, penalty="multiplier", dual=dual))
    print(f"dual step {step:.4g} (a twentieth of the price-equivalent multiplier {discounted_total(args.m, spec.beta):.4g} per rollout)")
else:
    callbacks.append(BudgetMonitor(spec.beta, stage.m))
anchor = noise = None
if args.config in ("anchored", "noise"):
    anchor = ParameterAnchor(ppo, kappa_p=args.kappa_p).install()     # prior = the initial actor; evaluate it as a candidate
    prior_eval = evaluate(lambda: make_wear_env(stage, model), sb3_policy(ppo), args.eval, spec=spec, stage=stage, seed=100, T=19)
    print("prior policy (the initial actor): " + prior_eval.summary().split("\n")[0])
    if args.config == "noise":
        noise = ParameterNoise(ppo, kappa1=args.kappa1, seed=args.seed).install()

ppo.learn(total_timesteps=args.steps, callback=callbacks)
mon = callbacks[0]
print("\nTRAINING TRACE DIAGNOSTICS\n" + mon.explain(window=min(20, len(mon.trace))))
final_stage = stage
if args.config == "adaptive":
    diag = dual.diagnose(spec.beta); print("\nMULTIPLIER: " + diag.explanation)
    final_stage = stage.with_(multiplier=dual.multiplier)
if anchor is not None:
    print(f"\nparameter distance from the prior: {anchor.distance():.4g} (per-parameter spread {anchor.spread():.3g})")

print("\nEVALUATION OF THE FINAL STOCHASTIC POLICY AND OF ITS MEAN-ACTION POLICY (two candidates)")
env = lambda: make_wear_env(final_stage, model)
stoch = evaluate(env, sb3_policy(ppo, False), args.eval, spec=spec, stage=final_stage, seed=10_000 + args.seed, T=19)
mean = evaluate(env, sb3_policy(ppo, True), args.eval, spec=spec, stage=final_stage, seed=20_000 + args.seed, T=19)
print("stochastic:   " + stoch.summary().replace("\n", "\n              "))
print("mean-action:  " + mean.summary().replace("\n", "\n              "))
reduced = evaluate(lambda: make_wear_env(final_stage, model, true_budget_scale=0.9), sb3_policy(ppo, False), args.eval, spec=spec, stage=final_stage, seed=30_000 + args.seed, T=19)
print(f"stochastic policy under a true allowance of 0.9 b: q_beta = {reduced.q_beta:.4g} (near-exhaustion at the nominal allowance: {stoch.near[0.05]:.3g})")

exact = [Run(m=m, q_beta=q, U=U) for m, (q, U) in EXACT_FRONTIER.items()]
gap = (interpolate_frontier(exact, stoch.q_beta) - stoch.U) / interpolate_frontier(exact, stoch.q_beta)
print(f"relative shortfall from the exact frontier at the attained risk: {gap:.2%} (the measured optimization error)")

effects = None
if anchor is not None:
    effects = regularizer_effects(spec, final_stage.with_(kappa=args.kappa_p), dict(q_beta=stoch.q_beta, U=stoch.U, q_se=stoch.q_se, U_se=stoch.U_se),
                                  prior_policy=dict(q_beta=prior_eval.q_beta, U=prior_eval.U, q_se=prior_eval.q_se, U_se=prior_eval.U_se),
                                  noise_kappa1=(args.kappa1 if noise is not None else None), anchor_kappa2=args.kappa_p,
                                  spread=(anchor.spread() if noise is not None else None), **anchor.effects_inputs())
print("\n" + interpret(spec, req, final_stage, stoch, sup_U=EXACT_FRONTIER[0.0][1], delta=max(gap, 0.0) * stoch.U, effects=effects,
                       multiplier=(dual.diagnose(spec.beta) if args.config == "adaptive" else None),
                       trends=mon.diagnostics(window=min(20, len(mon.trace))), exact_frontier_gap=gap).text())
