# %% [markdown]
# # 2. Training a standard learner through the wrapper
#
# The point of barrier policy gradients is that the learner stays unchanged: the remaining budget enters the
# state, the reward is replaced by the clipped barrier reward, and PPO trains as usual. This notebook trains
# PPO (Stable-Baselines3) on the wear-allowance problem of notebook 1 and supplies the evidence a practitioner
# needs afterwards:
#
# * training traces that show whether the run has settled,
# * the deployed candidates (stochastic and mean-action) evaluated separately,
# * the optimization error, measured against the exact frontier that this problem happens to have,
# * a sweep of prices, checked against the price–risk law and the price-sequence condition,
# * an adaptive multiplier toward a prescribed level, with a diagnosis of its trace,
# * the margin bought by the interior barrier under a misjudged allowance,
# * the report.
#
# Requires the RL extras: `pip install -e ".[rl]"`. Each run here trains for two million steps (about a minute and
# a half on one CPU core) with the paper's PPO settings; the paper's campaign used eight million steps and eight
# seeds per configuration. The whole notebook takes about eleven minutes on one CPU core.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bpg
from bpg import Requirement, Stage, evaluate, interpret, sb3_policy
from bpg.callbacks import BudgetMonitor, DualAscent, dual_step_suggestion
from bpg.convergence import seed_dispersion
from bpg.examples.wear import WearModel, make_wear_env, EXACT_FRONTIER, EXACT_BARRIER
from bpg.frontier import Run, check_runs, interpolate_frontier, price_sequence_check
from bpg.spec import discounted_total, price_from_multiplier

torch.set_num_threads(1)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
model = WearModel(); spec = model.spec(ubar=1.0)
req = Requirement(T=19, alpha=0.05, zeta=0.05, family=12)       # every learned candidate inspected below shares the error

# %% [markdown]
# ## A training helper
#
# The environment is the wrapper of notebook 1. PPO uses algorithmic discount one: the wrapper's geometric stopping
# implements $\beta$ exactly. Reward normalization keeps the scale of the failure tail $m/(1-\beta)$ manageable,
# which matters when the price is large. The settings are the paper's: 32 environments, rollouts of 256 steps,
# minibatches of 1024, ten epochs, a learning rate decaying linearly to zero, no entropy bonus.

# %%
def train(stage, seed=0, steps=2_000_000, callbacks=(), n_envs=32):
    venv = DummyVecEnv([lambda: make_wear_env(stage, model) for _ in range(n_envs)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    ppo = PPO("MlpPolicy", venv, seed=seed, n_steps=256, batch_size=1024, n_epochs=10, learning_rate=lambda p: 3e-4 * p, gamma=1.0,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0),
              device="cpu", verbose=0)
    ppo.learn(total_timesteps=steps, callback=list(callbacks))
    return ppo


def exact_value(m):
    """V_m of the exact failure-only optimum at price m (grid values from the paper's dynamic program)."""
    q, U = EXACT_FRONTIER[m]
    return U - m * q / (1 - spec.beta)

# %% [markdown]
# ## A fixed price, with its training trace
#
# `BudgetMonitor` records per rollout the failure rate, the stopped utility, the clipped value and the action
# dispersion, and tests the last rollouts for a remaining trend. A trend means the candidate is the policy at the
# end of the run, not a settled one.

# %%
stage = Stage(m=1.0)
mon = BudgetMonitor(spec.beta, stage.m)
ppo = train(stage, seed=0, callbacks=[mon])
print(mon.explain(window=40))

fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
steps = mon.series("steps") / 1e3
for ax, key, lab in zip(axes, ("failure_rate", "U", "log_std"), ("failure rate per rollout", "stopped utility U", "log action std")):
    ax.plot(steps, mon.series(key), lw=0.8); ax.set_xlabel("thousand steps"); ax.set_title(lab, fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Two candidates from one network
#
# The stochastic policy and its **mean-action** policy (the same network without its dispersion) are different
# candidates; each needs its own evaluation and validation. Because this problem has an exact solution, we can
# also measure the **optimization error**: the shortfall of the learned clipped value from the exact optimum at the
# same price. Proposition 3.3 charges exactly this quantity, written $\delta$.

# %%
env = lambda: make_wear_env(stage, model)
ev_s = evaluate(env, sb3_policy(ppo, deterministic=False), 10_000, spec=spec, stage=stage, seed=1, T=19)
ev_d = evaluate(env, sb3_policy(ppo, deterministic=True), 10_000, spec=spec, stage=stage, seed=2, T=19)
for name, e in (("stochastic", ev_s), ("mean-action", ev_d)):
    delta = exact_value(1.0) - e.V_m
    print(f"{name:11s} q_beta = {e.q_beta:.4f}  U = {e.U:.3f}  V_m = {e.V_m:.3f}  delta = {delta:+.3f}  "
          f"near-exhaustion = {e.near[0.05]:.3f}")
print(f"exact optimum at m = 1: q_beta = {EXACT_FRONTIER[1.0][0]:.4f}, U = {EXACT_FRONTIER[1.0][1]:.3f}, V_m = {exact_value(1.0):.3f}")

# %% [markdown]
# ## A sweep of prices
#
# The price selects a point on the frontier; it does not prescribe a risk level. Training at three prices, with a
# second seed at $m=1$, shows the price–risk trade-off on the learned policies. `check_runs` compares them with the
# price–risk law and with monotonicity in the price, and flags runs that are dominated (a sign of optimization
# error, not of the formulation).

# %%
prices = (0.3, 1.0, 3.0)
sweep = {(1.0, 0): ev_s}                                   # the run above
for mm, sd in ((0.3, 0), (3.0, 0), (1.0, 1)):
    st = Stage(m=mm)
    pol = train(st, seed=sd)
    sweep[(mm, sd)] = evaluate(lambda: make_wear_env(st, model), sb3_policy(pol), 10_000, spec=spec, stage=st, seed=100 + sd, T=19)
for (mm, sd), e in sorted(sweep.items()):
    print(f"m = {mm:<4g} seed {sd}: q_beta = {e.q_beta:.4f} (exact {EXACT_FRONTIER[mm][0]:.4f}), U = {e.U:.3f} "
          f"(exact {EXACT_FRONTIER[mm][1]:.3f}), delta = {exact_value(mm) - e.V_m:+.3f}")

# %%
runs = [Run(m=mm, q_beta=e.q_beta, U=e.U, label=f"seed {sd}") for (mm, sd), e in sweep.items()]
deltas = {mm: max(0.0, max(exact_value(mm) - e.V_m for (k, sd), e in sweep.items() if k == mm)) for mm in prices}
chk = check_runs(runs, beta=spec.beta, sup_U=EXACT_FRONTIER[0.0][1], delta=max(deltas.values()), tol=0.005)
print(chk.explain())

qf = np.array([v[0] for v in EXACT_FRONTIER.values()]); Uf = np.array([v[1] for v in EXACT_FRONTIER.values()]); o = np.argsort(qf)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(qf[o], Uf[o], color="0.2", label="exact frontier")
for i, mm in enumerate(prices):
    pts = [e for (k, sd), e in sweep.items() if k == mm]
    ax.plot([e.q_beta for e in pts], [e.U for e in pts], "o", color=f"C{i}", label=f"PPO, m = {mm:g}")
    ax.plot(*EXACT_FRONTIER[mm], "x", color=f"C{i}", ms=8)
ax.axvline(0.05, color="C3", ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel("breakdown probability $q_\\beta$"); ax.set_ylabel("stopped utility $U$"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# Crosses mark the exact optimum at each price. The learned policies follow the price: a higher price buys a lower
# breakdown probability at a utility cost. Each lies short of the exact point at its price by its optimization
# error $\delta$, which this problem lets us measure.
#
# **Can raising the price reach a tolerance?** Online Proposition E.4 says a sequence of prices reaches a tolerance
# when some policy has risk to spare and the charged errors grow slower than the price. `price_sequence_check` reads
# that condition from the measured shortfalls. In the paper's emissions study PPO failed it: its shortfall grew in
# proportion to the price.

# %%
print(price_sequence_check(list(deltas), list(deltas.values())).explain())

# %% [markdown]
# With three prices and one or two seeds this is a rough reading of one learner at one training budget. The check is
# most useful in a pilot sweep before committing to a large price: a shortfall that grows in proportion to the
# price says that raising the price will not reach a smaller tolerance with this learner.
#
# ## A prescribed level: the adaptive multiplier
#
# A multiplier method charges a flat penalty $\nu$ at failure and updates it after each rollout,
# $\nu\leftarrow\max\{0,\nu+c(\hat q-\alpha)\}$. If it settles at $\nu$, it returns the fixed-price optimum at
# $m=(1-\beta)\nu$ (Section 3.4). The step $c$ must respect two time scales: the multiplier should move by a small
# fraction of its eventual level per rollout. We take the expected level from the sweep: a price near $0.5$ attains
# about three percent, so $\nu\approx0.5/(1-\beta)=10$.
#
# The prescribed level is set **below** the tolerance: a multiplier settled at the tolerance itself does not pass
# validation at that tolerance.

# %%
alpha_level = 0.02
dual = DualAscent(alpha=alpha_level, step=dual_step_suggestion(discounted_total(0.5, spec.beta), rollouts_to_settle=5))
adaptive_stage = Stage(m=0.0, penalty="multiplier")
mon_a = BudgetMonitor(spec.beta, 0.0, penalty="multiplier", dual=dual)
ppo_a = train(adaptive_stage, seed=0, callbacks=[dual, mon_a])
diag = dual.diagnose(spec.beta, window=40)
print(diag.status, "|", diag.explanation)

nu = np.array([t["multiplier"] for t in dual.trace]); fr = np.array([t["failure_rate"] for t in dual.trace])
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
axes[0].plot(nu * (1 - spec.beta)); axes[0].set_title("price-equivalent $(1-\\beta)\\nu$ per rollout", fontsize=9)
axes[1].plot(fr, lw=0.6); axes[1].axhline(alpha_level, color="C3", ls=":"); axes[1].set_title("failure rate per rollout", fontsize=9)
for ax in axes: ax.set_xlabel("rollout")
plt.tight_layout(); plt.show()

final_stage = adaptive_stage.with_(multiplier=dual.multiplier)
ev_a = evaluate(lambda: make_wear_env(final_stage, model), sb3_policy(ppo_a), 10_000, spec=spec, stage=final_stage, seed=3, T=19)
print(f"adaptive candidate: q_beta = {ev_a.q_beta:.4f}, U = {ev_a.U:.3f}, final nu = {dual.multiplier:.2f} "
      f"(price-equivalent m = {price_from_multiplier(dual.multiplier, spec.beta):.3f})")

# %% [markdown]
# Read the diagnosis printed above. *Settled* means the candidate is the fixed-price optimum at the price-equivalent
# $m=(1-\beta)\nu$, up to its optimization error. *Drifting* means the multiplier was still moving when training
# stopped: the candidate is not a fixed-price optimum at any identified price, and longer training or a different
# step is needed. *Runaway* means every episode fails and the flat penalty carries no gradient (the fuel-tank study).
#
# ## Buying a margin: the interior barrier under a misjudged allowance
#
# A failure-only price leaves no reserve: its optimal policies run close to exhaustion whenever that pays. If the
# true allowance is ten percent smaller than the one the policy observes, breakdowns rise sharply. The interior
# barrier ($\theta>0$) keeps a graded margin at a utility cost bounded by its interior loss. The wrapper option
# `true_budget_scale=0.9` judges failure against the reduced allowance while the policy sees the nominal reserve.

# %%
barrier_stage = Stage(m=1.0, theta=0.1)
ppo_b = train(barrier_stage, seed=0)
rows = []
for name, pol, st in (("failure-only price", ppo, stage), ("interior barrier", ppo_b, barrier_stage)):
    nominal = evaluate(lambda: make_wear_env(st, model), sb3_policy(pol), 6000, spec=spec, stage=st, seed=5)
    reduced = evaluate(lambda: make_wear_env(st, model, true_budget_scale=0.9), sb3_policy(pol), 6000, spec=spec, stage=st, seed=5)
    rows.append((name, nominal.q_beta, nominal.U, nominal.I_m, nominal.near[0.05], reduced.q_beta))
print(f"{'':20s} {'q_beta':>8s} {'U':>6s} {'I_m':>6s} {'near':>6s} {'q at 0.9b':>10s}")
for r in rows:
    print(f"{r[0]:20s} {r[1]:8.4f} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:10.4f}")

# %% [markdown]
# The barrier gives up utility at the nominal allowance and keeps breakdowns low when the allowance is misjudged.
# If the size of the possible error is known, planning with a correspondingly tightened allowance is the more
# economical margin; the barrier is the graded margin for an error of unknown size.
#
# ## Validate and report
#
# The fixed-price stochastic candidate is the one we would deploy. Its optimization error is known here from the
# exact optimum, so the report can state the fixed-price reading with that $\delta$. Every learned candidate
# inspected in this notebook shares the validation error (`family=12`). The requirement $p_{19}\le0.05$ is tested
# here through $q_\beta\le0.05\,\beta^{19}$, which is conservative under geometric stopping; notebook 4 tests
# $p_T$ directly.

# %%
delta_s = max(0.0, exact_value(1.0) - ev_s.V_m)
gap = (interpolate_frontier([Run(m=k, q_beta=q, U=U) for k, (q, U) in EXACT_FRONTIER.items()], ev_s.q_beta) - ev_s.U) / ev_s.U
seeds_q = seed_dispersion([sweep[(1.0, 0)].q_beta, sweep[(1.0, 1)].q_beta])
rep = interpret(spec, req, stage, ev_s, sup_U=EXACT_FRONTIER[0.0][1], delta=delta_s, trends=mon.diagnostics(window=40),
                seeds={"q_beta at m = 1": seeds_q}, exact_frontier_gap=gap)
print(rep.text())

# %% [markdown]
# The route through $q_\beta$ is conservative under geometric stopping: $q_\beta$ counts every failure before
# maintenance access, including failures long after action 19. The requirement concerns $p_{19}$, so we validate it
# directly on fresh episodes over a fixed horizon of 20 actions, without the clock. The same network is the same
# candidate here; only the evaluation changes.

# %%
from bpg import BudgetWrapper
from bpg.examples.wear import WearCore, wear_cost
fixed_spec = spec.with_(horizon=req.T + 1)
ev_fixed = evaluate(lambda: BudgetWrapper(WearCore(model), wear_cost, fixed_spec, stage, geometric_stop=False), sb3_policy(ppo),
                    10_000, spec=fixed_spec, stage=stage, seed=9, T=req.T)
rep_fixed = interpret(fixed_spec, req, stage, ev_fixed, delta=None)
print(f"p_19 = {ev_fixed.p_T:.4f}; certified: {rep_fixed.certified}")
print("\n".join(rep_fixed.text().split("\n")[:12]))

# %% [markdown]
# ## Summary
#
# * PPO is trained unchanged; the wrapper carries the budget and the clipped reward.
# * The learned fixed-price policies follow the price and fall short of the exact optimum by an optimization error
#   that this problem lets us measure. On a new problem there is no exact frontier: notebook 5 shows what can be said then.
# * A multiplier method targets a level, needs a step that respects the time scales, and should aim below the tolerance.
# * The barrier buys a measurable margin against a misjudged allowance.
#
# Next: **notebook 3** looks at the two regularizers and the roles of their parts.
