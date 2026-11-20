# %% [markdown]
# # T2. Training a learner against a grid-based benchmark
#
# The point of barrier policy gradients is that the learner stays unchanged: the remaining budget enters the
# state, the reward is replaced by the clipped barrier reward, and PPO trains as usual. This notebook trains
# PPO (Stable-Baselines3) on the wear-allowance problem of notebook T1 and supplies the evidence a practitioner
# needs afterwards:
#
# * training traces that show whether the run has converged,
# * the deployed candidates (stochastic and mean-action) evaluated separately,
# * the measured clipped-value shortfall against the grid benchmark,
# * estimated performance across tested prices and discussion of the price-sequence condition,
# * an adaptive multiplier toward a prescribed level, with a diagnosis of its trace,
# * the reserve kept by the interior barrier when the true allowance is smaller than planned,
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

import barrierpg
from barrierpg import Requirement, Stage, evaluate, interpret, sb3_policy
from barrierpg.callbacks import BudgetMonitor, DualAscent, dual_step_suggestion
from barrierpg.examples.wear import WearModel, make_wear_env, EXACT_FRONTIER
from barrierpg.frontier import Run, interpolate_frontier
from barrierpg.spec import discounted_total, price_from_multiplier

torch.set_num_threads(1)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
model = WearModel(); spec = model.spec(ubar=1.0)
req = Requirement(T=19, alpha=0.05, zeta=0.05, family=2)       # two final evaluation routes

# %% [markdown]
# ## A training helper
#
# The environment is the wrapper of notebook T1. PPO uses algorithmic discount one: the wrapper's geometric stopping
# implements $\beta$ exactly. Reward normalization keeps the scale of the charge at failure, $m/(1-\beta)$, manageable,
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


def benchmark_value(m):
    """V_m of the failure-only grid benchmark at price m."""
    q, U = EXACT_FRONTIER[m]
    return U - m * q / (1 - spec.beta)

# %% [markdown]
# ## A fixed price, with its training trace
#
# `BudgetMonitor` records per rollout the failure rate, the stopped utility, the clipped value and the action
# dispersion, and tests the last rollouts for a remaining trend. A trend means the candidate is the policy at the
# end of the run, not one at which training has converged.

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
# candidates; each needs its own evaluation and validation. We compare the learned clipped value with the grid-based
# benchmark at the same price. The shortfall is a benchmark comparison. A certificate requires a justified upper bound on the optimization gap.

# %%
env = lambda: make_wear_env(stage, model)
ev_s = evaluate(env, sb3_policy(ppo, deterministic=False), 10_000, spec=spec, stage=stage, seed=1, T=19)
ev_d = evaluate(env, sb3_policy(ppo, deterministic=True), 10_000, spec=spec, stage=stage, seed=2, T=19)
for name, e in (("stochastic", ev_s), ("mean-action", ev_d)):
    shortfall = benchmark_value(1.0) - e.V_m
    print(f"{name:11s} q_beta = {e.q_beta:.4f}  U = {e.U:.3f}  V_m = {e.V_m:.3f}  shortfall = {shortfall:+.3f}  "
          f"reserve below 5% of b = {e.near[0.05]:.3f}")
print(f"grid benchmark at m = 1: q_beta = {EXACT_FRONTIER[1.0][0]:.4f}, U = {EXACT_FRONTIER[1.0][1]:.3f}, V_m = {benchmark_value(1.0):.3f}")

# %% [markdown]
# ## Training at several prices
#
# The price selects a point on the frontier; it does not prescribe a risk level. Training at three prices, with a
# second seed at $m=1$, shows the measured price–risk trade-off.

# %%
prices = (0.3, 1.0, 3.0)
by_price = {(1.0, 0): ev_s}                                   # the run above
for mm, sd in ((0.3, 0), (3.0, 0), (1.0, 1)):
    st = Stage(m=mm)
    pol = train(st, seed=sd)
    by_price[(mm, sd)] = evaluate(lambda: make_wear_env(st, model), sb3_policy(pol), 10_000, spec=spec, stage=st, seed=100 + sd, T=19)
for (mm, sd), e in sorted(by_price.items()):
    print(f"m = {mm:<4g} seed {sd}: q_beta = {e.q_beta:.4f} (benchmark {EXACT_FRONTIER[mm][0]:.4f}), U = {e.U:.3f} "
          f"(benchmark {EXACT_FRONTIER[mm][1]:.3f}), shortfall = {benchmark_value(mm) - e.V_m:+.3f}")

# %%
qf = np.array([v[0] for v in EXACT_FRONTIER.values()]); Uf = np.array([v[1] for v in EXACT_FRONTIER.values()]); o = np.argsort(qf)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(qf[o], Uf[o], color="0.2", label="grid benchmark")
for i, mm in enumerate(prices):
    pts = [e for (k, sd), e in by_price.items() if k == mm]
    ax.plot([e.q_beta for e in pts], [e.U for e in pts], "o", color=f"C{i}", label=f"PPO, m = {mm:g}")
    ax.plot(*EXACT_FRONTIER[mm], "x", color=f"C{i}", ms=8)
ax.axvline(req.alpha * spec.beta ** req.T, color="C3", ls=":", lw=1, label="sufficient geometric level")
ax.set_xscale("log"); ax.set_xlabel("breakdown probability $q_\\beta$"); ax.set_ylabel("stopped utility $U$"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# Crosses mark the grid benchmark. The points show how estimated risk and utility change across the tested prices.
#
# **Can raising the price reach a tolerance?** By the paper's *Price sequences reaching a tolerance*, a sequence of prices reaches a tolerance
# when some policy has risk to spare and the charged errors grow slower than the price. A finite set of benchmark shortfalls does not verify this limit.

# %%
print({m: [round(benchmark_value(m) - e.V_m, 3) for (price, _), e in by_price.items() if price == m] for m in prices})

# %% [markdown]
# With three prices and one or two seeds this is a rough reading of one learner at one training budget. The check is
# useful on short pilot runs before committing to a large price. A shortfall that grows roughly in proportion
# to price is reason to investigate the learner's optimization error. A few noisy grid comparisons cannot determine
# whether that error grows sublinearly along a price sequence.
#
# ## A prescribed level: the adaptive multiplier
#
# A multiplier method charges a flat penalty $\nu$ at failure and updates it after each rollout,
# $\nu\leftarrow\max\{0,\nu+c(\hat q-\alpha)\}$. At a fixed $\nu$, the failure-only objective in this setting
# is the fixed-price objective at $m=(1-\beta)\nu$. This identifies the objective even when PPO's optimization
# error is unknown. We size the step using the fixed-price runs. If they bracket the target with decreasing,
# positive risks, we interpolate on a log scale. Otherwise we use the tested price whose estimated risk was closest.
#
# The step is half the expected level per rollout in which every episode fails. Early in training, when many episodes
# fail, the multiplier rises within a few rollouts. Near the tolerance, where the failure rate is off by about a
# percentage point, one rollout moves it by about half a percent of that reference level.
#
# Leave room for sampling error when choosing a target for the same event as validation. This pilot targets
# geometric risk 0.02. The finite-horizon test below uses a stricter sufficient geometric level, $0.05\beta^{19}$.

# %%
alpha_level = 0.02
m_runs = sorted(prices)
q_runs = np.array([np.mean([e.q_beta for (k, sd), e in by_price.items() if k == mm]) for mm in m_runs])
if np.all(np.diff(q_runs) < 0) and 0 < q_runs[-1] <= alpha_level <= q_runs[0]:
    m_expected = float(np.exp(np.interp(np.log(alpha_level), np.log(q_runs[::-1]), np.log(m_runs[::-1]))))
    print(f"Interpolated trial price for geometric risk {alpha_level:g}: {m_expected:.2f}")
else:
    m_expected = m_runs[int(np.argmin(np.abs(q_runs - alpha_level)))]
    print(f"Using the closest tested price, {m_expected:g}, to size the multiplier step.")
nu_expected = discounted_total(m_expected, spec.beta)
print(f"Reference multiplier nu = {nu_expected:.1f}")
dual = DualAscent(alpha=alpha_level, step=dual_step_suggestion(nu_expected, rollouts_to_reach=2))
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
# The diagnosis summarizes two features of the recent training trace: drift in the multiplier updates and
# proximity of the sampled risk to the target. A stable trace does not bound PPO's optimization error. A growing
# multiplier warrants checking the step size, training budget and feasibility. Persistent drift alone cannot
# establish infeasibility. Re-evaluate the final policy on fresh episodes at the deployment horizon.
#
# ## Keeping a reserve: the interior barrier when the true allowance is smaller than planned
#
# A failure-only price can favor policies that operate close to exhaustion. Compare both trained policies
# when the true allowance is ten percent smaller than the one the policy observes. The interior
# barrier ($\theta>0$) charges for using up the reserve before failure. The wrapper option
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
# These are simulator estimates for the two trained policies. Compare the nominal utility as well as the
# breakdown probabilities at both allowances. A known reduction can be represented directly in the budget used for training.
#
# ## Validate and report
#
# We use new seeds for the final validation of the stochastic policy. The two evaluation routes split the error.
# The benchmark comparison is descriptive. The requirement $p_{19}\le0.05$ is tested
# here through $q_\beta\le0.05\,\beta^{19}$, which is conservative under geometric stopping; notebook T4 tests
# $p_T$ directly.

# %%
torch.manual_seed(100_000)
final_geo = evaluate(env, sb3_policy(ppo), 10_000, spec=spec, stage=stage, seed=100_000, T=req.T)
grid = [Run(m=k, q_beta=q, U=U) for k, (q, U) in EXACT_FRONTIER.items()]
if min(r.q_beta for r in grid) <= final_geo.q_beta <= max(r.q_beta for r in grid):
    benchmark_U = interpolate_frontier(grid, final_geo.q_beta)
    gap = (benchmark_U - final_geo.U) / benchmark_U
    print(f"Relative utility shortfall against the interpolated grid benchmark: {gap:.2%}")
else:
    print("Measured risk lies outside the grid benchmark range.")
rep = interpret(spec, req, stage, final_geo, trends=mon.diagnostics(window=40))
print(rep.text())

# %% [markdown]
# The route through $q_\beta$ is conservative under geometric stopping: $q_\beta$ counts every failure before
# maintenance access, including failures long after action 19. The requirement concerns $p_{19}$, so we validate it
# directly on fresh episodes over a fixed horizon of 20 actions, without the clock. The same network is the same
# candidate here; only the evaluation changes.

# %%
from barrierpg import BudgetWrapper
from barrierpg.examples.wear import WearCore, wear_cost
fixed_spec = spec.with_(horizon=req.T + 1)
torch.manual_seed(200_000)
ev_fixed = evaluate(lambda: BudgetWrapper(WearCore(model), wear_cost, fixed_spec, stage, geometric_stop=False), sb3_policy(ppo),
                    10_000, spec=fixed_spec, stage=stage, seed=200_000, T=req.T)
rep_fixed = interpret(fixed_spec, req, stage, ev_fixed, delta=None)
print(f"p_19 = {ev_fixed.p_T:.4f}; certified: {rep_fixed.certified}")
print("\n".join(rep_fixed.text().split("\n")[:12]))

# %% [markdown]
# ## Summary
#
# * PPO is trained unchanged; the wrapper carries the budget and the clipped reward.
# * The grid-based comparisons illustrate measured performance. The validation test uses the candidate's failure counts.
# * A multiplier method targets a level, needs a step that respects the time scales, and should aim below the tolerance.
# * The barrier keeps a measurable reserve for a smaller true allowance.
#
# Next: **notebook T3** looks at the two regularizers and the roles of their parts.
