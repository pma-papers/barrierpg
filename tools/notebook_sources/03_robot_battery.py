# %% [markdown]
# # Locomotion under a battery budget
#
# We impose a battery allowance on MuJoCo's HalfCheetah over a mission of 1,000 actions. Consumption includes an idle load and motor losses that increase with squared torque. An unobserved efficiency parameter varies by mission. The robot is stranded when a step first makes the battery reserve negative.
#
# The observations include charge and, in the clock configurations, elapsed mission time. We compare fixed failure prices and an interior barrier using distance, stranding probability and the minimum reserve. To measure sensitivity to available energy, we also evaluate the fixed policies with 90% of the nominal allowance while retaining their nominal reserve observations.
#
# The illustrations use saved runs of about two million interactions per seed and three seeds per configuration. The training function is included for local reruns. The notebook requires Stable-Baselines3 and Gymnasium with MuJoCo. [Practitioner guide](../docs/practitioner-guide.md)
#
# ### Locomotion with an energy constraint
#
# Let $a_t$ denote the vector of motor commands after clipping to $[-1,1]$. At reset, the simulator draws a latent efficiency multiplier $E=\exp(0.15G-0.15^2/2)$ with $G$ standard normal. Energy use is $c_t=E(0.5+\|a_t\|_2^2)$ and the reserve follows $z_{t+1}=z_t-c_t$ with $z_0=2500$. Stranding occurs at the first action $\tau$ with $z_{t+1}<0$. The policy observes the simulator output and nominal reserve. Some configurations also include the mission clock.
#
# Take $u_t$ to be HalfCheetah's reward, which includes forward motion and the simulator's control penalty. The objective is expressed in reward units, and distance is reported separately. Let $U_{1000}(\pi)=\mathbb E_\pi[\sum_{0\le t<1000,\ t<\tau}0.999^t u_t]$. With $\Pi$ the actor class, the deployment problem is
#
# $$
# \begin{aligned}
# \underset{\pi\in\Pi}{\mathrm{maximize}}\quad & U_{1000}(\pi)\\
# \text{subject to}\quad & \Pr_\pi(\tau\le999)\le0.05.
# \end{aligned}
# $$
#
# In the fixed-price problem, $F_t=m\sum_{j=0}^{999-t}0.999^j$ is the charge if failure occurs at $t$. PPO searches for a policy with high value under this charge.
#
# $$
# \underset{\pi\in\Pi}{\mathrm{maximize}}\quad U_{1000}(\pi)-\mathbb E_\pi[0.999^\tau F_\tau\mathbf1_{\lbrace\tau<1000\rbrace}].
# $$
#
# The reserve-penalty run further deducts the log penalty on the positive post-action reserve. Evaluating with 90% of the initial battery changes the crossing event while retaining the actor's nominal reserve observation.

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from barrierpg import BudgetSpec, Requirement, Stage, BudgetWrapper, evaluate, interpret
from barrierpg.callbacks import BudgetMonitor
from barrierpg.problems.energy import half_cheetah     # HalfCheetah-v5, without MuJoCo's warning about its model file

sys.path.insert(0, str(Path("..").resolve() / "campaigns"))
from records import Campaign                     # the records of the full campaign
camp = Campaign("robot")
torch.set_num_threads(1)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# %% [markdown]
# ## 1. Define the problem as a budget
#
# ### The resource: a battery added to the task
#
# The task has no battery, so a small Gymnasium wrapper adds one: it draws an efficiency for each mission and reports
# the energy of every step in `info`. The cost function then reads it.

# %%
class Battery(gym.Wrapper):
    """HalfCheetah with a battery draw in info['energy']: (idle load + squared torques) times the mission's efficiency."""

    def __init__(self, idle=0.5, efficiency_sd=0.15, mission=1000):
        super().__init__(half_cheetah(max_episode_steps=mission))
        self.idle, self.sd = idle, efficiency_sd
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng([int(seed), 0x62617474])
        self.efficiency = float(np.exp(self.sd * self.rng.standard_normal() - 0.5 * self.sd ** 2))   # mean one
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        info = dict(info, energy=self.efficiency * (self.idle + float(a @ a)), efficiency=self.efficiency)
        return obs, reward, terminated, truncated, info


def energy_used(obs, action, next_obs, reward, info):
    return [info["energy"]]

# %% [markdown]
# ### The budget, the requirement and the price
#
# * `budgets`: a battery of 2,500 units. A gait trained without a battery uses about 3,100 per mission (section 3).
# * `horizon`: the mission, 1000 steps; `beta = 0.999`, so that a late stranding is not cheap.
# * The historical campaign uses 20 and 21 as nominal reward scales. The task's velocity reward has no
#   demonstrated bound at those values, so the tutorial uses direct failure validation.
# * The requirement is stranding in at most 5% of missions. Split the final validation error between two candidates.
# * The price: `Stage(m=3, clip_utility=False)` prices only failure and preserves rewards before stranding.

# %%
spec = BudgetSpec(budgets=[2500.0], beta=0.999, ubar=20.0, u_minus=21.0, horizon=1000)
req = Requirement(T=spec.horizon - 1, alpha=0.05, zeta=0.05, family=2)
stage = Stage(m=3.0, clip_utility=False)

# %% [markdown]
# ## 2. What the policy must observe
#
# The policy has to pace the battery, and pacing needs two inputs: how much charge is left and how much of the mission
# is left. The wrapper appends the remaining charge, and with `observe_time=True` also the elapsed fraction of the
# mission; the task's own observation has no clock.
#
# **Theory.** The results are stated for policies that act on the state of the problem. With a fixed horizon, the time
# left is part of that state: whether a given charge suffices depends on it. A policy that cannot see the time solves
# a different, harder problem, and no price makes up for it.

# %%
def make_env(stage, observe_time=True, **options):
    return BudgetWrapper(Battery(), energy_used, spec, stage, observe_time=observe_time, **options)


print(Campaign.show(camp.table(["no-price-no-clock", "no-price", "price-3-no-clock", "price-3"], extras=("distance",)), extras=("distance",)))

# %% [markdown]
# The campaign trained the robot with and without the clock. Without it, the robot is stranded in more than half of its
# missions when there is no price, and a price helps only in some seeds. With the clock, stranding without a price falls
# to about a fifth. With a fixed horizon, observe the time.
#
# ## 3. Check the wrapped task, and train
#
# ### How PPO updates the actor
#
# A rollout contains actions sampled from the current actor $\pi_{w_0}$ and the rewards of the stated training objective. The critic $v(o_t)$ predicts remaining return from observation $o_t$. At a nonterminal step, the prediction residual is $e_t=r_t+\gamma_{\rm alg}v(o_{t+1})-v(o_t)$. At a true terminal step continuation is zero. Generalized advantage estimation sums these residuals using weights $(\gamma_{\rm alg}\lambda_{\rm GAE})^j$. The resulting estimate $\widehat A_t$ measures how favorable the sampled action was relative to the critic's prediction.
#
# For a trial actor $\pi_w$, compute $\ell_t=\pi_w(a_t\mid o_t)/\pi_{w_0}(a_t\mid o_t)$. The actor increases the minibatch average of $\min\lbrace\ell_t\widehat A_t,\mathrm{clip}(\ell_t,1-\epsilon,1+\epsilon)\widehat A_t\rbrace$. The critic is fitted to the estimated returns. PPO takes several minibatch updates on the rollout and then collects a fresh one. The clipping level $\epsilon$ limits the incentive for large ratio changes. The independent evaluation cells test the candidate's risk.
#
# The simulator has a physical horizon, so the algorithm uses $\gamma_{\rm alg}=\beta=0.999$.
#
# Full torque on every motor draws about 6.5 units per step, so at average efficiency the battery runs flat after about
# 380 steps; the
# failing step should receive the price's charge. The last two entries of the observation are the charge and the time.

# %%
env = make_env(stage)
obs, _ = env.reset(seed=0)
for t in range(1000):
    obs, reward, terminated, truncated, info = env.step(np.ones(6, dtype=np.float32))
    if terminated or truncated:
        break
print(f"ran flat at step {info['t']} (efficiency {info['efficiency']:.2f}): failure = {info['failure']}, reward = {reward:.0f}; "
      f"charge and time observed: {np.round(obs[-2:], 3)}")

# %% [markdown]
# The training function builds eight wrapped tasks, normalizes the observations and the reward scale, and records the
# stranding rate and the reward of every batch with `BudgetMonitor`.
#
# **Theory.** The learner's discount `gamma` must be the problem's `beta`. Normalizing observations changes the inputs
# the network sees, so the normalization statistics are part of the policy: save them with it (the campaign does) and
# apply them whenever the policy acts.

# %%
PPO_SETTINGS = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=lambda p: 3e-4 * p, gamma=spec.beta,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5))


def train(stage, steps, seed=1, **env_options):
    venv = DummyVecEnv([lambda: make_env(stage, **env_options) for _ in range(8)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=1e3, clip_reward=1e9, gamma=spec.beta)
    model = PPO("MlpPolicy", venv, seed=seed, **PPO_SETTINGS)
    monitor = BudgetMonitor(spec.beta, stage.m, horizon=spec.horizon)
    model.learn(steps, callback=[monitor])
    return model, venv, monitor

# %% [markdown]
# This figure uses all three archived runs at price 3, about two million interactions per seed. The trace
# measures outcomes during training. The evaluation table uses the final fixed policies. Run
# `python campaigns/run.py robot price-3 --seed 1` from the repository root to repeat one seed.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for rec in camp.records("price-3"):
    steps = [row["steps"] / 1e6 for row in rec["monitor"]]
    for ax, key in zip(axes, ("U", "failure_rate")):
        ax.plot(steps, [row[key] for row in rec["monitor"]], lw=0.8, label=f"seed {rec['seed']}")
for ax, ylabel in zip(axes, ("stopped utility per rollout", "stranding frequency per rollout")):
    ax.set(xlabel="million training interactions", ylabel=ylabel)
axes[0].legend()
plt.tight_layout(); plt.show()
print(camp.convergence_table(["price-3"]))

# %% [markdown]
# The reference gait was trained without an effective battery limit. Its energy use in the simulator provides
# a scale for the allowance.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for s in camp.seeds("no-battery"):
    ax.hist(camp.record("no-battery", s)["evaluation"]["energy"], bins=30, histtype="step", label=f"gait without a battery, seed {s}")
ax.axvline(spec.budgets[0], color="C3", ls="--", label="the battery")
ax.set_xlabel("energy used in a mission"); ax.set_ylabel("missions"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. The full campaign: prices

# %%
prices = ["no-price", "price-1", "price-3", "price-10"]
print(Campaign.show(camp.table(prices, extras=("distance", "energy")), extras=("distance", "energy")))

print("\nConvergence at the end of training (the toolbox's check on each run's record):")
print(camp.convergence_table(prices + ["barrier"]))

# %% [markdown]
# Stranding falls steadily with the price, and so does the distance covered: pacing the battery means running slower.
# At price 10, the observed stranding frequency is below 5% for each training seed. Simultaneous validation depends on the confidence endpoints. The table compares the final policies at the common training length.
#
# The observed stranding rates compare the fitted candidates at the tested prices and horizon. Use the upper confidence bound for the deployment tolerance. For exact failure-only optimization in the continuing model, the monotonicity result concerns geometric risk.
# ## 5. A reserve against a weaker battery
#
# Every configuration was also evaluated with a battery that holds 10% less than the robot believes: the wrapper's
# `true_budget_scale=0.9` judges failure against the smaller battery while the robot sees the nominal charge. The
# campaign also trained an interior barrier at the price 3, `Stage(m=3, theta=0.5, clip_utility=False)`, which charges
# a small amount every step that grows as the charge falls.
#
# A failure-only price deducts utility at a crossing. A positive interior barrier can also deduct utility at a small positive reserve. Compare the resulting reserve distributions with the measured stranding rates. If the reduced allowance is known in advance, it can be used as the training budget.

# %%
for cfg in ["no-price", "price-1", "price-3", "price-10", "barrier"]:
    nominal = [np.mean(r["evaluation"]["failed"]) for r in camp.records(cfg)]
    weak = [np.mean(r["extra_evaluations"][0]["episodes"]["failed"]) for r in camp.records(cfg)]
    dist = [np.mean(r["evaluation"]["distance"]) for r in camp.records(cfg)]
    print(f"{cfg:10s} stranded: nominal battery {np.mean(nominal):6.1%}, battery 10% weaker {np.mean(weak):6.1%};  distance {np.mean(dist):6.0f}")

# %% [markdown]
# With a battery that holds 10% less, every policy is stranded more often. A policy that meets the tolerance with room
# to spare (the price of 10) keeps a reserve under the weaker battery as well. The interior barrier at a lower price (3)
# lowered stranding in two of three seeds, under both batteries, but its results varied widely across seeds; on this
# problem a higher price was the more reliable way to a reserve.
#
# ## 6. Validate and report
#
# We validate the candidate with the least stranding on its first evaluation, on 500 fresh missions, as a stochastic
# policy and as its mean action. `camp.policy` applies the observation statistics saved with it.
# The final episodes are fresh, so the earlier selection leaves two candidates in the validation family.

# %%
cands = [(c, s) for c in prices + ["barrier"] for s in camp.seeds(c)]
best = min(cands, key=lambda cs: np.mean(camp.record(*cs)["evaluation"]["failed"]))
print("candidate:", best, camp.record(*best)["note"])
best_stage = camp.stage(*best)
for det in (False, True):
    e = evaluate(lambda: make_env(best_stage), camp.policy(*best, deterministic=det), 500, spec=spec, stage=best_stage,
                 T=req.T, seed=555)
    name = f"the {'mean action of the' if det else 'stochastic'} policy {best[0]}, seed {best[1]}"
    print()
    report = interpret(spec, req, best_stage, e, candidate_name=name, include_physical_value=False)
    print(report.summary(resource="charge", period="steps"))

# %% [markdown]
# The reported verdicts apply to the two tested action rules. If you deploy the mean action, its own validation counts.
#
# ## Regularizers
#
# The robot learns from scratch, and nothing trusted is available. The gait trained without a battery is a policy the new
# requirement rejects: it uses more energy than the battery holds, and anchoring to it would keep that, as the lander's
# parameter anchor kept its controller's flaw. So the campaign used no regularizer. If you have a reference policy you
# trust (a gait that respects the battery, say), the code of notebook 1 applies unchanged: start from it, and add
# `ParameterAnchor(model, kappa_p).install()` or an `ActionAnchor(eta, log_reference)` callback to `train`.
#
# ## Your own task
#
# The cells to change for a task of your own: the wrapper that adds the resource to the task (`Battery`), the cost
# function (`energy_used`), `spec`, `req` and `stage`, and `make_env`, keeping `observe_time=True` when the horizon is
# fixed and the task does not observe the time. `python campaigns/run.py robot price-3 --seed 4` reruns one
# configuration of the campaign.
