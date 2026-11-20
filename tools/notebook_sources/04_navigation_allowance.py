# %% [markdown]
# # Mean damage and failure probability in navigation
#
# Safety-Gymnasium's PointGoal1 task rewards movement toward successive goals. A step in a hazard incurs one unit of damage. We use the same numerical allowance of 25 to distinguish two requirements. One caps mean damage per episode. The other limits the probability that cumulative damage exceeds 25 during the 1,000-action episode. The measured mean below a cap can coexist with frequent crossings.
#
# In this notebook, PPO is trained on a fixed-price reward for the crossing event. We compare several prices using the saved training runs. The final binomial calculation tests the crossing probability at the stated horizon. The recorded mean damage is measured under termination at the first crossing, so it concerns that stopped process.
#
# Training and fresh simulation require a separate environment with Python 3.10, Safety-Gymnasium 1.0, Gymnasium 0.28 and MuJoCo 2.3. The published comparisons use about two million interactions for each of three training seeds per configuration. The saved evaluations can be examined without that simulator. [Practitioner guide](../docs/practitioner-guide.md) · [Formulations](../docs/formulation-comparison.md)
#
# ### Mean damage, crossing risk and the trained objective
#
# Let $c_t$ be the hazard cost reported by the simulator and $D_t=\sum_{j=0}^{t}c_j$ the accumulated damage. The navigation policy controls motion to earn progress rewards and goal bonuses. Let $V^{(H)}(\pi)$ be expected discounted utility through the original task's horizon $H=1000$. For the mean-cost formulation, $D_{999}$ is measured over the original episode, with subsequent costs zero after a genuine terminal state.
#
# The probability formulation uses budget $b=25$, reserve $z_{t+1}=25-D_t$, and failure time $\tau=\inf\lbrace t:D_t>25\rbrace$. Write $U_H(\pi)$ for expected discounted utility earned before failure. With $\Pi$ the actor class, the mean-cost problem on the original task is
#
# $$
# \begin{aligned}
# \underset{\pi\in\Pi}{\mathrm{maximize}}\quad & V^{(H)}(\pi)\\
# \text{subject to}\quad & \mathbb E_\pi[D_{999}]\le25.
# \end{aligned}
# $$
#
# The stopped task imposes a crossing-probability constraint.
#
# $$
# \begin{aligned}
# \underset{\pi\in\Pi}{\mathrm{maximize}}\quad & U_H(\pi)\\
# \text{subject to}\quad & \Pr_\pi(\tau\le999)\le0.20.
# \end{aligned}
# $$
#
# One can encode the second constraint in a mean-cost algorithm by using cost one on the first crossing and zero otherwise. Here a failure at $t$ costs $F_t=m\sum_{j=0}^{H-1-t}\beta^j$, with $\beta=0.999$. PPO trains on
#
# $$
# \underset{\pi\in\Pi}{\mathrm{maximize}}\quad U_H(\pi)-\mathbb E_\pi[\beta^\tau F_\tau\mathbf1_{\lbrace\tau<H\rbrace}].
# $$
#
# The recorded mean damage uses episodes stopped at first crossing. It concerns the stopped simulator, whereas the expected-cost problem above concerns the original episodic process.

# %%
import sys
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from barrierpg import BudgetSpec, Requirement, Stage, BudgetWrapper, interpret, plan

sys.path.insert(0, str(Path("..").resolve() / "campaigns"))
from records import Campaign                     # the records of the full campaign
camp = Campaign("navigation")
HAVE_SAFETY_GYMNASIUM = importlib.util.find_spec("safety_gymnasium") is not None
print("Safety-Gymnasium available:", HAVE_SAFETY_GYMNASIUM)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# %% [markdown]
# ## 1. Two requirements behind one limit
#
# The suite poses the task as a limit on the **expected** cost per episode: on average at most 25 units. Read as a
# damage allowance, the same limit is a **budget**: the robot can take 25 units of damage in a mission, and exhausting
# it ends the mission. The requirement is then a tolerated probability of exhausting the allowance.
#
# The two are different requirements. An expected cost of 25 allows exhausting a budget of 25 in half of the
# episodes (half at 0, half at 50). Which one the task means is a question about the application, not about the
# learner: a robot that may be damaged more on some missions as long as the average is fine is the first; a robot that
# must finish most of its missions is the second. This notebook takes the second, and tolerates exhausting the
# allowance in one mission in five.
#
# **Theory.** The method prices the event of running out, not the amount by which the limit is exceeded and not the
# average cost. If your requirement is the expected cost, a constrained-MDP method is built for it.
#
# ## 2. Define the problem as a budget, and train
#
# ### How PPO updates the actor
#
# A rollout contains actions sampled from the current actor $\pi_{w_0}$ and the rewards of the stated training objective. The critic $v(o_t)$ predicts remaining return from observation $o_t$. At a nonterminal step, the prediction residual is $e_t=r_t+\gamma_{\rm alg}v(o_{t+1})-v(o_t)$. At a true terminal step continuation is zero. Generalized advantage estimation sums these residuals using weights $(\gamma_{\rm alg}\lambda_{\rm GAE})^j$. The resulting estimate $\widehat A_t$ measures how favorable the sampled action was relative to the critic's prediction.
#
# For a trial actor $\pi_w$, compute $\ell_t=\pi_w(a_t\mid o_t)/\pi_{w_0}(a_t\mid o_t)$. The actor increases the minibatch average of $\min\lbrace\ell_t\widehat A_t,\mathrm{clip}(\ell_t,1-\epsilon,1+\epsilon)\widehat A_t\rbrace$. The critic is fitted to the estimated returns. PPO takes several minibatch updates on the rollout and then collects a fresh one. The clipping level $\epsilon$ limits the incentive for large ratio changes. The independent evaluation cells test the candidate's risk.
#
# The simulator has a physical horizon, so the algorithm uses $\gamma_{\rm alg}=\beta=0.999$.
#
# * The resource: Safety-Gymnasium returns the step's cost separately; its `SafetyGymnasium2Gymnasium` wrapper moves it
#   into `info["cost"]`, where the cost function reads it.
# * `budgets`: 25 units of damage; `horizon`: 1000 steps; `beta = 0.999`.
# * `ubar`, `u_minus`: 1.1 and 0.1 are nominal reward scales in the archived configuration.
# * The price: `Stage(m=0.03, clip_utility=False)`, one of the campaign's prices; only the failure is priced.
# * `observe_time=True`: the task's observation has no clock, and the allowance has to last for the episode.
#
# Costs come in whole units, so the allowance can reach exactly zero without being exceeded; that is not a failure.
#
# **Theory.** The learner's discount `gamma` must be the problem's `beta`. Observations are normalized, so the
# normalization statistics are part of the policy and are saved with it.

# %%
def hazard_cost(obs, action, next_obs, reward, info):
    return [float(info["cost"])]


spec = BudgetSpec(budgets=[25.0], beta=0.999, ubar=1.1, u_minus=0.1, horizon=1000)
stage = Stage(m=0.03, clip_utility=False)
PPO_SETTINGS = dict(n_steps=1024, batch_size=64, n_epochs=4, learning_rate=lambda p: 3e-4 * p, gamma=spec.beta,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64], log_std_init=0.0))

if HAVE_SAFETY_GYMNASIUM:
    import safety_gymnasium
    from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from barrierpg.callbacks import BudgetMonitor

    def make_env(stage, **options):
        task = SafetyGymnasium2Gymnasium(safety_gymnasium.make("SafetyPointGoal1-v0"))
        return BudgetWrapper(task, hazard_cost, spec, stage, observe_time=True, **options)

    def train(stage, steps, seed=1):
        venv = DummyVecEnv([lambda: make_env(stage) for _ in range(8)])
        venv.seed(seed)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=1e3, clip_reward=1e9, gamma=spec.beta)
        model = PPO("MlpPolicy", venv, seed=seed, **PPO_SETTINGS)
        monitor = BudgetMonitor(spec.beta, stage.m, horizon=spec.horizon)
        model.learn(steps, callback=[monitor])
        return model, venv, monitor

    env = make_env(stage)                                    # check the wrapped task: one episode of random actions
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for t in range(1000):
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 2))
        if terminated or truncated:
            break
    print(f"episode ended at step {info['t']}: failure = {info['failure']}, allowance left {info['reserve'][0]:.0f}")
else:
    print("Safety-Gymnasium is not installed in this environment: the code above is what the campaign ran with it.\n"
          "The rest of the notebook reads the campaign's records.")

print(camp.convergence_table(["price-0.03"]))

# %% [markdown]
# ## 3. The full campaign: prices
#
# The campaign trained the robot from scratch (2 million steps, three seeds) at prices of 0, 0.003, 0.01, 0.03 and 0.1
# per failed period through the physical horizon, and evaluated each fixed policy on 300 fresh episodes.
# The local training function can be run in the Safety-Gymnasium environment at the same budget,
# for example with `train(stage, 2_000_000, seed=1)`.

# %%
configs = ["no-price", "price-0.003", "price-0.01", "price-0.03", "price-0.1"]
print(Campaign.show(camp.table(configs, extras=("cost", "ret")), extras=("cost", "ret")))
print("\nConvergence at the end of training (the toolbox's check on each run's record):")
print(camp.convergence_table(configs))

# %% [markdown]
# Counted up to the step at which the allowance runs out (the episode ends there), the episodes of the robot trained
# without a price average below 25 units of damage, and more than half of them exhaust the allowance. A mean cost below
# the limit says little about how often the limit is exceeded: with whole units, a mean of 19 is compatible with up to
# three quarters of the episodes reaching 26. The figure places every trained policy by its mean cost (counted up to
# exhaustion) and by how often it exhausts the allowance; the dotted line is the most exhaustion a mean cost allows.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, c in enumerate(configs):
    for r in camp.records(c):
        ax.plot(np.mean(r["evaluation"]["cost"]), np.mean(r["evaluation"]["failed"]), "o", color=f"C{i}",
                label=c if r["seed"] == 1 else None)
xs = np.linspace(0, 26, 50)
ax.plot(xs, np.minimum(xs / 26, 1.0), "k:", lw=0.8, label="the most exhaustion a mean cost allows\n(every episode costs 0 or 26)")
ax.set_xlabel("mean cost per episode, counted up to exhaustion"); ax.set_ylabel("allowance exhausted"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# Exhaustion falls sharply between the prices 0.01 and 0.03, and the return falls with it: avoiding hazards means
# taking longer ways to the goals. At 0.1 the robot almost stops reaching goals. A failure at the start of an episode is
# then charged about four times what an episode earns, and the learner gives up earning in order not to fail.
#
# Changing the price alters the training reward. The fitted candidate's attained risk also depends on optimization and the policy class. Use `plan` to compare the failure charge at the start of the episode with the scale of episode earnings.

# %%
earns = np.mean([np.mean(r["evaluation"]["ret"]) for r in camp.records("no-price")])
for m in (0.03, 0.1):
    p = plan(spec, Requirement(T=spec.horizon - 1, alpha=0.2), stage=Stage(m=m, clip_utility=False), episode_earnings=earns)
    print(f"price {m}: " + ("; ".join(p.warnings) if p.warnings else "; ".join(n for n in p.notes if "earns" in n)))

# %% [markdown]
# ## 4. Validation: the direct test decides
#
# A value certificate requires a verified bound on the shaped step rewards. Here the confidence calculation uses only the first-crossing indicator.
# We validate the three policies trained at the price of 0.03,
# on the 300 fresh episodes of their records, with the error shared among the fifteen trained policies.
#
# **Theory.** For a fixed policy, zero failures in $N$ independent episodes give an upper endpoint
# $1-e^{1/N}$ at error $e$ per statement. Failure to certify does not prove a violation.

# %%
req = Requirement(T=spec.horizon - 1, alpha=0.2, zeta=0.05, family=15)
for s in camp.seeds("price-0.03"):
    rep = interpret(spec, req, camp.stage("price-0.03", s), camp.evaluation("price-0.03", s),
                    candidate_name=f"the policy at price 0.03, seed {s}", training=camp.convergence("price-0.03", s),
                    include_physical_value=False)
    print("\n".join(rep.summary(resource="the damage allowance", period="steps").split("\n")[:3]))
    print("  " + next(line for line in rep.summary().split("\n") if line.startswith("Training")))
    print(f"  (return per episode {np.mean(camp.record('price-0.03', s)['evaluation']['ret']):.2f})\n")

# %% [markdown]
# Compare the simultaneous upper endpoints above with the 20% requirement. The requirement is established when the one-sided upper confidence endpoint is at most 20%. Training seeds also differ in return at the same price.
#
# ## 5. Training without a reference policy
#
# The reported training starts from a random initialization and uses no reference-based penalty. To evaluate a new action or parameter reference, first measure the behavior of the policy it represents. Then compare the fitted policies under common task and evaluation conditions.
# ## Your own task
#
# The cells to change: the cost function (`hazard_cost`), `spec`, `stage`, and `make_env` (your task inside
# `BudgetWrapper`). In the Python environment with Safety-Gymnasium (here `~/.venvs/constrained-rl-safety310`, with the
# toolbox installed without dependencies), one configuration of the campaign reruns with
# `~/.venvs/constrained-rl-safety310/bin/python campaigns/run.py navigation price-0.03 --seed 4` (about 35 minutes).
