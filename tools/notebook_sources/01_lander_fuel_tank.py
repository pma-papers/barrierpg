# %% [markdown]
# # Fine-tuning a lunar lander under a fuel budget
#
# We start with an existing PPO controller for Gymnasium's LunarLanderContinuous and impose an allowance of 24 units of fuel. The controller observes the normalized fuel reserve along with the lander observation. Fuel failure occurs at the first action after which the reserve is negative. A crash is a separate terminal outcome in this archived experiment and carries the task's original reward.
#
# We first evaluate the existing controller with the new tank. We then continue training at several failure prices and compare references in action and parameter space. The validation step measures fuel failure through the specified horizon. Landing and crash frequencies are reported separately because the training price applies to fuel failure in this task.
#
# The short live runs illustrate the procedure. The comparison uses the archived runs with three million frames per fine-tuning run and three seeds. The paper's current landing task pays revenue for a successful landing and includes loss of the lander in its failure event. [Current landing example](../README.md#train-on-a-budget-constrained-task) · [Practitioner guide](../docs/practitioner-guide.md)
#
# The notebook requires Stable-Baselines3 and Gymnasium with Box2D.
#
# ### Policy optimization with the fuel limit
#
# At frame $t$, the policy observes the lander observation and normalized fuel reserve $z_t/24$. The archived configuration omits an explicit clock, so the actor class uses that restricted observation. Its action $a_t\in[-1,1]^2$ supplies the main and side engine commands. The simulator maps them to powers $p_t^{\rm main}$ and $p_t^{\rm side}$. The consumption and reserve are $c_t=0.30p_t^{\rm main}+0.03p_t^{\rm side}$ and $z_{t+1}=z_t-c_t$, with $z_0=24$. Write $\tau=\inf\{t:z_{t+1}<0\}$. An ordinary landing or crash terminates the episode and contributes its normal task reward if fuel failure has not occurred on the same action.
#
# Let $U_H(\pi)=\mathbb E_\pi[\sum_{0\le t<H,\ t<\tau}\beta^t u_t]$ be stopped Gymnasium utility, with $H=1000$ and $\beta=0.999$. Utility after genuine task termination is zero. Let $\Pi$ be the class represented by the actor. The deployment problem is
#
# $$
# \begin{aligned}
# \underset{\pi\in\Pi}{\operatorname{maximize}}\quad & U_H(\pi)\\
# \text{subject to}\quad & \Pr_\pi(\tau\le999)\le0.05.
# \end{aligned}
# $$
#
# In training, a failure at time $t$ is charged $F_t=0.5\sum_{j=0}^{H-1-t}\beta^j$. PPO searches for a solution to the priced objective.
#
# $$
# \underset{\pi\in\Pi}{\operatorname{maximize}}\quad U_H(\pi)-\mathbb E_\pi[\beta^\tau F_\tau\mathbf1_{\{\tau<H\}}].
# $$
#
# The price assigns a greater charge to early fuel failure. The deployment constraint counts every crossing within the mission once. The validation cell tests that event and records crashes and landings separately.

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from barrierpg import BudgetSpec, Requirement, Stage, BudgetWrapper, evaluate, interpret, plan
from barrierpg.callbacks import BudgetMonitor, ParameterAnchor, ActionAnchor
from barrierpg.spec import failure_charge

sys.path.insert(0, str(Path("..").resolve() / "campaigns"))
from records import Campaign                     # the records of the full campaign
camp = Campaign("lander")
torch.set_num_threads(1)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# %% [markdown]
# ## 1. Define the problem as a budget
#
# ### The resource
#
# Write a function that says how much of the resource one step consumes. It receives what the environment returns and
# returns one nonnegative number per budget. The lander's task already defines its fuel use: 0.30 per unit of
# main-engine power and 0.03 per unit of side-engine power.

# %%
def fuel_burned(obs, action, next_obs, reward, info):
    """Fuel burned in one frame, by the task's own engine-power rule."""
    main, side = float(action[0]), float(action[1])
    main_power = (min(max(main, 0.0), 1.0) + 1.0) / 2.0 if main > 0.0 else 0.0
    side_power = min(max(abs(side), 0.5), 1.0) if abs(side) > 0.5 else 0.0
    return [0.30 * main_power + 0.03 * side_power]

# %% [markdown]
# ### The budget and the requirement
#
# `BudgetSpec` describes the budget problem; `Requirement` states what the deployed policy must satisfy.
#
# * `budgets`: the tank, 24 units.
# * `horizon`: an episode lasts at most 1000 frames. The price is charged from a failure to the end of the horizon.
# * `beta`: the discount of the task's reward. With a horizon of 1000 frames, a discount close to one keeps a late
#   failure from looking cheap.
# * The terminal rewards are ±100. These values are retained in the historical campaign spec. The tutorial uses
#   direct failure validation because the terminal rewards alone do not bound the shaped step rewards.
# * The requirement is at most 5% fuel exhaustion within 1000 frames, with the validation error split
#   between the stochastic and mean-action policies on the final evaluation sample.
#
# A verified upper bound on step utility can be conservative. Increasing that bound can make a value certificate uninformative. The tolerance is the maximum failure probability and the validation error governs the confidence bound.

# %%
spec = BudgetSpec(budgets=[24.0], beta=0.999, ubar=100.0, u_minus=100.0, horizon=1000)
req = Requirement(T=spec.horizon - 1, alpha=0.05, zeta=0.05, family=2)

# %% [markdown]
# ### The price
#
# A `Stage` sets how a failure is charged. `Stage(m=0.5)` charges a price of 0.5 per frame from the failure to the end
# of the horizon. This stage has no interior barrier, so it preserves the task's rewards before fuel exhaustion.
#
# **Theory.** The finite-horizon price charges earlier failures more. The resulting objective weights failure time
# as well as its frequency. The evaluation below measures the probability of any fuel exhaustion within the mission.

# %%
stage = Stage(m=0.5, clip_utility=False)
for t in (0, 500, 900):
    print(f"a failure at frame {t} is charged {failure_charge(stage.m, spec.beta, t, spec.horizon):6.1f}")

# %% [markdown]
# ### The wrapper
#
# `BudgetWrapper` puts these together around the task. It adds the remaining fuel, as a fraction of the tank, to the
# observation; the policy needs it. It ends the episode when the tank runs dry and charges the price there. A separate Gymnasium wrapper records landing and crash outcomes.

# %%
class LanderOutcome(gym.Wrapper):
    """Adds info['landed'] and info['crashed'] at the last frame, read from the simulator's state."""

    def __init__(self):
        super().__init__(gym.make("LunarLanderContinuous-v3"))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if terminated:
            core = self.env.unwrapped
            info["crashed"] = bool(core.game_over or abs(obs[0]) >= 1.0)
            info["landed"] = bool(not info["crashed"] and not core.lander.awake)
        return obs, reward, terminated, truncated, info


def make_env(stage, **options):
    return BudgetWrapper(LanderOutcome(), fuel_burned, spec, stage, **options)

# %% [markdown]
# ## 2. Check the wrapped environment, and plan
#
# Before training, check that the wrapper does what you mean. With the main engine at full power the lander burns
# 0.30 per frame, so a 24-unit tank runs dry after 80 frames; the failing frame should receive the price's charge.

# %%
env = make_env(stage)
obs, _ = env.reset(seed=0)
print("observation:", obs.shape, "the last entry is the fuel gauge:", obs[-1])
for t in range(1000):
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    if terminated or truncated:
        break
print(f"ran dry at frame {info['t']}: failure = {info['failure']}, reward = {reward:.1f}, "
      f"the price's charge at that frame = {-failure_charge(stage.m, spec.beta, info['t'], spec.horizon):.1f}")

# %% [markdown]
# `plan` states, before anything is trained, whether another method is the better choice, how many validation
# episodes are needed, and other checks.
#
# **Theory.** `plan` also compares what a failure is charged with what an episode earns: a price that charges far more
# than an episode earns makes a learner give up earning in order not to fail.

# %%
p = plan(spec, req, stage=stage, failure_permissible_at_deployment=True, violations_during_training_matter=False,
         loss_grows_with_excess_consumption=False, model_known_and_small=False, episode_earnings=240.0)
print(p.explain())

# %% [markdown]
# ## 3. Train, starting from the controller in use
#
# ### How PPO updates the actor
#
# A rollout contains actions sampled from the current actor $\pi_{w_0}$ and the rewards of the stated training objective. The critic $v(o_t)$ predicts remaining return from observation $o_t$. At a nonterminal step, the prediction residual is $e_t=r_t+\gamma_{\rm alg}v(o_{t+1})-v(o_t)$. At a true terminal step continuation is zero. Generalized advantage estimation sums these residuals using weights $(\gamma_{\rm alg}\lambda_{\rm GAE})^j$. The resulting estimate $\widehat A_t$ measures how favorable the sampled action was relative to the critic's prediction.
#
# For a trial actor $\pi_w$, compute $L_t=\pi_w(a_t\mid o_t)/\pi_{w_0}(a_t\mid o_t)$. The actor increases the minibatch average of $\min\{L_t\widehat A_t,\operatorname{clip}(L_t,1-\epsilon,1+\epsilon)\widehat A_t\}$. The critic is fitted to the estimated returns. PPO takes several minibatch updates on the rollout and then collects a fresh one. The clipping level $\epsilon$ limits the incentive for large ratio changes. The independent evaluation cells test the candidate's risk.
#
# The simulator has a physical horizon, so the algorithm uses $\gamma_{\rm alg}=\beta=0.999$.
#
# ### The controller in use
#
# The controller was trained without a limit. Its fuel gauge was there but never ran dry: `true_budget_scale` judges
# failure against a far larger tank than the one the policy observes. The cell below shows how; it takes about ten
# minutes, so it is switched off and the campaign's controller is loaded instead.
#
# **Theory.** The learner's discount `gamma` must be the problem's `beta`: the price is defined in that discounted
# objective. An entropy bonus (`ent_coef > 0`) is action regularization toward the uniform policy, which biases the
# learned policy; we set it to zero.

# %%
PPO_SETTINGS = dict(n_steps=1024, batch_size=64, n_epochs=4, learning_rate=lambda p: 3e-4 * p, gamma=spec.beta,
                    gae_lambda=0.95, ent_coef=0.0, device="cpu", verbose=0)


def vec_env(stage, seed, n_envs=16, **options):
    venv = DummyVecEnv([lambda: make_env(stage, **options) for _ in range(n_envs)])
    venv.seed(seed)
    return venv


TRAIN_THE_CONTROLLER = False          # about ten minutes on one core
if TRAIN_THE_CONTROLLER:
    controller = PPO("MlpPolicy", vec_env(Stage(m=0.0, clip_utility=False), seed=2, true_budget_scale=1e6), seed=2,
                     **PPO_SETTINGS)
    controller.learn(5_000_000)
else:
    controller = camp.model("controller", seed=2)               # the campaign's controller (a stable_baselines3 PPO)

# %% [markdown]
# How much fuel does the controller burn, and how often would it run dry under the new tank? The campaign trained three
# controllers (seeds 1 to 3) and evaluated them without a limit and with the tank.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for s in camp.seeds("controller"):
    ax.hist(camp.record("controller", s)["evaluation"]["fuel"], bins=40, histtype="step", label=f"controller, seed {s}")
ax.axvline(spec.budgets[0], color="C3", ls="--", label="the new tank")
ax.set_xlabel("fuel burned in a landing (no limit)"); ax.set_ylabel("landings"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
for s in camp.seeds("controller"):
    e = camp.record("controller", s)["extra_evaluations"][0]["episodes"]
    print(f"controller seed {s} under the tank: ran dry {np.mean(e['failed']):6.1%}, landed {np.mean(e['landed']):6.1%}, "
          f"crashed {np.mean(e['crashed']):5.1%}")

# %% [markdown]
# Training a new controller with the tank from scratch is no way out: random exploration empties the tank long before it
# learns to land.

# %%
per_frame = np.mean([fuel_burned(None, a, None, 0.0, {})[0] for a in np.random.default_rng(0).uniform(-1, 1, (100_000, 2))])
print(f"random actions burn {per_frame:.3f} per frame: the tank is empty after about {spec.budgets[0] / per_frame:.0f} frames")

# %% [markdown]
# ### Continue training with the tank
#
# The training function builds the wrapped environments, starts PPO from the controller's weights, and records the
# failure rate and the reward of every batch with `BudgetMonitor`. Here it runs for 300,000 frames; the campaign ran
# the same function for 3 million.

# %%
def train(stage, steps, start, seed=1, anchor=None, action_anchor=None):
    model = PPO("MlpPolicy", vec_env(stage, seed), seed=seed, **PPO_SETTINGS)
    model.policy.load_state_dict(start.policy.state_dict())             # start from the controller in use
    monitor = BudgetMonitor(spec.beta, stage.m, horizon=spec.horizon)   # records every batch; checks convergence
    callbacks, regularizers = [monitor], {}
    if anchor:                                                          # weight decay toward the starting weights
        regularizers["anchor"] = ParameterAnchor(model, anchor, decoupled=False).install()   # the campaign's form, see below
    if action_anchor:                                                   # charge departures from the start's actions
        def log_start(obs, act):
            with torch.no_grad():
                return start.policy.evaluate_actions(torch.as_tensor(obs), torch.as_tensor(act))[1].numpy()
        regularizers["action anchor"] = ActionAnchor(action_anchor, log_start)
        callbacks.insert(0, regularizers["action anchor"])
    model.learn(steps, callback=callbacks)
    return model, monitor, regularizers


def landings(model, stage, n=200, seed=5):
    """Ran dry, landed and crashed, over n fresh episodes of the stochastic policy."""
    env, counts = make_env(stage), dict(ran_dry=0, landed=0, crashed=0)
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        while True:
            obs, _, terminated, truncated, info = env.step(model.predict(obs)[0])
            if terminated or truncated:
                break
        counts["ran_dry"] += info["failure"]; counts["landed"] += info.get("landed", False); counts["crashed"] += info.get("crashed", False)
    return {k: v / n for k, v in counts.items()}


tank_alone = Stage(m=0.0, clip_utility=False)
model, monitor, _ = train(tank_alone, 300_000, controller)
monitor.plot(title="the tank alone, 300,000 frames"); plt.show()
print(monitor.convergence().explain())
print("fresh landings:", {k: f"{v:.1%}" for k, v in landings(model, tank_alone).items()})

# %% [markdown]
# This run is short, and the convergence check says so: the lander still runs dry about as often as the controller
# did. `BudgetMonitor.convergence()` compares the last tenth of the training batches with the tenth
# before; `BudgetMonitor(check_every=k, stop_when_converged=True)` checks during training and stops once the run has
# converged. The campaign continued every controller for 3 million frames. Its learning curves, recorded by the same
# monitor, show when the lander learns to save fuel.

# %%
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for r in camp.records("tank-no-price"):
    ax.plot([t["steps"] / 1e6 for t in r["monitor"]], [t["failure_rate"] for t in r["monitor"]], lw=0.7, label=f"seed {r['seed']}")
ax.axhline(req.alpha, color="C3", ls=":", lw=1)
ax.set_xlabel("million frames"); ax.set_ylabel("ran dry (per batch)"); ax.set_title("the tank alone, the full campaign", fontsize=9)
ax.legend(fontsize=7); plt.tight_layout(); plt.show()

# %% [markdown]
# Running dry ends the episode before the landing bonus, so the task's own reward already punishes it, and the wrapper
# only had to make the tank part of the problem; given enough frames, the lander learns to land within it. Check the
# learning curves before comparing configurations: a run stopped early tells you about its length, not about its
# configuration. The campaign ran the tank alone and prices from 0.25 to 2, all for 3 million frames.
#
# ## 4. The full campaign: prices

# %%
prices = ["tank-no-price", "price-0.25", "price-0.5", "price-1", "price-2"]
print(Campaign.show(camp.table(prices, extras=("landed", "crashed"))))

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
x = np.arange(len(prices))
for k, key in enumerate(["failed", "crashed"]):
    for s in (1, 2, 3):
        vals = [np.mean(camp.record(c, s)["evaluation"][key]) if s in camp.seeds(c) else np.nan for c in prices]
        ax[k].plot(x, vals, "o-", lw=0.8, label=f"seed {s}")
    ax[k].set_xticks(x, [c.replace("tank-", "") for c in prices], fontsize=8)
    ax[k].set_ylabel({"failed": "ran dry", "crashed": "crashed"}[key]); ax[k].legend(fontsize=7)
ax[0].axhline(req.alpha, color="C3", ls=":", lw=1)
plt.tight_layout(); plt.show()
print("Convergence at the end of training (the toolbox's check on each run's record):")
print(camp.convergence_table(prices))

# %% [markdown]
# Most runs had converged by the end of training. Where one had not, its policy is the policy at the end of the run:
# it is validated as such, and a longer run might do better.
#
# Across these runs, higher prices changed the measured fuel-failure frequency little. Several fitted controllers crashed more often. Once running dry costs more than a crash, the learner
# sometimes chooses the crash.
#
# Fuel exhaustion and a crash have different consequences in the archived reward. A price change can therefore alter the mix of observed terminal outcomes. Compare fuel failure, crashes and successful landings on the same horizon when interpreting these runs.

# ## 5. Regularizers that use the controller in use
#
# The controller in use is the policy one would accept if training gained nothing, which makes it a natural reference.
# The `train` function above adds either regularizer with one argument; the campaign ran both at the price 0.5:
#
# * `anchor=kappa_p`: a **parameter anchor**, which pulls the network's weights back toward the controller's (strengths
#   from 0.01 to 10);
# * `action_anchor=eta`: an **action anchor**, which charges, at every frame, how far the policy's action distribution
#   departs from the controller's (weight 0.05).
#
# **Theory.** A parameter anchor biases the learned policy toward the prior policy, the policy at the anchor's weights,
# and the learner inherits that policy's risk. An action anchor keeps the policy's failure risk close to the
# reference's, by an amount set by how far the two action distributions are apart.
#
# Here are the three, trained live for 300,000 frames each at the price 0.5. `ParameterAnchor.distance()` measures how
# far the weights moved from the controller's, and `ActionAnchor.mean_log_ratio` how far the actions moved.

# %%
priced = Stage(m=0.5, clip_utility=False)
for name, options in {"no regularizer": {}, "parameter anchor, kappa_p = 1": dict(anchor=1.0),
                      "action anchor, eta = 0.05": dict(action_anchor=0.05)}.items():
    m_, mon, regs_ = train(priced, 300_000, controller, **options)
    moved = ""
    if "anchor" in regs_:
        moved = f"; weights moved {regs_['anchor'].distance():.3f}"
    if "action anchor" in regs_:
        moved = f"; mean log ratio of actions {regs_['action anchor'].mean_log_ratio:.3f}"
    outcome = landings(m_, priced)
    print(f"{name:30s} ran dry {outcome['ran_dry']:6.1%}, landed {outcome['landed']:6.1%}, crashed {outcome['crashed']:5.1%}"
          f"{moved}; training {mon.convergence().status}")

# %% [markdown]
# These are short runs, too short for the convergence check to judge, so the numbers show how to use the regularizers
# rather than what they do after full training. Two limits of the check matter here as everywhere. A run can stop
# changing in a bad place, and a learning rate that decays to zero makes any run stop changing at its end. Converged
# means that the run stopped changing, not that its policy is good; the validation decides that. The full campaign
# trained each for 3 million frames:

# %%
regs = ["price-0.5", "anchor-0.01", "anchor-0.1", "anchor-1", "anchor-10", "action-anchor"]
print(Campaign.show(camp.table(regs, extras=("landed", "crashed"))))
for cfg in ["anchor-0.01", "anchor-0.1", "anchor-1", "anchor-10"]:
    print(f"{cfg}: weight distance from the controller at the end {np.round([r['anchor_distance'] for r in camp.records(cfg)], 3)}")
print("\nConvergence at the end of training:")
print(camp.convergence_table(regs))

# %% [markdown]
# The initial controller often fails the tighter fuel requirement. At each tested coupled-anchor strength, the fitted policies continued to show high fuel-failure frequencies. The comparison uses the controller induced by the reference weights as a separate candidate.
#
# These runs, and the short ones above, use the coupled form of the anchor (`decoupled=False`): the drift is added to
# the policy gradient before Adam's step, the form of the paper's runs. Adam divides every gradient coordinate by its
# running root-mean-square, so the drift's step is about the learning rate whatever the distance to the anchor and
# whatever the strength, and the strength does not read as a curvature. The toolbox's default is the decoupled form,
# `theta <- theta - lr kappa_p (theta - theta_0)` applied to the weights after the step, under which every weight
# contracts toward the anchor by `1 - lr kappa_p` per step and the strength keeps its meaning (the paper's guidance;
# `ParameterAnchor`'s docstring). Whether the anchored lander still runs dry under the decoupled form at these
# strengths is not tested here.
#
# The action anchor holds the behavior near the controller and gives way where the tank demands a change. With it the
# lander learned to save fuel as well as without it; its effect on crashes was mixed across seeds.
#
# When the policy in use violates the new limit: start from it, and if you regularize toward it, regularize its
# behavior, not its weights.
#
# ## 6. Validate the policy to deploy, and report
#
# Use the archived evaluations for candidate selection. Take the highest landing rate among policies whose
# estimated failure rate is below the tolerance. If none qualify, take the lowest estimated failure rate.
# Test the selected policy and its mean-action version on 2,000 fresh episodes each.
#
# **Theory.** The exact binomial test holds for any fixed policy, however it was trained; it is the deployment
# decision. Fresh final data permit selection on any number of earlier runs. The final test splits the error over
# the two candidate policies evaluated for deployment.

# %%
cands = [(c, s) for c in prices + regs[1:] for s in camp.seeds(c)]
ok = [cs for cs in cands if np.mean(camp.record(*cs)["evaluation"]["failed"]) < req.alpha]
best = (max(ok, key=lambda cs: np.mean(camp.record(*cs)["evaluation"]["landed"])) if ok else
        min(cands, key=lambda cs: np.mean(camp.record(*cs)["evaluation"]["failed"])))
print("candidate:", best, camp.record(*best)["note"])
best_stage = camp.stage(*best)
for det in (False, True):
    e = evaluate(lambda: make_env(best_stage), camp.policy(*best, deterministic=det), 2000, spec=spec, stage=best_stage,
                 T=req.T, seed=777)
    name = f"the {'mean action of the' if det else 'stochastic'} policy {best[0]}, seed {best[1]}"
    print()
    report = interpret(spec, req, best_stage, e, candidate_name=name, training=camp.convergence(*best),
                       include_physical_value=False)
    print(report.summary(resource="fuel", period="frames"))

# %% [markdown]
# The report gives a separate result for each action rule. Use the result for the rule that will actually be deployed.
#
# ## Your own task
#
# To add a budget to a task you already train, change these cells:
#
# 1. `fuel_burned`: the resource one step consumes (nonnegative; for a resource that is also replenished, wrap the
#    signed use in `barrierpg.RunningMaxCost`);
# 2. `spec` and `req`: the budget, the horizon, the discount, bounds on the reward, and the requirement;
# 3. `stage`: the price, or a lump sum if failure has a known cost (`Stage(penalty="multiplier", multiplier=F)`);
# 4. `make_env`: your Gymnasium task inside `BudgetWrapper` (add `observe_time=True` if the horizon is fixed and the
#    task does not observe the time);
# 5. `PPO_SETTINGS` and `train`: your learner's settings, with `gamma = spec.beta`.
#
# Then check the wrapped environment, plan, train at a few prices, and validate the policy you deploy. The campaign's
# configurations are in `campaigns/problems.py`; `python campaigns/run.py lander price-0.5 --seed 4` reruns one.
