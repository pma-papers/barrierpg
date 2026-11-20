"""Train and validate a policy under a fixed fuel price in a Gymnasium task.

The example uses LunarLanderContinuous-v3 with a 24-unit fuel budget. It
defines consumption from the engine commands, appends reserve and time to
the observation, trains PPO and evaluates stochastic and mean-action
candidates with a shared validation error budget. The default training
length is illustrative. The final test estimates fuel failure, while
crashes retain their original terminal reward.

For a different task, replace make_task, resource_used, spec, req and
stage. The optional initial controller must use the same observation
and action interfaces. Install the RL and Box2D extras, then run
    python examples/your_problem.py
"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from barrierpg import BudgetSpec, Requirement, Stage, BudgetWrapper, evaluate, interpret, plan
from barrierpg.callbacks import ActionAnchor, BudgetMonitor, ParameterAnchor

STEPS = 300_000          # illustrative PPO training length

# --- YOUR PROBLEM: the task, the resource one step consumes, the budget, the requirement, the price ---------------------

def make_task():
    return gym.make("LunarLanderContinuous-v3")


def resource_used(obs, action, next_obs, reward, info):
    """Nonnegative consumption of each budget in one step (here: fuel). For a resource that is also replenished
    (cash, a battery that recharges), return the signed use and wrap this function in barrierpg.RunningMaxCost."""
    main, side = float(action[0]), float(action[1])
    main_power = (min(max(main, 0.0), 1.0) + 1.0) / 2.0 if main > 0.0 else 0.0
    side_power = min(max(abs(side), 0.5), 1.0) if abs(side) > 0.5 else 0.0
    return [0.30 * main_power + 0.03 * side_power]


spec = BudgetSpec(budgets=[24.0],        # the size of each budget
                  beta=0.999,            # the discount of the task's reward; the learner must use the same
                  ubar=100.0,            # nominal reward scale here; establish a true bound before using value certificates
                  u_minus=100.0,         # nominal magnitude of a loss
                  horizon=1000)          # the episode length if it is fixed, else None
req = Requirement(T=spec.horizon - 1,    # run out within T + 1 steps ...
                  alpha=0.05,            # ... with probability at most alpha ...
                  zeta=0.05,             # ... shown at confidence 1 - zeta ...
                  family=2)              # ... with the error shared between the two final action rules
stage = Stage(m=0.5,                     # the price of failure risk, per step from the failure to the end of the horizon;
              clip_utility=False)        # or Stage(m=0, penalty="multiplier", multiplier=F, clip_utility=False) for a
                                         # known lump-sum cost F; theta=0 retains all pre-failure task utility
OBSERVE_TIME = True                      # Include elapsed time for the fixed-horizon task.


def make_env(stage, **options):
    return BudgetWrapper(make_task(), resource_used, spec, stage, observe_time=OBSERVE_TIME, **options)

# Optional references for the training objective.
# START_FROM initializes the actor from a compatible PPO checkpoint.
# ANCHOR penalizes departures from the checkpoint parameters through a
# decoupled update. ACTION_ANCHOR penalizes departures from the reference
# distribution through an action term in the rollout reward.
# Evaluate the reference under the new budget before choosing these terms.
# To initialize from a rule, fit the mean on representative observations,
# specify the action dispersion, and evaluate the fitted stochastic actor.
START_FROM = None        # e.g. "controller.zip"
ANCHOR = None            # e.g. 1.0
ACTION_ANCHOR = None     # e.g. 0.05

# --- 1. check the wrapped environment ----------------------------------------------------------------------------------

env = make_env(stage)
obs, _ = env.reset(seed=0)
env.action_space.seed(0)
for t in range(spec.horizon or 10_000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
print(f"one episode of random actions: ended at step {info['t']}, failure = {info['failure']}, "
      f"budget left {info['reserve']}, last reward {reward:.1f}")
env.close()

# --- 2. plan: another method?, validation size, a check of the price against what an episode earns ---------------------

print(plan(spec, req, stage=stage, failure_permissible_at_deployment=True, violations_during_training_matter=False,
           loss_grows_with_excess_consumption=False, model_known_and_small=False, episode_earnings=200.0).explain())

# --- 3. train, watching the failure rate and the reward per batch -------------------------------------------------------

venv = VecNormalize(DummyVecEnv([lambda: make_env(stage) for _ in range(16)]), norm_obs=False, norm_reward=True,
                    clip_reward=1e9, gamma=spec.beta)
model = PPO("MlpPolicy", venv, gamma=spec.beta, ent_coef=0.0, n_steps=1024, batch_size=64, n_epochs=4,
            learning_rate=lambda p: 3e-4 * p, seed=1, device="cpu", verbose=0)
# the monitor records every batch and checks convergence every 10 batches (stop_when_converged=True would stop there)
monitor = BudgetMonitor(spec.beta, stage.m, horizon=spec.horizon, check_every=10, verbose=1)
callbacks = [monitor]
if START_FROM:
    start = PPO.load(START_FROM, device="cpu", custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
    model.policy.load_state_dict(start.policy.state_dict())
    if ANCHOR:
        anchor = ParameterAnchor(model, ANCHOR).install()
    if ACTION_ANCHOR:
        def log_start(obs, act):
            with torch.no_grad():
                return start.policy.evaluate_actions(torch.as_tensor(obs), torch.as_tensor(act))[1].numpy()
        callbacks.insert(0, ActionAnchor(ACTION_ANCHOR, log_start))
model.learn(STEPS, callback=callbacks)
training = monitor.convergence()
print(training.explain())                # a short run will not have converged: train longer before deploying

# --- 4. validate the policy you deploy, and report ----------------------------------------------------------------------

for deterministic in (False, True):          # the stochastic policy and its mean action are different candidates
    ev = evaluate(lambda: make_env(stage), lambda o: model.predict(o, deterministic=deterministic)[0], 500, spec=spec,
                  stage=stage, T=req.T, seed=1000)
    name = "the mean action" if deterministic else "the stochastic policy"
    print()
    # Use the direct failure test. The +/-100 terminal rewards alone do not bound the shaped rewards.
    report = interpret(spec, req, stage, ev, candidate_name=name, training=training, include_physical_value=False)
    print(report.summary(resource="fuel", period="steps"))
venv.close()
