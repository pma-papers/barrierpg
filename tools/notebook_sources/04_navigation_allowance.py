# %% [markdown]
# # 4. A navigation robot with a damage allowance: which requirement does the task mean?
#
# In Safety-Gymnasium's PointGoal1 task a point robot earns the distance by which it closes in on a goal and a bonus
# each time it reaches one; a new goal then appears, so an episode of 1000 steps visits as many goals as the robot
# reaches. Eight hazards are scattered over the floor, and each step inside one costs a unit. The suite's customary
# limit is 25 units per episode.
#
# This notebook shows:
#
# 1. that the same limit can mean two different requirements, and why the difference matters;
# 2. what prices do to the probability of exhausting the allowance and to the return;
# 3. a price that charges more for a failure than an episode earns;
# 4. why the direct validation test, not a certificate from the value, decides here;
# 5. the regularizers when nothing trusted is available.
#
# Safety-Gymnasium needs its own Python environment (Python 3.10, Gymnasium 0.28, MuJoCo 2.3). This notebook only
# reads the records of the campaign in `campaigns/`, so it runs in the toolbox's main environment; training and new
# evaluations need the other one.

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("..").resolve() / "campaigns"))
from records import Campaign
from bpg import Requirement, Stage, interpret, plan

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
camp = Campaign("navigation")
spec = camp.problem.spec

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
# must finish its missions with probability 95% is the second. This notebook takes the second.

# %%
configs = ["unpriced", "price-0.003", "price-0.01", "price-0.03", "price-0.1"]
print(Campaign.show(camp.table(configs, extras=("cost", "ret")), extras=("cost", "ret")))

# %% [markdown]
# Counted up to the step at which the allowance runs out (the episode ends there), the episodes of the robot trained
# without a price average below 25 units of damage, and more than half of them exhaust the allowance. A mean cost
# below the limit says little about how often the limit is exceeded: with whole units, a mean of 19 is compatible with
# up to three quarters of the episodes reaching 26. The figure places every trained policy by its mean cost (counted up
# to exhaustion) and by how often it exhausts the allowance; the dotted line is the most exhaustion a mean cost allows.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, c in enumerate(configs):
    for r in camp.records(c):
        cost = np.array(r["evaluation"]["cost"])
        ax.plot(cost.mean(), np.mean(r["evaluation"]["failed"]), "o", color=f"C{i}", label=c if r["seed"] == 1 else None)
xs = np.linspace(0, 26, 50)
ax.plot(xs, np.minimum(xs / 26, 1.0), "k:", lw=0.8, label="the most exhaustion a mean cost allows\n(every episode costs 0 or 26)")
ax.set_xlabel("mean cost per episode, counted up to exhaustion"); ax.set_ylabel("allowance exhausted"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. The price and the return
#
# The campaign trained the robot from scratch (2 million steps, three seeds) at prices of 0, 0.003, 0.01, 0.03 and 0.1
# per step from the exhaustion to the end of the episode. Exhaustion falls sharply between 0.01 and 0.03, and the
# return falls with it: avoiding hazards means taking longer ways to the goals. At 0.1 the robot almost stops reaching
# goals. A failure at the start of an episode is then charged about four times what an episode earns, and the learner
# gives up earning in order not to fail. `plan` compares the two before anything is trained.

# %%
earns = np.mean([np.mean(r["evaluation"]["ret"]) for r in camp.records("unpriced")])
for m in (0.03, 0.1):
    p = plan(spec, Requirement(T=spec.horizon - 1, alpha=0.2), stage=Stage(m=m, clip_utility=False), episode_earnings=earns)
    print(f"price {m}: " + ("; ".join(p.warnings) if p.warnings else "; ".join(n for n in p.notes if "earns" in n)))
#
# ## 3. Validation: the direct test decides
#
# Suppose the mission tolerates exhausting the allowance in one episode in five. A certificate from the value bounds
# the risk using the largest possible reward per step and the discounted length of the episode; here that bound is far
# above every return the robot attains, so the certificate is valid and useless. The direct test on fresh episodes
# does not depend on it. We validate the three policies trained at the price of 0.03, on the 300 fresh episodes of
# their records, with the error shared among the fifteen trained policies.

# %%
req = Requirement(T=spec.horizon - 1, alpha=0.2, zeta=0.05, family=15)
for s in camp.seeds("price-0.03"):
    rep = interpret(spec, req, camp.stage("price-0.03", s), camp.evaluation("price-0.03", s),
                    candidate_name=f"the policy at price 0.03, seed {s}")
    print("\n".join(rep.summary(resource="the damage allowance", period="steps").split("\n")[:3]))
    print(f"  (return per episode {np.mean(camp.record('price-0.03', s)['evaluation']['ret']):.2f})\n")
value = [b for b in rep.bounds if "Value" in b.source]
print("The value certificate of the last one:", value[0].statement().split("\n")[0] if value else "not computed")

# %% [markdown]
# Policies that exhaust the allowance in one episode in twenty-five or in one in eight pass; one that exhausts it in
# one in seven, close to the tolerance, cannot be shown to meet it with 300 episodes and the error shared among fifteen
# policies. That is not evidence of a violation: a larger test may certify it. The seeds also differ widely in how much
# they earn at the same price, so validate the policy you will deploy, not the configuration.
#
# ## 4. Regularizers when nothing is trusted
#
# The robot learns from scratch, and no controller or rule is available to serve as a prior or a reference. An anchor
# would pull the policy toward an arbitrary starting point; there is nothing worth keeping. Action regularization needs
# a trusted reference, which is missing too. The campaign therefore uses neither, and only the learner's own defaults.
#
# ## Train it yourself
#
# In the Python environment with Safety-Gymnasium (here `~/.venvs/constrained-rl-safety310`, with the toolbox
# installed without dependencies):
#
# ```bash
# ~/.venvs/constrained-rl-safety310/bin/python campaigns/run.py navigation price-0.03 --seed 4     # about 35 minutes
# ```
