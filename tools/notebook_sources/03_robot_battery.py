# %% [markdown]
# # 3. A legged robot on a battery: the price, a clock, and a reserve margin
#
# A robot runs forward (MuJoCo's HalfCheetah) on a mission of 1000 steps. It now carries a battery: each step draws an
# idle load plus the resistive losses of its motors, which grow with the square of the torques, scaled by an
# efficiency that varies from mission to mission with battery health and temperature and is not observed. A robot
# whose battery runs flat before the end of the mission is stranded.
#
# Running fast earns the most per step and drains the battery fastest. This notebook shows:
#
# 1. how often a gait trained without a battery would be stranded;
# 2. what the policy must observe to pace a battery: the charge left and the time left;
# 3. what prices do to stranding and to distance covered;
# 4. a reserve margin (the interior barrier) against a battery that holds less than the robot believes;
# 5. validation and the report.
#
# The trained policies come from the campaign in `campaigns/` (PPO, 2 million steps, three seeds per configuration).
# Re-evaluating them needs Gymnasium with MuJoCo.

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("..").resolve() / "campaigns"))
from records import Campaign
from bpg import Requirement, Stage, evaluate, interpret, plan
from bpg.problems.energy import Battery, make_robot_env

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
camp = Campaign("robot")
battery = Battery()
spec = battery.spec()

# %% [markdown]
# ## 1. A gait trained without a battery
#
# The energy a gait uses over a mission varies with the mission's efficiency. Against a battery of 2,500 units, a
# gait trained without a battery would be stranded in most missions.

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for s in camp.seeds("no-battery"):
    e = np.array(camp.record("no-battery", s)["evaluation"]["energy"])
    ax.hist(e, bins=30, histtype="step", label=f"gait without a battery, seed {s}")
ax.axvline(battery.charge, color="C3", ls="--", label="the battery")
ax.set_xlabel("energy used in a mission"); ax.set_ylabel("missions"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. The requirement, and what the policy must observe
#
# The robot should be stranded in at most 5% of missions, shown at 95% confidence. The task's reward can be negative
# (running backwards), so the stage prices only the failure (`clip_utility=False`).
#
# The policy has to pace the battery, and pacing needs two inputs: how much charge is left and how much of the
# mission is left. The task's own observation has no clock. The wrapper appends the remaining charge, and with
# `observe_time=True` also the elapsed fraction of the mission. Without the clock the robot cannot tell whether its
# charge will last.

# %%
req = Requirement(T=spec.horizon - 1, alpha=0.05, zeta=0.05, family=10)
print(Campaign.show(camp.table(["unpriced-no-clock", "unpriced", "price-3-no-clock", "price-3"], extras=("distance",)), extras=("distance",)))

# %% [markdown]
# Without the clock, the robot is stranded in more than half of its missions when there is no price, and a price helps
# only in some seeds. With the clock, stranding without a price falls to about a fifth. A policy that cannot see the
# time left cannot pace a budget, however the failure is priced. With a fixed horizon, observe the time.
#
# ## 3. Prices
#
# The campaign trained the robot, with the clock, at prices of 0, 1, 3 and 10 per step from the failure to the end of
# the mission.

# %%
prices = ["unpriced", "price-1", "price-3", "price-10"]
print(Campaign.show(camp.table(prices, extras=("distance", "energy")), extras=("distance", "energy")))

# %% [markdown]
# Stranding falls steadily with the price, and so does the distance covered: pacing the battery means running slower.
# Only the price of 10 meets the 5% tolerance in every seed. The price chooses how much distance to give up for a lower
# risk; the tolerance decides which price is enough.
#
# ## 4. A reserve margin against a weaker battery
#
# Every configuration was also evaluated with a battery that holds 10% less than the robot believes (the robot still
# sees the nominal charge). A price on failure alone leaves no reserve; the interior barrier charges a small amount
# every step that grows as the charge falls.

# %%
rows = []
for cfg in ["unpriced", "price-1", "price-3", "price-10", "barrier"]:
    nominal = [np.mean(r["evaluation"]["failed"]) for r in camp.records(cfg)]
    weak = [np.mean(r["extra_evaluations"][0]["episodes"]["failed"]) for r in camp.records(cfg)]
    dist = [np.mean(r["evaluation"]["distance"]) for r in camp.records(cfg)]
    print(f"{cfg:10s} stranded: nominal battery {np.mean(nominal):6.1%}, battery 10% weaker {np.mean(weak):6.1%};  distance {np.mean(dist):6.0f}")

# %% [markdown]
# With a battery that holds 10% less, every policy is stranded more often. A policy that meets the tolerance with room
# to spare (the price of 10) keeps a margin under the weaker battery as well. The interior barrier at a lower price (3)
# lowered stranding in two of three seeds, under both batteries, but its results varied widely across seeds; on this
# problem a higher price was the more reliable way to a reserve. If the size of a possible battery error is known,
# planning with the smaller battery is the cheaper margin.
#
# ## 5. Validate and report
#
# We validate the candidate with the least stranding on its first evaluation, on 500 fresh missions, as a stochastic
# policy and as its mean action. Ten candidates were inspected, so the error is shared among ten.

# %%
cands = [(c, s) for c in prices + ["barrier"] for s in camp.seeds(c)]
best = min(cands, key=lambda cs: np.mean(camp.record(*cs)["evaluation"]["failed"]))
print("candidate:", best, camp.record(*best)["note"])
stage = camp.stage(*best)
for det in (False, True):
    e = evaluate(lambda: make_robot_env(stage, battery), camp.policy(*best, deterministic=det), 500, spec=spec, stage=stage,
                 T=req.T, seed=555)
    name = f"the {'mean action of the' if det else 'stochastic'} policy {best[0]}, seed {best[1]}"
    print()
    print(interpret(spec, req, stage, e, candidate_name=name).summary(resource="charge", period="steps"))

# %% [markdown]
# Both pass. The mean action covers more distance with no more stranding: it spends no effort on the noise that the
# stochastic policy adds to its actions. If you deploy the mean action, it is the candidate whose validation counts.
#
# ## Train it yourself
#
# ```bash
# python campaigns/run.py robot price-3 --seed 4       # about 5 minutes on one core
# ```
