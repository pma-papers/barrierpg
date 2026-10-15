"""Generate the figure on the README landing page from the campaign records: python tools/make_readme_figure.py

(a) the reward the wrapper gives, over the remaining budget; (b) the retailer: what the cost of insolvency does to
learned policies, next to the rule in use; (c) the robot: how often it is stranded, with the battery it expects and
with one that holds 10% less, by price.
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bpg import LogBarrier, Stage, evaluate                                  # noqa: E402
from bpg.problems.inventory import Retailer, make_inventory_env, BaseStockRule  # noqa: E402

RESULTS = ROOT / "campaigns" / "results"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "legend.frameon": False})
INK, BLUE, ORANGE, GREEN, RED, GREY = "#222222", "#2f6db5", "#e07b28", "#2e9e5b", "#c0392b", "#8a8a8a"


def records(problem, config):
    return [json.loads(p.read_text()) for p in sorted((RESULTS / problem / config).glob("seed*/record.json"))]


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), gridspec_kw=dict(wspace=0.38))

# (a) the reward the wrapper gives
ax = axes[0]
u, m, theta = 0.6, 1.0, 0.6
z = np.linspace(1e-3, 1.0, 800)
B = np.array([LogBarrier(theta, [1.0])([zz]) for zz in z])
ax.plot(z, np.full_like(z, u), color=GREY, ls="--", lw=1, label="the task's reward")
ax.plot(z, np.maximum(-m, u - B), color=BLUE, lw=2.2, label="with an interior barrier")
ax.axhline(-m, color=RED, lw=0.8)
ax.text(0.22, -1.45, "after running out: $-m$ every period,\nso $m$ is the price of failure risk", color=RED, fontsize=7.5)
ax.set_ylim(-2.0, 1.0); ax.set_xlim(0, 1)
ax.set_xlabel("budget left after the step"); ax.set_ylabel("reward per step")
ax.set_title("What the wrapper charges")
ax.legend(fontsize=7.5, loc="center right")

# (b) the retailer
ax = axes[1]
costs = [("unpriced", "no cost", GREY), ("lump-2000", "2,000", "#9ecae1"), ("lump-5000", "5,000", "#4292c6"),
         ("lump-20000", "20,000", "#08519c")]
for cfg, label, col in costs:
    rs = records("retailer", cfg)
    ax.plot([np.mean(r["evaluation"]["failed"]) for r in rs], [np.mean(r["evaluation"]["U"]) for r in rs], "o", color=col,
            ms=5, label=f"learned, insolvency costs {label}" if cfg != "unpriced" else "learned, no cost of insolvency")
retailer = Retailer()
stage = Stage(m=0.0, penalty="multiplier", multiplier=5000.0, clip_utility=False)
rule = evaluate(lambda: make_inventory_env(stage, retailer), BaseStockRule(retailer, 2.0, 300.0).deterministic, 2000,
                spec=retailer.spec(), stage=stage, T=retailer.weeks - 1, seed=11)
ax.plot(rule.p_T, rule.U, "s", color=INK, ms=6, label="the rule in use")
ax.axvline(0.02, color=RED, ls=":", lw=1)
ax.text(0.0203, ax.get_ylim()[1], "tolerance", color=RED, fontsize=7.5, va="top")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
ax.set_xlabel("probability of insolvency within the year"); ax.set_ylabel("profit per year")
ax.set_title("Retailer: the cost of failure sets the trade-off")
leg = ax.legend(fontsize=6.6, loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)

# (c) the robot
ax = axes[2]
cfgs = [("unpriced", "no price"), ("price-1", "1"), ("price-3", "3"), ("price-10", "10")]
x = np.arange(len(cfgs))
nom = [np.mean([np.mean(r["evaluation"]["failed"]) for r in records("robot", c)]) for c, _ in cfgs]
weak = [np.mean([np.mean(r["extra_evaluations"][0]["episodes"]["failed"]) for r in records("robot", c)]) for c, _ in cfgs]
ax.bar(x - 0.18, nom, 0.36, color=BLUE, label="the battery it expects")
ax.bar(x + 0.18, weak, 0.36, color=ORANGE, label="a battery 10% weaker")
ax.axhline(0.05, color=RED, ls=":", lw=1)
ax.set_xticks(x, [lab for _, lab in cfgs]); ax.set_xlabel("price of failure risk")
ax.set_ylabel("missions stranded"); ax.set_title("Robot: stranding falls with the price")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.set_ylim(0, 0.68)
ax.legend(fontsize=7.5, loc="upper right")

out = ROOT / "docs" / "images" / "overview.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", out)
