"""Generate the overview figure of the under-the-hood track (numpy only): python tools/make_under_the_hood_figure.py"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bpg import LogBarrier                                        # noqa: E402
from bpg.examples.wear import EXACT_FRONTIER, EXACT_BARRIER       # noqa: E402
from bpg.examples.plateau import PlateauModel, simulate_particles  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "legend.frameon": False})
INK, BLUE, ORANGE, GREEN, RED, GREY = "#222222", "#2f6db5", "#e07b28", "#2e9e5b", "#c0392b", "#8a8a8a"

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), gridspec_kw=dict(wspace=0.38))

# (a) the clipped barrier reward
ax = axes[0]
u, m, theta = 0.6, 1.0, 0.6
z = np.linspace(1e-3, 1.0, 800)
B = np.array([LogBarrier(theta, [1.0])([zz]) for zz in z])
ax.plot(z, u - B, color=GREY, ls="--", lw=1, label="unbounded barrier")
ax.plot(z, np.maximum(-m, u - B), color=BLUE, lw=2.2, label="clipped at $-m$")
ax.fill_between(z, np.maximum(-m, u - B), u, color=BLUE, alpha=0.10, lw=0)
ax.axhline(-m, color=RED, lw=0.8)
ax.text(0.22, -1.42, "after failure: $-m$ every period,\nso $m$ is the price of failure risk", color=RED, fontsize=7.5)
ax.text(0.30, 0.68, "the interior loss keeps a reserve", color=BLUE, fontsize=7.5)
ax.set_ylim(-2.6, 1.0); ax.set_xlim(0, 1)
ax.set_xlabel("remaining budget after the action"); ax.set_ylabel("one-step reward")
ax.set_title("A finite price on failure")
ax.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.12))

# (b) the exact frontier and the price-risk law
ax = axes[1]
items = sorted((q, U, mm) for mm, (q, U) in EXACT_FRONTIER.items() if mm > 0)
q = np.array([i[0] for i in items]); U = np.array([i[1] for i in items]); ms = np.array([i[2] for i in items])
ax.plot(q, U, color=INK, lw=1.6, label="exact frontier (failure-only price)")
qb = [v[0] for v in EXACT_BARRIER.values()]; Ub = [v[1] for v in EXACT_BARRIER.values()]
ax.plot(qb, Ub, color=GREEN, ls="--", marker="o", ms=3, lw=1, label="with an interior barrier")
for mm in (0.1, 1.0, 10.0):
    i = int(np.argmin(np.abs(ms - mm)))
    ax.plot(q[i], U[i], "o", color=ORANGE, ms=5)
    ax.annotate(f"$m={mm:g}$", (q[i], U[i]), textcoords="offset points", xytext=(5, -10), fontsize=7.5, color=ORANGE)
ax.axvline(0.05, color=RED, ls=":", lw=1); ax.text(0.055, 3.55, "tolerance", color=RED, fontsize=7.5)
ax.set_xscale("log"); ax.set_xlabel("breakdown probability $q_\\beta$"); ax.set_ylabel("stopped utility $U$")
ax.set_title("The price selects an attained risk")
ax.legend(fontsize=7.5, loc="upper left")

# (c) roles of the parameter regularizers on the solvable flow
ax = axes[2]
flow = PlateauModel(n_x=80, n_s=80)
plateau = flow.stats(0.0)["V_m"]
for (k1, k2), col, ls, lw, z, lab in (((0.0, 1.0), GREY, "-", 3.0, 2, "pull only (a potential): units collapse"),
                                       ((0.1, 10.0), ORANGE, "--", 1.8, 3, "small noise, strong pull: stays on the plateau"),
                                       ((0.1, 0.0), BLUE, "-", 1.8, 1, "noise and pull (relative entropy): escapes")):
    r = simulate_particles(flow, k1, k2, n=1500, seed=1)
    ax.plot(r.times, r.V_m - plateau, color=col, ls=ls, lw=lw, label=lab, zorder=z)
ax.set_xscale("log"); ax.set_xlabel("learning time (solvable flow)"); ax.set_ylabel("clipped value gained")
ax.set_title("Regularizers play different roles")
ax.legend(fontsize=7.2, loc="upper left")
ax.set_ylim(-0.02, 0.42)

out = ROOT / "docs" / "images" / "under-the-hood-overview.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", out)
