"""Reading the campaign records: per-run summaries, evaluations and trained policies, for the notebooks.

    from records import Campaign
    c = Campaign("retailer")
    c.table(["unpriced", "lump-5000"])           # one row per configuration: failure rate and utility over seeds
    ev = c.evaluation("lump-5000", seed=1)        # a barrierpg.Evaluation rebuilt from the record's episodes
    policy = c.policy("lump-5000", seed=1)        # obs -> action, the trained network (stochastic by default)
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
# a saved PPO model stores its schedules as pickled functions; replacing them on loading keeps a model loadable anywhere
LOAD_OBJECTS = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2}
RESULTS = HERE / "results"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from barrierpg import Stage, evaluation_from_episodes  # noqa: E402


def _stage(d: dict) -> Stage:
    names = {f.name for f in fields(Stage)}
    return Stage(**{k: v for k, v in d.items() if k in names})


class Campaign:
    def __init__(self, problem: str, results: Path = RESULTS):
        import problems as P
        self.problem = P.get(problem)
        self.root = Path(results) / problem

    # --- records ---------------------------------------------------------------------------------------------------
    def seeds(self, config: str) -> List[int]:
        return sorted(int(p.name[4:]) for p in (self.root / config).glob("seed*") if (p / "record.json").exists())

    def record(self, config: str, seed: int) -> dict:
        return json.loads((self.root / config / f"seed{seed}" / "record.json").read_text())

    def records(self, config: str) -> List[dict]:
        return [self.record(config, s) for s in self.seeds(config)]

    def stage(self, config: str, seed: int) -> Stage:
        """The stage under which the run was evaluated (for an adaptive multiplier: its final penalty)."""
        return _stage(self.record(config, seed)["stage"])

    # --- convergence of the training runs --------------------------------------------------------------------------
    def convergence(self, config: str, seed: int, **kw):
        """The toolbox's convergence check (barrierpg.convergence.training_convergence) on the run's per-batch record."""
        from barrierpg.convergence import training_convergence
        r = self.record(config, seed)
        trace = [dict(t) for t in r["monitor"]]
        if r.get("dual") and len(r["dual"]["trace"]) == len(trace):
            for t, d in zip(trace, r["dual"]["trace"]):
                t["multiplier"] = d["multiplier"]
        return training_convergence(trace, **kw)

    def convergence_table(self, configs: List[str], **kw) -> str:
        """For each configuration: in how many seeds the run had converged at the end, and what was still changing."""
        lines = []
        for c in configs:
            res = [(s, self.convergence(c, s, **kw)) for s in self.seeds(c)]
            ok = sum(r.converged for _, r in res)
            moving = "; ".join(f"seed {s}: " + ", ".join(m.label.replace("the ", "") for m in r.moving)
                               for s, r in res if not r.converged)
            lines.append(f"{c:20s} converged in {ok} of {len(res)} seeds" + (f"  (still changing: {moving})" if moving else ""))
        return "\n".join(lines)

    # --- evaluations -----------------------------------------------------------------------------------------------
    def evaluation(self, config: str, seed: int, which: str = "evaluation", T: Optional[int] = None, extra: int = 0):
        """The record's evaluation as a barrierpg.Evaluation. which: 'evaluation' (the trained policy), 'start_evaluation'
        (the policy before training), or 'extra' (the extra evaluation number `extra`)."""
        r = self.record(config, seed)
        ep = r["extra_evaluations"][extra]["episodes"] if which == "extra" else r[which]
        stage = _stage(r["stage"] if which != "start_evaluation" else r["trained_stage"])
        return evaluation_from_episodes(ep["failed"], ep["tau"], ep["U"], ep["I"], spec=self.problem.spec, stage=stage,
                                        T=self.problem.T if T is None else T, length=ep["length"], min_reserve=ep["min_reserve"],
                                        clipped_returns=ep["clipped"], seed=None)

    def column(self, config: str, key: str, which: str = "evaluation") -> np.ndarray:
        """One per-episode quantity (for example 'landed' or 'final_cash') over all seeds, one row per seed."""
        return np.array([r[which][key] for r in self.records(config)], dtype=float)

    def table(self, configs: List[str], which: str = "evaluation", extras: tuple = ()) -> List[Dict]:
        """Failure rate and utility by configuration: the mean over seeds and the range across seeds."""
        rows = []
        for c in configs:
            recs = self.records(c)
            if not recs:
                continue
            fail = np.array([np.mean(r[which]["failed"]) for r in recs])
            U = np.array([np.mean(r[which]["U"]) for r in recs])
            row = dict(config=c, note=recs[0]["note"], seeds=len(recs), failure=fail.mean(), failure_min=fail.min(),
                       failure_max=fail.max(), U=U.mean(), U_min=U.min(), U_max=U.max())
            for k in extras:
                v = np.array([np.mean(r[which][k], axis=0) for r in recs])
                row[k] = v.mean(axis=0)
            rows.append(row)
        return rows

    @staticmethod
    def show(rows: List[Dict], extras: tuple = (), fmt: str = "{:.3f}") -> str:
        head = f"{'configuration':16s} {'seeds':>5s} {'failure (min-max)':>24s} {'utility (min-max)':>26s}" + "".join(f" {k:>12s}" for k in extras)
        out = [head]
        for r in rows:
            fail = f"{r['failure']:.3%} ({r['failure_min']:.2%}-{r['failure_max']:.2%})"
            U = f"{r['U']:.4g} ({r['U_min']:.4g}-{r['U_max']:.4g})"
            ex = "".join(f" {np.array2string(np.round(np.atleast_1d(r[k]), 3), separator=','):>12s}" for k in extras)
            out.append(f"{r['config']:16s} {r['seeds']:>5d} {fail:>24s} {U:>26s}{ex}")
        return "\n".join(out)

    # --- policies --------------------------------------------------------------------------------------------------
    def model(self, config: str, seed: int):
        """The trained stable_baselines3 PPO model of a run (for training further, or as a reference or a prior)."""
        from stable_baselines3 import PPO
        return PPO.load(str(self.root / config / f"seed{seed}" / "model.zip"), device="cpu", custom_objects=LOAD_OBJECTS)

    def policy(self, config: str, seed: int, deterministic: bool = False):
        """The trained network of a run as obs -> action (observations normalized as in training when needed)."""
        from stable_baselines3 import PPO
        d = self.root / config / f"seed{seed}"
        model = PPO.load(str(d / "model.zip"), device="cpu", custom_objects=LOAD_OBJECTS)
        norm = None
        if (d / "vecnormalize.pkl").exists():
            import pickle
            with open(d / "vecnormalize.pkl", "rb") as f:
                norm = pickle.load(f)
            norm.training = False

        def act(obs):
            x = norm.normalize_obs(obs) if norm is not None else obs
            return model.predict(x, deterministic=deterministic)[0]
        return act

    def env(self, stage: Optional[Stage] = None, config: Optional[str] = None, **env_kw):
        """A fresh environment of the problem under a stage (default: the stage of `config`)."""
        if stage is None:
            stage = self.problem.configs[config].stage
        return self.problem.make_env(stage, **env_kw)


__all__ = ["Campaign", "RESULTS"]
