"""Checks across runs at several prices: the price-risk law, monotonicity, the frontier of U - I_m against
q_beta, the unused risk charged at the price, and whether a sequence of prices can reach a tolerance
(the fixed-price proposition, the matched-risk comparison and the price-sequence proposition of the paper)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .bounds import Bound
from .certificates import price_monotonicity, price_risk_law
from .paper import cite


@dataclass
class Run:
    """One learned or exact policy at a fixed price: its attained quantities."""
    m: float
    q_beta: float
    U: float
    I_m: float = 0.0
    label: str = ""
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def net_utility(self) -> float:
        """U - I_m: the quantity every fixed-price optimum is optimal in at its attained risk (the fixed-price proposition)."""
        return self.U - self.I_m


@dataclass
class FrontierCheck:
    law_bounds: List[Bound]
    law_violations: List[str]
    monotonicity_violations: List[str]
    dominated: List[str]
    table: List[dict]

    @property
    def consistent(self) -> bool:
        return not self.law_violations and not self.monotonicity_violations

    def explain(self) -> str:
        lines = []
        for row in self.table:
            lines.append(f"m={row['m']:<8g} q={row['q_beta']:.4g}  U-I={row['net_utility']:.4g}  law bound={row['law_bound']:.4g}"
                         + ("  (violates the price-risk law)" if row["violates_law"] else ""))
        if self.monotonicity_violations:
            lines += [f"monotonicity ({cite('fixed_price', 'iii')}) violated: " + v for v in self.monotonicity_violations]
        else:
            lines.append(f"monotonicity in the price: consistent within the stated optimization errors [{cite('fixed_price', 'iii')}]")
        if self.dominated:
            lines += ["dominated in (q_beta, U - I_m): " + d for d in self.dominated]
        lines.append("a fixed-price optimum lies on the frontier of U - I_m against q_beta; a point strictly below another with no greater risk "
                     "indicates optimization error (or a non-optimal policy), not a property of the formulation")
        return "\n".join(lines)


def check_runs(runs: Sequence[Run], beta: float, sup_U: float, U0: float = 0.0, I0: float = 0.0, q0: float = 0.0,
               delta: float = 0.0, ubar: Optional[float] = None, u_minus: float = 0.0, tol: float = 0.0) -> FrontierCheck:
    """For runs at prices m >= u_minus: compare each attained q_beta with the price-risk law (with the
    comparison policy (U0, I0, q0), idle by default), check monotonicity between consecutive prices
    using delta as each run's optimization error (and the observed interior losses for Delta), and
    flag runs dominated in (q_beta, U - I_m). tol is an additive slack for Monte Carlo error in q; a run whose
    extra carries q_se (the standard error of its estimated q_beta) gets a further slack of two standard errors."""
    rs = sorted(runs, key=lambda r: (r.m, r.seed if r.seed is not None else 0))
    bounds, viol, table = [], [], []
    for r in rs:
        b = price_risk_law(r.m, beta, sup_U, U0, I0, q0, delta, ubar)
        bounds.append(b)
        v = r.q_beta > b.value + tol + 2.0 * r.extra.get("q_se", 0.0)
        if v:
            viol.append(f"m={r.m:g}{(' ' + r.label) if r.label else ''}: q={r.q_beta:.4g} > bound {b.value:.4g}; the policy is not a delta-optimizer "
                        f"at this price (or sup_U/delta are misstated)")
        table.append(dict(m=r.m, q_beta=r.q_beta, U=r.U, I_m=r.I_m, net_utility=r.net_utility, law_bound=b.value, violates_law=v, label=r.label))
    mono = []
    by_m = {}
    for r in rs:
        by_m.setdefault(r.m, []).append(r)
    ms = sorted(by_m)
    for m1, m2 in zip(ms[:-1], ms[1:]):
        if m1 < u_minus:
            continue
        g1 = by_m[m1]; g2 = by_m[m2]
        q1 = float(np.mean([r.q_beta for r in g1])); q2 = float(np.mean([r.q_beta for r in g2]))
        # Delta = I_{m2}(pi_1) - I_{m1}(pi_1) is not observable from I_{m1}(pi_1) alone; it is zero for a
        # failure-only price and at most (m2 - m1) * (discounted barrier mass) otherwise. Use the failure-only form
        # unless the runs carry 'Delta' in extra.
        Delta = float(np.mean([r.extra.get("Delta", 0.0) for r in g1]))
        b = price_monotonicity(q1, m1, m2, beta, Delta, delta, delta)
        if q2 > b.value + tol:
            mono.append(f"q at m={m2:g} ({q2:.4g}) exceeds the bound {b.value:.4g} from m={m1:g} ({q1:.4g})")
    dom = []
    for r in rs:
        for s in rs:
            if s is r:
                continue
            if s.q_beta <= r.q_beta + tol and s.net_utility > r.net_utility + 1e-12 and (s.q_beta < r.q_beta - tol or s.net_utility > r.net_utility):
                dom.append(f"m={r.m:g}{(' ' + r.label) if r.label else ''} (q={r.q_beta:.4g}, U-I={r.net_utility:.4g}) is dominated by m={s.m:g}{(' ' + s.label) if s.label else ''} "
                           f"(q={s.q_beta:.4g}, U-I={s.net_utility:.4g})")
                break
    return FrontierCheck(bounds, viol, mono, dom, table)


def complementarity_gap(alpha: float, q_hat: float, m: float, beta: float) -> float:
    """m [alpha - q_hat] / (1 - beta): unused permitted risk charged at the price, the utility a fixed-price
    policy may leave on the table against a policy at exactly the (geometric) tolerance (the matched-risk comparison)."""
    return m * (alpha - q_hat) / (1.0 - beta)


def frontier_points(runs: Sequence[Run]) -> List[Run]:
    """The nondominated runs in (q_beta, U - I_m), sorted by risk."""
    rs = sorted(runs, key=lambda r: r.q_beta)
    out, best = [], -np.inf
    for r in rs:
        if r.net_utility > best:
            out.append(r); best = r.net_utility
    return out


def interpolate_frontier(points: Sequence[Run], q: float) -> float:
    """Net utility of the frontier at risk q by log-linear interpolation in q (the gap statistic of the study)."""
    pts = frontier_points(points)
    qs = np.array([max(p.q_beta, 1e-12) for p in pts]); us = np.array([p.net_utility for p in pts])
    return float(np.interp(np.log(max(q, qs.min())), np.log(qs), us))


@dataclass
class PriceSequenceCheck:
    rows: List[dict]
    loglog_slope: float
    shrinking: bool
    explanation: str
    informative: bool = True

    def explain(self) -> str:
        lines = [f"m={r['m']:<8g} charged={r['charged']:.4g}  charged/m={r['ratio']:.4g}" for r in self.rows]
        return "\n".join(lines + [self.explanation])


def price_sequence_check(prices: Sequence[float], shortfalls: Sequence[float], *, interior_loss: Optional[Sequence[float]] = None,
                         regularization_cost: float = 0.0, ratio_tolerance: float = 0.8) -> PriceSequenceCheck:
    """Can raising the price reach a tolerance? 'Price sequences reaching a tolerance': a sequence of prices m_k reaches it when some
    comparison policy has geometric risk strictly below the target and the charged terms, its interior loss, the
    regularization cost and the optimization error delta_k, grow slower than the price. This reads that condition on
    measured runs: shortfalls[k] is the optimization error at prices[k] (for example the gap to an exact optimum),
    interior_loss[k] the comparison policy's interior loss at that price. The charged total divided by the price
    should shrink; if it does not at the largest prices, the learner's error grows with the price and raising the
    price is not the remedy (PPO in the emissions study). Evidence about these runs, not a proof."""
    m = np.asarray(prices, dtype=float); d = np.asarray(shortfalls, dtype=float)
    I = np.zeros_like(m) if interior_loss is None else np.asarray(interior_loss, dtype=float)
    order = np.argsort(m); m, d, I = m[order], d[order], I[order]
    charged = I + regularization_cost + np.maximum(d, 0.0)
    ratio = charged / m
    rows = [dict(m=float(a), charged=float(c), ratio=float(r)) for a, c, r in zip(m, charged, ratio)]
    pos = (charged > 0) & (m > 0)
    slope = float(np.polyfit(np.log(m[pos]), np.log(charged[pos]), 1)[0]) if pos.sum() >= 2 else float("nan")
    k = len(m)
    if not np.any(charged > 0):
        return PriceSequenceCheck(rows, float("nan"), True,
                                  "no charged error was measured at any price (zero shortfalls, or zero lower bounds on them): the check says nothing "
                                  f"about whether raising the price can reach a tolerance [{cite('price_sequence')}]", informative=False)
    mid = ratio[k // 2] if k >= 3 else ratio[0]
    shrinking = bool(ratio[-1] < ratio_tolerance * mid)
    if shrinking:
        text = (f"the charged errors shrink relative to the price (charged/m {mid:.3g} -> {ratio[-1]:.3g}; log-log slope {slope:.2f}): "
                f"consistent with a sequence of prices reaching a tolerance that some comparison policy meets with risk to spare [{cite('price_sequence')}]")
    else:
        text = (f"the charged errors do not shrink relative to the price (charged/m {mid:.3g} -> {ratio[-1]:.3g}; log-log slope {slope:.2f}): "
                f"the learner's error grows with the price, so raising the price is not the remedy; consider another learner, a structured "
                f"policy, reward normalization or a prescribed-level formulation [{cite('price_sequence')}; {cite('application')}]")
    return PriceSequenceCheck(rows, slope, shrinking, text)


__all__ = ["Run", "FrontierCheck", "check_runs", "complementarity_gap", "frontier_points", "interpolate_frontier",
           "PriceSequenceCheck", "price_sequence_check"]
