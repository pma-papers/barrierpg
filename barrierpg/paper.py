"""The paper's results, cited by name.

The toolbox cites the result behind every bound it returns. Statement numbers change between revisions of a
manuscript, so the toolbox never prints or documents them. It cites each result by the title the paper gives it
(for example "Fixed price of failure risk"), which is searchable in any revision, and names parts of a statement
by what they say (for example "price-risk bound"). All citations go through `cite`:

    >>> cite("fixed_price", "ii")
    "paper, 'Fixed price of failure risk' (price-risk bound)"

`RESULTS` maps a stable key to (kind, title). The kind (proposition, theorem, remark, ...) is kept for the results
map only; it is never printed, since a revision may also change a statement's kind. `PAPER["checked_against"]`
records the last revision of the paper against which the toolbox's statements were checked.

Results marked "toolbox" are proved in `docs/under-the-hood/supplementary-results.md` and are not stated in the paper.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

PAPER = dict(
    title="The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization",
    authors="Pekka Malo, Lauri Viitasaari, Antti Suominen, Eeva Vilkkumaa, Olli Tahvonen",
    checked_against="revision 76 (2026-11-12)",
)

# key -> (kind, title in the paper)
RESULTS: Dict[str, Tuple[str, str]] = {
    # the problem and the model
    "problem": ("section", "The problem"),
    "related_requirements": ("section", "Types of constraint in related formulations"),
    "when_to_use": ("section", "Which formulation to prefer"),
    "model": ("section", "Budgets and the clipped-barrier mechanism"),
    "risk_lower_bound": ("section", "The represented class"),
    # risk certificates and performance comparisons
    "running_example": ("example", "An asset under a wear allowance"),
    "value_certificate": ("proposition", "Value and objective certificates"),
    "failure_time": ("remark", "The geometric risk as a risk measure of the failure time"),
    "identity": ("section", "Utility at the attained risk level"),
    "matched_risk": ("theorem", "Matched-risk performance comparison"),
    "fixed_price": ("proposition", "Fixed price of failure risk"),
    "fixed_price_section": ("section", "The fixed price of failure risk"),
    "action_risk": ("proposition", "Action regularization and failure risk"),
    # convergence and the prior
    "admissible": ("definition", "Admissible parameter regularizer"),
    "admissible_families": ("proposition", "Admissible families"),
    "flow": ("theorem", "Convergence of the population flow"),
    "optimization_error": ("corollary", "Optimization error and risk along the flow"),
    "prior_improvement": ("remark", "Improvement over the prior policy"),
    "bias": ("remark", "The regularization bias is priced and detected"),
    "particle": ("section", "The prior and the particle system of the flow"),
    "finite_training": ("proposition", "Finite-network training under a potential"),
    "gaussian_prior": ("section", "The prior and the particle system of the flow"),
    "centering": ("section", "Improvement relative to the prior policy"),
    "curvature_constants": ("proposition", "Value and action-law curvature"),
    "weight_decay": ("proposition", "Admissible families"),
    "collapse": ("proposition", "Finite-network training under a potential"),
    "action_slack": ("proposition", "Improvement over the prior policy (its proof, with the action relative entropy)"),
    "first_variations": ("proposition", "First variations"),
    # deployment and validation
    "transfer": ("theorem", "Population-to-network transfer"),
    "algorithm": ("section", "A workflow for practitioners"),
    "validation": ("section", "Validate the deployed candidate at its specified horizon"),
    "physical_price": ("proposition", "The risk priced by a physical horizon"),
    # related formulations (online appendix)
    "frontier_support": ("proposition", "Fixed price of failure risk"),
    "one_shot": ("section", "Clipping the combined reward"),
    "penalties": ("proposition", "Chance, time and overshoot penalties"),
    "cvar_multiplier": ("proposition", "The price as a CVaR multiplier"),
    "cvar_failure_time": ("section", "The conditional value-at-risk of the failure time"),
    "delay_example": ("example", "Delay without eventual feasibility"),
    "price_sequence": ("proposition", "Price sequences reaching a tolerance"),
    # the studies
    "wear_study": ("study", "the wear-allowance study"),
    "lunar_study": ("study", "the fuel-tank study"),
    "flow_test": ("study", "the solvable flow"),
    "anchor_study": ("study", "the parameter-anchor study"),
    "roles": ("study", "the roles of the regularizers"),
    "benchmark_suite": ("study", "the benchmark-suite study"),
    # practice
    "guidance": ("section", "Choice of method, scope and conclusion"),
    "scope": ("section", "Scope of the guarantees"),
    "toolbox_appendix": ("section", "The companion toolbox"),
    # supplementary results of the toolbox
    "small_reserve": ("toolbox", "certificate for a small reserve"),
    "ever_failing": ("toolbox", "structural certificate for the probability of ever failing"),
}

# keys of version 0.7.1 and earlier, accepted by `cite` with a DeprecationWarning
RENAMED_KEYS = {"risk_floor": "risk_lower_bound", "learning_error": "optimization_error", "near_exhaustion": "small_reserve",
                "eventual_risk": "ever_failing"}

FIXED_PRICE_PARTS = {"i": "optimality at the attained risk", "ii": "price-risk bound",
                     "iii": "monotonicity in the price", "iv": "learned laws", "v": "the price supports the frontier"}


def cite(key: str, part: Optional[str] = None) -> str:
    """The citation printed with a bound: the paper's title for the result, and the name of the part if given.
    Parts of 'Fixed price of failure risk' may be given as i, ii, iii, iv; they are printed by name."""
    if key in RENAMED_KEYS:
        warnings.warn(f"the result key {key!r} is deprecated; use {RENAMED_KEYS[key]!r}", DeprecationWarning, stacklevel=2)
        key = RENAMED_KEYS[key]
    kind, title = RESULTS[key]
    if kind == "toolbox":
        return f"toolbox, {title} (docs/under-the-hood/supplementary-results.md)"
    where = f"paper, {title}" if kind == "study" else f"paper, '{title}'"
    if part is None:
        return where
    label = FIXED_PRICE_PARTS.get(part, part) if key == "fixed_price" else part
    return f"{where} ({label})"


def results_table() -> str:
    """Markdown table of every registered result (key, kind, title), for the documentation."""
    rows = ["| Key | Kind | Title in the paper |", "|---|---|---|"]
    for k, (kind, title) in RESULTS.items():
        rows.append(f"| `{k}` | {kind} | {title} |")
    return "\n".join(rows)


__all__ = ["PAPER", "RESULTS", "FIXED_PRICE_PARTS", "cite", "results_table"]
