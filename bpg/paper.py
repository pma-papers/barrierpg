"""The paper's results, cited by name.

The toolbox cites the result behind every bound it returns. Statement numbers change between revisions of a
manuscript, so the toolbox never prints or documents them. It cites each result by the title the paper gives it
(for example "Fixed price of failure risk"), which is searchable in any revision, and names parts of a statement
by what they say (for example "price-risk law"). All citations go through `cite`:

    >>> cite("fixed_price", "ii")
    "paper, 'Fixed price of failure risk' (price-risk law)"

`RESULTS` maps a stable key to (kind, title). The kind (proposition, theorem, remark, ...) is kept for the results
map only; it is never printed, since a revision may also change a statement's kind. `PAPER["checked_against"]`
records the last revision of the paper against which the toolbox's statements were checked.

Results marked "toolbox" are proved in `docs/supplementary-results.md` and are not stated in the paper.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PAPER = dict(
    title="The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization",
    authors="Pekka Malo, Lauri Viitasaari, Antti Suominen, Eeva Vilkkumaa, Olli Tahvonen",
    checked_against="revision 65 (2026-10-13)",
)

# key -> (kind, title in the paper)
RESULTS: Dict[str, Tuple[str, str]] = {
    # the problem and the model
    "problem": ("section", "The problem"),
    "related_requirements": ("section", "Related requirements, and what a price adds"),
    "when_to_use": ("section", "Which formulation to prefer"),
    "model": ("section", "Budgets and the clipped-barrier mechanism"),
    "risk_floor": ("section", "The represented class"),
    # risk certificates and performance comparisons
    "running_example": ("section", "Running example: an operating rate and a wear allowance"),
    "value_certificate": ("proposition", "Value and objective certificates"),
    "failure_time": ("remark", "The geometric risk as a risk measure of the failure time"),
    "identity": ("section", "Utility at the attained risk level"),
    "matched_risk": ("theorem", "Matched-risk performance comparison"),
    "fixed_price": ("proposition", "Fixed price of failure risk"),
    "fixed_price_section": ("section", "The fixed price of failure risk"),
    "action_risk": ("proposition", "Action regularization and trajectory risk"),
    # convergence and the prior
    "admissible": ("definition", "Admissible parameter regularizer"),
    "admissible_families": ("proposition", "Admissible families"),
    "flow": ("theorem", "Fixed-stage population convergence"),
    "learning_error": ("corollary", "Learning error and budget risk"),
    "prior_improvement": ("remark", "Improvement over the prior policy"),
    "bias": ("remark", "The regularization bias is priced and detected"),
    "particle": ("section", "The prior and the particle form of the flow"),
    "finite_training": ("proposition", "Finite-network training under a potential"),
    "gaussian_prior": ("remark", "The Gaussian prior as entropy, weight decay, and a policy"),
    "centering": ("section", "Improvement relative to the prior policy"),
    "curvature_constants": ("proposition", "Value and action-law curvature"),
    "weight_decay": ("remark", "Potential regularizers and weight decay"),
    "collapse": ("remark", "Collapse under a potential alone"),
    "action_slack": ("proposition", "Improvement over the prior policy (its proof, with the action relative entropy)"),
    "first_variations": ("proposition", "First variations"),
    # deployment and validation
    "transfer": ("theorem", "Population-to-network transfer"),
    "algorithm": ("algorithm", "Fixed-stage learning and independent deployment validation"),
    "validation": ("section", "Validate the deployed candidate at its specified horizon"),
    # related formulations (online appendix)
    "frontier_support": ("proposition", "The price supports the frontier"),
    "one_shot": ("lemma", "The one-shot equivalent"),
    "penalties": ("proposition", "Chance, time and overshoot penalties"),
    "cvar_multiplier": ("proposition", "The price as a CVaR multiplier"),
    "cvar_failure_time": ("proposition", "CVaR of the failure time"),
    "delay_example": ("example", "Delay without eventual feasibility"),
    "price_sequence": ("proposition", "Price sequences reaching a tolerance"),
    # the studies
    "wear_study": ("study", "the wear-allowance study"),
    "lunar_study": ("study", "the fuel-tank study"),
    "flow_test": ("study", "the solvable flow"),
    "anchor_study": ("study", "the parameter-anchor study"),
    "roles": ("study", "the roles of the regularizers"),
    "application": ("study", "the emissions application"),
    "benchmark_suite": ("study", "the benchmark-suite study"),
    # practice
    "guidance": ("section", "Using the bounded barrier"),
    "scope": ("section", "Scope of the guarantees"),
    "toolbox_appendix": ("section", "The companion toolbox"),
    # supplementary results of the toolbox
    "near_exhaustion": ("toolbox", "near-exhaustion certificate"),
    "eventual_risk": ("toolbox", "structural certificate for eventual risk"),
}

FIXED_PRICE_PARTS = {"i": "optimality at the attained risk", "ii": "price-risk law",
                     "iii": "monotonicity in the price", "iv": "learning form"}


def cite(key: str, part: Optional[str] = None) -> str:
    """The citation printed with a bound: the paper's title for the result, and the name of the part if given.
    Parts of 'Fixed price of failure risk' may be given as i, ii, iii, iv; they are printed by name."""
    kind, title = RESULTS[key]
    if kind == "toolbox":
        return f"toolbox, {title} (docs/supplementary-results.md)"
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
