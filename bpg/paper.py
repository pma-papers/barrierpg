"""Where each result lives in the paper: one registry, so a renumbering is a one-file edit.

The toolbox cites every bound it returns. Statement numbers change between revisions of a
manuscript, so all citations go through `cite`, which joins the number of the current revision
with the statement's name. `RESULTS` maps a stable key to (number, name); `PAPER` describes the
revision the numbers refer to.

    >>> cite("fixed_price", "ii")
    'Proposition 3.3(ii) (Fixed price of failure risk: price-risk law)'

Results marked `supplementary` are proved in `docs/supplementary-results.md` of the toolbox and
are not stated in the paper's current revision.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PAPER = dict(
    title="The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization",
    authors="Pekka Malo, Lauri Viitasaari, Antti Suominen, Eeva Vilkkumaa, Olli Tahvonen",
    revision=62,
    date="2026-10-12",
)

# key -> (number in the current revision, name of the statement)
RESULTS: Dict[str, Tuple[str, str]] = {
    # Section 1
    "when_to_use": ("Section 1.2", "When to use the bounded barrier, and when another method"),
    # Section 2
    "model": ("Section 2", "Budgets and the clipped-barrier mechanism"),
    "risk_floor": ("Section 2.3", "risk floor of a bounded-score class"),
    # Section 3
    "running_example": ("Section 3.1", "Running example: an operating rate and a wear allowance"),
    "value_certificate": ("Proposition 3.1", "Value and objective certificates"),
    "identity": ("Section 3.3", "clipping identity V_m = U - I_m - m q_beta/(1-beta)"),
    "matched_risk": ("Theorem 3.2", "Matched-risk performance comparison"),
    "fixed_price": ("Proposition 3.3", "Fixed price of failure risk"),
    "dual_unbounded": ("Section 3.4", "a prescribed level below the risk floor is unattainable"),
    "action_risk": ("Proposition 3.4", "Action regularization and trajectory risk"),
    # Section 4
    "admissible": ("Definition 4.1", "Admissible parameter regularizer"),
    "admissible_families": ("Proposition 4.2", "Admissible families"),
    "flow": ("Theorem 4.3", "Fixed-stage population convergence"),
    "learning_error": ("Corollary 4.4", "Learning error and budget risk"),
    "prior_improvement": ("Remark 4.5", "Improvement over the prior policy"),
    "bias": ("Remark 4.6", "The regularization bias is priced and detected"),
    "particle": ("Section 4.2", "particle form of the flow"),
    "finite_training": ("Proposition 4.7", "Finite-network training under a potential"),
    "gaussian_prior": ("Remark 4.8", "The Gaussian prior as entropy, weight decay, and a policy"),
    "centering": ("Section 4.1", "centering the features at the prior"),
    # Section 5
    "transfer": ("Theorem 5.1", "Population-to-network transfer"),
    "algorithm": ("Algorithm 1", "Fixed-stage learning and independent deployment validation"),
    "validation": ("Section 5.2", "Validate the deployed candidate at its specified horizon"),
    # Section 6
    "wear_study": ("Section 6.1", "the wear-allowance study"),
    "lunar_study": ("Section 6.2", "the fuel-tank study"),
    "flow_test": ("Section 6.3", "the solvable flow"),
    "anchor_study": ("Section 6.4", "The prior on a learned controller"),
    "roles": ("Section 6.5", "The roles of the regularizers"),
    "application": ("Section 6.6", "the emissions application"),
    # Section 7
    "guidance": ("Section 7.1", "Using the bounded barrier"),
    "scope": ("Section 7.2", "Scope of the guarantees"),
    # Appendix B
    "curvature_constants": ("Proposition B.1", "Value and action-law curvature"),
    # Online appendix
    "weight_decay": ("Online Remark F.1", "Potential regularizers and weight decay"),
    "collapse": ("Online Remark F.2", "Collapse under a potential alone"),
    "action_slack": ("Online Proposition F.3", "second risk slack with the action relative entropy"),
    "first_variations": ("Online Proposition E.2", "First variations"),
    "delay_example": ("Online Example E.3", "Delay without eventual feasibility"),
    "price_sequence": ("Online Proposition E.4", "Price sequences reaching a tolerance"),
    "toolbox_appendix": ("Online Appendix M", "The companion toolbox"),
    # Supplementary results of the toolbox, not in the current revision
    "near_exhaustion": ("supplementary", "near-exhaustion certificate"),
    "eventual_risk": ("supplementary", "structural certificate for eventual risk"),
}

FIXED_PRICE_PARTS = {"i": "optimality at the attained risk", "ii": "price-risk law",
                     "iii": "monotonicity in the price", "iv": "learning form"}


def cite(key: str, part: Optional[str] = None) -> str:
    """'Number(part) (Name: part name)' for a registered result; supplementary results say so."""
    number, name = RESULTS[key]
    if number == "supplementary":
        return f"{name} (toolbox supplementary result, docs/supplementary-results.md)"
    if part is None:
        return f"{number} ({name})"
    label = FIXED_PRICE_PARTS.get(part, part) if key == "fixed_price" else part
    return f"{number}({part}) ({name}: {label})"


def number(key: str) -> str:
    return RESULTS[key][0]


def results_table() -> str:
    """Markdown table of every registered result, for the documentation."""
    rows = ["| Key | Paper | Statement |", "|---|---|---|"]
    for k, (n, name) in RESULTS.items():
        rows.append(f"| `{k}` | {n} | {name} |")
    return "\n".join(rows)


__all__ = ["PAPER", "RESULTS", "cite", "number", "results_table"]
