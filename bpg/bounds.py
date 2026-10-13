"""A bound with its provenance: the value, the result it comes from, and the hypotheses it needs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Bound:
    name: str                     # what is bounded, e.g. 'q_beta upper bound'
    value: float                  # the numerical bound
    source: str                   # the manuscript result, e.g. 'Proposition 3.1 (value certificate)'
    hypotheses: List[str] = field(default_factory=list)   # what must hold for the bound to be valid
    inputs: Dict[str, float] = field(default_factory=dict)
    note: Optional[str] = None    # how to read it, including when it is uninformative

    @property
    def informative(self) -> bool:
        return self.value < 1.0

    def statement(self) -> str:
        s = f"{self.name}: {self.value:.6g}  [{self.source}]"
        if self.note:
            s += f"\n    {self.note}"
        if self.hypotheses:
            s += "\n    requires: " + "; ".join(self.hypotheses)
        return s

    def __str__(self):
        return self.statement()


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


__all__ = ["Bound", "clip01"]
