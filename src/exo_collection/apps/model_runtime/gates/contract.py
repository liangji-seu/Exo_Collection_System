from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class GateDecision:
    ready: bool
    allow: bool
    score: float | None
    raw_score: float | None
    threshold: float | None
    reason: str
    source_grid_ns: int
    sequence: int
    mode: str


def threshold_allows(score: float, threshold: float) -> bool:
    """Frozen boundary: equality passes and non-finite scores fail closed."""
    import math

    return math.isfinite(float(score)) and float(score) <= float(threshold)


class GateEngine(Protocol):
    mode: str

    def reset(self) -> None: ...

    def step(self, source_grid_ns: int, values: Sequence[float], valid: bool = True) -> GateDecision | None: ...
