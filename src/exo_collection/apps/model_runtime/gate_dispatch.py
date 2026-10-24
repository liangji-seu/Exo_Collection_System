"""Fail-closed selection of completed gate decisions for a source endpoint."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateStatus:
    mode: str
    state: str
    score: float | None = None
    threshold: float | None = None
    reason: str | None = None
    sequence: int | None = None
    source_grid_ns: int | None = None
    computed_at_ns: int | None = None
    age_ms: float | None = None
    compute_ms: float | None = None
    generation: int = 0


class GateDispatch:
    """Keeps only current-generation decisions and enforces the 100 ms TTL."""
    def __init__(self, mode: str, *, max_age_ns: int = 100_000_000) -> None:
        if mode not in {"none", "ao", "phase_ensemble"}:
            raise ValueError(f"unknown gate mode: {mode}")
        self.mode = mode
        self.max_age_ns = int(max_age_ns)
        self.generation = 0
        self._decisions = []
        self.status = GateStatus(mode, "disabled" if mode == "none" else "warming")
        self._epoch = 0
        self._was_open = False
        self._last_allowed_result = None

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset(self, generation: int) -> None:
        self.generation = int(generation)
        self._decisions.clear()
        self._epoch += 1
        self._was_open = False
        self._last_allowed_result = None
        self.status = GateStatus(self.mode, "disabled" if self.mode == "none" else "warming",
                                 generation=self.generation)

    def publish(self, results) -> None:
        for result in results:
            if int(result.generation) != self.generation:
                continue
            decision = result.decision
            self._decisions.append(result)
        if len(self._decisions) > 64:
            self._decisions = self._decisions[-64:]

    def decide(self, *, source_grid_ns: int, now_ns: int, worker_closed: str | None = None,
               candidate_source_ns: int | None = None) -> tuple[bool, int | None, GateStatus]:
        if self.mode == "none":
            self.status = GateStatus(self.mode, "disabled", generation=self.generation)
            return True, None, self.status
        if worker_closed:
            return self._close("fault", worker_closed)
        # Gate decisions arrive at 20 Hz while model commands may be emitted
        # at 100 Hz. Select the latest completed decision at or before this
        # command endpoint and hold it within TTL. candidate_source_ns is
        # intentionally not part of decision selection: delayed-assistance
        # epoch recovery is checked by the caller against the decision's
        # source timestamp after selection.
        eligible = [r for r in self._decisions
                    if int(r.decision.source_grid_ns) <= int(source_grid_ns)]
        if not eligible:
            return self._close("warming", "no completed decision for candidate source")
        result = max(eligible, key=lambda r: int(r.decision.source_grid_ns))
        decision = result.decision
        computed_age = int(now_ns) - int(result.computed_at_ns)
        age = int(now_ns) - int(decision.source_grid_ns)
        source_age = int(source_grid_ns) - int(decision.source_grid_ns)
        if computed_age < 0 or age < 0 or source_age < 0:
            return self._close("stale", "decision is from the future")
        if max(computed_age, age, source_age) > self.max_age_ns:
            return self._close("stale", "gate decision expired", result, age)
        if not bool(decision.ready):
            return self._close("warming", str(decision.reason or "gate is not ready"), result, age)
        if not bool(decision.allow):
            return self._close("rejected", str(decision.reason or "gate rejected"), result, age)
        reopening = not self._was_open
        self._was_open = True
        self._last_allowed_result = result
        if reopening:
            self._epoch += 1
        self.status = self._status("allow", decision, result, age)
        return True, self._epoch, self.status

    def expire(self, now_ns: int) -> bool:
        """Return true when an open gate has aged past TTL without new input."""
        if self.mode == "none" or not self._was_open:
            return False
        result = self._last_allowed_result
        if (result is None or int(now_ns)-int(result.computed_at_ns) > self.max_age_ns
                or int(now_ns)-int(result.decision.source_grid_ns) > self.max_age_ns):
            self._close("stale", "gate decision expired", result,
                        None if result is None else int(now_ns)-int(result.computed_at_ns))
            return True
        return False

    def _close(self, state, reason, result=None, age=None):
        self._was_open = False
        if result is None:
            self.status = GateStatus(self.mode, state, reason=reason,
                                     generation=self.generation)
        else:
            self.status = self._status(state, result.decision, result, age, reason)
        return False, None, self.status

    def _status(self, state, decision, result, age, reason=None):
        return GateStatus(self.mode, state,
            score=getattr(decision, "score", None),
            threshold=getattr(decision, "threshold", None),
            reason=reason or getattr(decision, "reason", None),
            sequence=getattr(decision, "sequence", None),
            source_grid_ns=int(decision.source_grid_ns),
            computed_at_ns=int(result.computed_at_ns),
            age_ms=None if age is None else age/1e6,
            compute_ms=float(result.compute_ms), generation=int(result.generation))
