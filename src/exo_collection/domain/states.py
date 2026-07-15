"""Trial lifecycle states and orchestrator-only transition rules.

The transition graph in this module is deliberately small.  Device adapters and
user interfaces report facts to the orchestrator; they do not update a Trial's
state themselves.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from time import perf_counter_ns
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrialState(StrEnum):
    """Persistent states for a single atomic Trial."""

    IDLE = "IDLE"
    PREPARING = "PREPARING"
    READY = "READY"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    FINALIZING = "FINALIZING"
    FINALIZED = "FINALIZED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    RECOVERABLE = "RECOVERABLE"


# RECOVERABLE -> ABORTED is required by the recovery workflow in section 16.3:
# an operator may explicitly reject a recovered package instead of publishing it.
TRIAL_TRANSITIONS: Final[dict[TrialState, frozenset[TrialState]]] = {
    TrialState.IDLE: frozenset({TrialState.PREPARING}),
    TrialState.PREPARING: frozenset({TrialState.READY, TrialState.FAILED}),
    TrialState.READY: frozenset({TrialState.RECORDING, TrialState.FAILED}),
    TrialState.RECORDING: frozenset({TrialState.STOPPING, TrialState.ABORTED}),
    TrialState.STOPPING: frozenset(
        {TrialState.FINALIZING, TrialState.RECOVERABLE}
    ),
    TrialState.FINALIZING: frozenset(
        {TrialState.FINALIZED, TrialState.RECOVERABLE}
    ),
    TrialState.RECOVERABLE: frozenset(
        {TrialState.FINALIZED, TrialState.ABORTED}
    ),
    TrialState.FINALIZED: frozenset(),
    TrialState.FAILED: frozenset(),
    TrialState.ABORTED: frozenset(),
}


TERMINAL_TRIAL_STATES: Final[frozenset[TrialState]] = frozenset(
    {TrialState.FINALIZED, TrialState.FAILED, TrialState.ABORTED}
)


class InvalidTrialTransition(ValueError):
    """Raised when the orchestrator attempts an edge outside the state graph."""

    def __init__(self, source: TrialState, target: TrialState) -> None:
        self.source = source
        self.target = target
        super().__init__(f"illegal Trial transition: {source.value} -> {target.value}")


class TrialTransition(BaseModel):
    """Auditable record produced for every accepted state change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_state: TrialState
    to_state: TrialState
    reason: str | None = None
    actor: Literal["orchestrator"] = "orchestrator"
    occurred_at_utc: datetime = Field(default_factory=_utc_now)
    host_monotonic_ns: int = Field(default_factory=perf_counter_ns, ge=0)

    @field_validator("occurred_at_utc")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


def allowed_transitions(state: TrialState | str) -> frozenset[TrialState]:
    """Return the immutable set of legal successors for *state*."""

    return TRIAL_TRANSITIONS[TrialState(state)]


def can_transition(source: TrialState | str, target: TrialState | str) -> bool:
    """Return whether an orchestrator may move from *source* to *target*."""

    source_state = TrialState(source)
    target_state = TrialState(target)
    return target_state in TRIAL_TRANSITIONS[source_state]


def validate_transition(source: TrialState | str, target: TrialState | str) -> None:
    """Raise :class:`InvalidTrialTransition` unless the edge is legal."""

    source_state = TrialState(source)
    target_state = TrialState(target)
    if not can_transition(source_state, target_state):
        raise InvalidTrialTransition(source_state, target_state)


class TrialStateMachine:
    """Small in-memory state machine owned by a Trial orchestrator.

    ``state`` and ``history`` are read-only views.  The only mutation entry point
    is :meth:`transition`, which always validates the architecture transition
    graph and emits an audit record.
    """

    def __init__(
        self,
        initial_state: TrialState | str = TrialState.IDLE,
        *,
        history: tuple[TrialTransition, ...] | list[TrialTransition] = (),
    ) -> None:
        self._state = TrialState(initial_state)
        self._history = list(history)
        if self._history and self._history[-1].to_state != self._state:
            raise ValueError("restored transition history does not end at state")

    @property
    def state(self) -> TrialState:
        return self._state

    @property
    def history(self) -> tuple[TrialTransition, ...]:
        return tuple(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_TRIAL_STATES

    def can_transition_to(self, target: TrialState | str) -> bool:
        return can_transition(self._state, target)

    def transition(
        self,
        target: TrialState | str,
        *,
        reason: str | None = None,
        occurred_at_utc: datetime | None = None,
        host_monotonic_ns: int | None = None,
    ) -> TrialTransition:
        target_state = TrialState(target)
        validate_transition(self._state, target_state)
        values: dict[str, object] = {
            "from_state": self._state,
            "to_state": target_state,
            "reason": reason,
        }
        if occurred_at_utc is not None:
            values["occurred_at_utc"] = occurred_at_utc
        if host_monotonic_ns is not None:
            values["host_monotonic_ns"] = host_monotonic_ns
        record = TrialTransition.model_validate(values)
        self._state = target_state
        self._history.append(record)
        return record
