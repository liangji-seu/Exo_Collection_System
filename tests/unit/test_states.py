from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from exo_collection.domain.states import (
    InvalidTrialTransition,
    TrialState,
    TrialStateMachine,
    TrialTransition,
    allowed_transitions,
    can_transition,
)


def test_happy_path_reaches_finalized_and_records_audit_history() -> None:
    machine = TrialStateMachine()
    path = [
        TrialState.PREPARING,
        TrialState.READY,
        TrialState.RECORDING,
        TrialState.STOPPING,
        TrialState.FINALIZING,
        TrialState.FINALIZED,
    ]

    for index, state in enumerate(path, start=1):
        transition = machine.transition(
            state,
            reason=f"step {index}",
            host_monotonic_ns=index,
        )
        assert transition.to_state is state
        assert transition.actor == "orchestrator"

    assert machine.state is TrialState.FINALIZED
    assert machine.is_terminal
    assert tuple(item.to_state for item in machine.history) == tuple(path)
    assert allowed_transitions(TrialState.FINALIZED) == frozenset()


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (TrialState.IDLE, TrialState.RECORDING),
        (TrialState.READY, TrialState.FINALIZED),
        (TrialState.RECORDING, TrialState.FINALIZED),
        (TrialState.FINALIZED, TrialState.RECORDING),
        (TrialState.FAILED, TrialState.PREPARING),
    ],
)
def test_illegal_transition_is_rejected(
    source: TrialState, target: TrialState
) -> None:
    machine = TrialStateMachine(source)
    assert not can_transition(source, target)
    with pytest.raises(InvalidTrialTransition) as exc_info:
        machine.transition(target)
    assert exc_info.value.source is source
    assert exc_info.value.target is target
    assert machine.state is source
    assert machine.history == ()


def test_abort_and_recovery_edges_match_architecture() -> None:
    aborting = TrialStateMachine(TrialState.RECORDING)
    aborting.transition(TrialState.ABORTED)
    assert aborting.is_terminal

    recovered = TrialStateMachine(TrialState.RECOVERABLE)
    recovered.transition(TrialState.FINALIZED)
    assert recovered.state is TrialState.FINALIZED

    rejected_recovery = TrialStateMachine(TrialState.RECOVERABLE)
    rejected_recovery.transition(TrialState.ABORTED)
    assert rejected_recovery.state is TrialState.ABORTED


def test_state_is_read_only() -> None:
    machine = TrialStateMachine()
    with pytest.raises(AttributeError):
        machine.state = TrialState.RECORDING  # type: ignore[misc]


def test_transition_audit_timestamp_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        TrialTransition(
            from_state=TrialState.IDLE,
            to_state=TrialState.PREPARING,
            occurred_at_utc=datetime(2026, 1, 1),
        )

    record = TrialTransition(
        from_state=TrialState.IDLE,
        to_state=TrialState.PREPARING,
        occurred_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert record.occurred_at_utc.utcoffset().total_seconds() == 0

