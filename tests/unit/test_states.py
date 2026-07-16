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
        TrialState.WAITING_SYNC,
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


def test_start_and_stop_failures_reach_legal_failure_states() -> None:
    start_failed = TrialStateMachine(TrialState.READY)
    start_failed.transition(TrialState.FAILED, reason="device failed to start")
    assert start_failed.state is TrialState.FAILED
    assert start_failed.is_terminal

    missing_trigger = TrialStateMachine(TrialState.WAITING_SYNC)
    missing_trigger.transition(
        TrialState.RECOVERABLE,
        reason="stopped before synchronization trigger",
    )
    assert missing_trigger.state is TrialState.RECOVERABLE

    stop_interrupted = TrialStateMachine(TrialState.STOPPING)
    stop_interrupted.transition(
        TrialState.RECOVERABLE,
        reason="writer did not close cleanly",
    )
    assert stop_interrupted.state is TrialState.RECOVERABLE
    assert not stop_interrupted.is_terminal


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
