"""Recording stream transport types for bridging continuous preview workers
to the CollectorWorker without stopping or reconnecting the hardware Adapter.

Every connected preview handle owns a bounded raw recording queue and a duplex
control pipe.  The preview worker uses ``RecordingStreamProducer`` to forward
raw events with trial isolation; the CollectorWorker drains the queue through
a ``StreamProxyAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from queue import Full
from time import perf_counter_ns
from typing import Any, Union
from uuid import UUID

from exo_collection.domain.events import FrameBatch, SampleBatch, SyncPulseEvent

RawRecordingEvent = Union[FrameBatch, SampleBatch, SyncPulseEvent]
"""Type alias for any domain event that can be forwarded to a recording queue."""


# ── Control Protocol ──────────────────────────────────────────────────────


class RecordingCommandKind(StrEnum):
    START_RECORDING = "START_RECORDING"
    STOP_RECORDING = "STOP_RECORDING"
    SHUTDOWN = "SHUTDOWN"


@dataclass(frozen=True, slots=True)
class RecordingCommand:
    """Command sent from the UI to a preview worker via the control pipe."""

    kind: RecordingCommandKind
    trial_uuid: str | None = None

    def __post_init__(self) -> None:
        if self.kind in (
            RecordingCommandKind.START_RECORDING,
            RecordingCommandKind.STOP_RECORDING,
        ):
            object.__setattr__(self, "trial_uuid", normalize_trial_uuid(self.trial_uuid))
        elif self.trial_uuid is not None:
            object.__setattr__(self, "trial_uuid", normalize_trial_uuid(self.trial_uuid))


# ── Queue Boundary Markers ────────────────────────────────────────────────


class RecordingBoundaryKind(StrEnum):
    START = "START"
    END = "END"
    FAULT = "FAULT"


@dataclass(frozen=True, slots=True)
class RecordingBoundary:
    """Boundary marker written to the recording queue by the preview worker.

    START opens a trial session; END closes it; FAULT signals a fatal error.
    """

    kind: RecordingBoundaryKind
    trial_uuid: str
    modality: str
    device_id: str
    host_monotonic_ns: int = field(default_factory=perf_counter_ns)
    descriptor: dict[str, Any] = field(default_factory=dict)
    configuration_snapshot: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_uuid", normalize_trial_uuid(self.trial_uuid))


# ── Wrapper for individual raw events on the recording queue ──────────────


@dataclass(frozen=True, slots=True)
class RecordedRawEvent:
    """A raw domain event tagged with its trial UUID for isolation."""

    trial_uuid: str
    modality: str
    device_id: str
    event: RawRecordingEvent

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_uuid", normalize_trial_uuid(self.trial_uuid))
        if self.event.modality != self.modality:
            raise ValueError(
                f"raw event modality {self.event.modality!r} does not match "
                f"stream modality {self.modality!r}"
            )
        if self.event.device_id != self.device_id:
            raise ValueError(
                f"raw event device {self.event.device_id!r} does not match "
                f"stream device {self.device_id!r}"
            )


# ── Error Types ───────────────────────────────────────────────────────────


class RecordingStreamError(RuntimeError):
    """Base error for invalid recording stream state or messages."""


class RecordingStreamOverflow(RecordingStreamError):
    """A loss-intolerant recording queue could not accept the next item."""


# ── Producer State Machine (inside preview worker) ────────────────────────


def normalize_trial_uuid(value: str | UUID | None) -> str:
    if value is None:
        raise ValueError("trial_uuid is required")
    return str(UUID(str(value)))


class RecordingStreamProducer:
    """Single-producer state machine for one modality inside a preview worker.

    Writes ``RecordingBoundary.START``, ``RecordedRawEvent`` items, and
    ``RecordingBoundary.END`` on the same queue, preserving order.
    """

    def __init__(
        self,
        queue: Any,
        *,
        device_id: str,
        modality: str,
        descriptor: dict[str, Any],
        configuration_snapshot: dict[str, Any],
    ) -> None:
        self._queue = queue
        self._device_id = device_id
        self._modality = modality
        self._descriptor = dict(descriptor)
        self._configuration_snapshot = dict(configuration_snapshot)
        self._active_trial_uuid: str | None = None

    @property
    def active_trial_uuid(self) -> str | None:
        return self._active_trial_uuid

    @property
    def recording(self) -> bool:
        return self._active_trial_uuid is not None

    def begin(self, trial_uuid: str | UUID) -> RecordingBoundary:
        normalized = normalize_trial_uuid(trial_uuid)
        if self._active_trial_uuid is not None:
            raise RecordingStreamError(
                f"recording already active for {self._active_trial_uuid}"
            )
        boundary = self._boundary(RecordingBoundaryKind.START, normalized)
        self._put_lossless(boundary)
        self._active_trial_uuid = normalized
        return boundary

    def forward(self, event: RawRecordingEvent) -> bool:
        """Forward a complete event, or return False outside recording."""
        trial_uuid = self._active_trial_uuid
        if trial_uuid is None:
            return False
        if not isinstance(event, (FrameBatch, SampleBatch, SyncPulseEvent)):
            return False
        wrapped = RecordedRawEvent(
            trial_uuid=trial_uuid,
            modality=self._modality,
            device_id=self._device_id,
            event=event,
        )
        self._put_lossless(wrapped)
        return True

    def end(self, trial_uuid: str | UUID) -> RecordingBoundary:
        normalized = normalize_trial_uuid(trial_uuid)
        if self._active_trial_uuid != normalized:
            raise RecordingStreamError(
                f"cannot stop trial {normalized}; active trial is "
                f"{self._active_trial_uuid or 'none'}"
            )
        boundary = self._boundary(RecordingBoundaryKind.END, normalized)
        try:
            self._put_lossless(boundary)
        finally:
            self._active_trial_uuid = None
        return boundary

    def abort(self, message: str) -> RecordingBoundary | None:
        """Stop forwarding and best-effort publish a FAULT."""
        trial_uuid = self._active_trial_uuid
        self._active_trial_uuid = None
        if trial_uuid is None:
            return None
        boundary = self._boundary(
            RecordingBoundaryKind.FAULT, trial_uuid, message=message
        )
        try:
            self._put_lossless(boundary)
        except RecordingStreamOverflow:
            return None
        return boundary

    def _boundary(
        self,
        kind: RecordingBoundaryKind,
        trial_uuid: str,
        *,
        message: str = "",
    ) -> RecordingBoundary:
        return RecordingBoundary(
            kind=kind,
            trial_uuid=trial_uuid,
            modality=self._modality,
            device_id=self._device_id,
            descriptor=dict(self._descriptor),
            configuration_snapshot=dict(self._configuration_snapshot),
            message=message,
        )

    def _put_lossless(self, item: RecordingBoundary | RecordedRawEvent) -> None:
        try:
            self._queue.put_nowait(item)
        except Full as exc:
            raise RecordingStreamOverflow(
                f"recording queue full for {self._modality} ({self._device_id})"
            ) from exc


# ── Endpoint Descriptor (for passing stream info to CollectorWorker) ─────


@dataclass
class RecordingStreamEndpoint:
    """Pickle-safe endpoint passed to the recording process.

    The ``queue`` field is the multiprocessing recording queue.
    ``descriptor`` and ``configuration_snapshot`` provide the full adapter
    metadata needed by ``StreamProxyAdapter``.
    """

    queue: Any
    device_id: str
    modality: str
    descriptor: dict[str, Any]
    configuration_snapshot: dict[str, Any] = field(default_factory=dict)


# ── Module exports ────────────────────────────────────────────────────────

__all__ = [
    "RawRecordingEvent",
    "RecordedRawEvent",
    "RecordingBoundary",
    "RecordingBoundaryKind",
    "RecordingCommand",
    "RecordingCommandKind",
    "RecordingStreamEndpoint",
    "RecordingStreamError",
    "RecordingStreamOverflow",
    "RecordingStreamProducer",
    "normalize_trial_uuid",
]
