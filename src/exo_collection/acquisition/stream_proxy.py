"""Logical adapters backed by persistent preview-worker recording streams.

The proxy implements the normal :class:`ModalityAdapter` lifecycle without
opening a device.  It validates the loss-intolerant IPC protocol and exposes
only raw domain events between matching START and END boundaries.
"""

from __future__ import annotations

from queue import Empty
from time import monotonic
from typing import Any, Mapping

from exo_collection.acquisition.recording_stream import (
    RecordedRawEvent,
    RecordingBoundary,
    RecordingBoundaryKind,
    RecordingStreamEndpoint,
    normalize_trial_uuid,
)
from exo_collection.adapters.base import (
    AdapterError,
    AdapterLifecycleError,
    AdapterState,
    ModalityDescriptor,
    PreparedInfo,
    StartToken,
    StopReport,
    TrialContext,
)
from exo_collection.domain.events import (
    DeviceStatus,
    FrameBatch,
    HealthSnapshot,
    HealthStatus,
    SampleBatch,
    SyncPulseEvent,
)


RawDomainEvent = FrameBatch | SampleBatch | SyncPulseEvent


class RecordingStreamProtocolError(AdapterError):
    """The persistent producer violated the recording-stream contract."""


class RecordingStreamFault(AdapterError):
    """The persistent device process reported a fatal recording fault."""


def _coerce_descriptor(values: Mapping[str, Any]) -> ModalityDescriptor:
    raw = dict(values)
    for name in ("channels", "units", "sample_shape"):
        if name in raw:
            raw[name] = tuple(raw[name])
    raw["metadata"] = dict(raw.get("metadata") or {})
    return ModalityDescriptor(**raw)


class StreamProxyAdapter:
    """Read one modality from a :class:`RecordingStreamEndpoint`.

    Queue ownership remains with the UI-side preview handle.  ``close`` never
    closes or drains data belonging to a later Trial.
    """

    def __init__(
        self,
        endpoint: RecordingStreamEndpoint,
        *,
        start_boundary_timeout_s: float = 10.0,
    ) -> None:
        if start_boundary_timeout_s <= 0:
            raise ValueError("start_boundary_timeout_s must be positive")
        self._endpoint = endpoint
        self._descriptor = _coerce_descriptor(endpoint.descriptor)
        if self._descriptor.device_id != endpoint.device_id:
            raise ValueError("endpoint device_id differs from its descriptor")
        if self._descriptor.modality != endpoint.modality:
            raise ValueError("endpoint modality differs from its descriptor")
        self._configuration_snapshot = dict(endpoint.configuration_snapshot)
        self._start_boundary_timeout_s = float(start_boundary_timeout_s)
        self._state = AdapterState.DISCONNECTED
        self._trial_uuid: str | None = None
        self._start_seen = False
        self._end_boundary: RecordingBoundary | None = None
        self._fault: str | None = None
        self._batches_consumed = 0
        self._items_consumed = 0
        self._first_data_ns: int | None = None
        self._last_data_ns: int | None = None

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def stream_ended(self) -> bool:
        return self._end_boundary is not None

    @property
    def end_host_monotonic_ns(self) -> int | None:
        return (
            None
            if self._end_boundary is None
            else int(self._end_boundary.host_monotonic_ns)
        )

    def descriptor(self) -> ModalityDescriptor:
        return self._descriptor

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return dict(self._configuration_snapshot)

    def connect(self, config: Any = None) -> None:
        if self._state is not AdapterState.DISCONNECTED:
            raise AdapterLifecycleError(
                f"stream proxy connect not allowed from {self._state.value}"
            )
        if config is not None:
            raise AdapterLifecycleError("stream proxy configuration is immutable")
        self._state = AdapterState.CONNECTED

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        if self._state is not AdapterState.CONNECTED:
            raise AdapterLifecycleError(
                f"stream proxy prepare not allowed from {self._state.value}"
            )
        self._trial_uuid = normalize_trial_uuid(trial.trial_uuid)
        self._start_seen = False
        self._end_boundary = None
        self._fault = None
        self._batches_consumed = 0
        self._items_consumed = 0
        self._first_data_ns = None
        self._last_data_ns = None
        self._state = AdapterState.PREPARED
        return PreparedInfo(
            device_id=self._descriptor.device_id,
            modality=self._descriptor.modality,
            trial_uuid=self._trial_uuid,
            clock_domain=self._descriptor.clock_domain,
            nominal_rate_hz=self._descriptor.nominal_rate_hz,
            channels=self._descriptor.channels,
            units=self._descriptor.units,
            queue_capacity=self._queue_capacity(),
            metadata={"source": "persistent_preview_recording_stream"},
        )

    def start(self, start_token: StartToken | None = None) -> None:
        if self._state is not AdapterState.PREPARED or self._trial_uuid is None:
            raise AdapterLifecycleError(
                f"stream proxy start not allowed from {self._state.value}"
            )
        deadline = monotonic() + self._start_boundary_timeout_s
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                self._fail(
                    f"timed out waiting for START boundary for "
                    f"{self._descriptor.modality}"
                )
            try:
                item = self._endpoint.queue.get(timeout=remaining)
            except Empty:
                self._fail(
                    f"timed out waiting for START boundary for "
                    f"{self._descriptor.modality}"
                )
            if isinstance(item, (RecordingBoundary, RecordedRawEvent)):
                # A failed/cancelled Trial can leave already-enqueued raw data
                # or its END boundary behind.  Queue ownership is persistent
                # across Trials, so discard only messages that unambiguously
                # belong to another Trial while searching for this START.
                if item.trial_uuid != self._trial_uuid:
                    continue
            if isinstance(item, RecordedRawEvent):
                self._fail("raw event appeared before START boundary")
            if not isinstance(item, RecordingBoundary):
                self._fail(
                    f"invalid recording stream item before START: "
                    f"{type(item).__name__}"
                )
            self._validate_boundary(item)
            if item.kind is RecordingBoundaryKind.FAULT:
                self._raise_fault(item.message or "device stream fault before START")
            if item.kind is not RecordingBoundaryKind.START:
                self._fail(f"expected START boundary, received {item.kind.value}")
            self._validate_boundary_metadata(item)
            self._start_seen = True
            self._state = AdapterState.RUNNING
            return

    def get_event(self, timeout: float | None = None) -> RawDomainEvent | None:
        if self._state is AdapterState.FAULTED:
            raise RecordingStreamFault(self._fault or "recording stream fault")
        if self._state not in (AdapterState.RUNNING, AdapterState.STOPPED):
            return None
        if self._end_boundary is not None:
            return None
        try:
            if timeout is None:
                item = self._endpoint.queue.get()
            elif timeout <= 0:
                item = self._endpoint.queue.get_nowait()
            else:
                item = self._endpoint.queue.get(timeout=float(timeout))
        except Empty:
            return None

        if isinstance(item, RecordingBoundary):
            self._validate_boundary(item)
            if item.kind is RecordingBoundaryKind.FAULT:
                self._raise_fault(item.message or "device stream fault")
            if item.kind is RecordingBoundaryKind.START:
                self._fail("duplicate START boundary")
            if item.kind is RecordingBoundaryKind.END:
                self._end_boundary = item
                self._state = AdapterState.STOPPED
                return None
            self._fail(f"unknown recording boundary {item.kind!r}")

        if not isinstance(item, RecordedRawEvent):
            self._fail(f"invalid recording stream item {type(item).__name__}")
        self._validate_raw_wrapper(item)
        event = item.event
        self._batches_consumed += 1
        if isinstance(event, FrameBatch):
            item_count = int(event.frame_count)
        elif isinstance(event, SampleBatch):
            item_count = int(event.sample_count)
        else:
            item_count = 1
        self._items_consumed += item_count
        event_time = int(event.host_monotonic_ns)
        if self._first_data_ns is None:
            self._first_data_ns = event_time
        self._last_data_ns = event_time
        return event

    def stop(self) -> StopReport:
        if self._state is AdapterState.FAULTED:
            return self._stop_report()
        if self._state not in (AdapterState.RUNNING, AdapterState.STOPPED):
            raise AdapterLifecycleError(
                f"stream proxy stop not allowed from {self._state.value}"
            )
        if self._end_boundary is None:
            raise AdapterLifecycleError(
                f"cannot stop {self._descriptor.modality} before END boundary"
            )
        self._state = AdapterState.STOPPED
        return self._stop_report()

    def health(self) -> HealthSnapshot:
        faulted = self._state is AdapterState.FAULTED
        return HealthSnapshot(
            device_id=self._descriptor.device_id,
            modality=self._descriptor.modality,
            status=HealthStatus.UNHEALTHY if faulted else HealthStatus.HEALTHY,
            device_status=(
                DeviceStatus.FAULT
                if faulted
                else DeviceStatus.RECORDING
                if self._state is AdapterState.RUNNING
                else DeviceStatus.READY
            ),
            connected=self._state not in (
                AdapterState.DISCONNECTED,
                AdapterState.CLOSED,
            ),
            ready=self._state in (
                AdapterState.PREPARED,
                AdapterState.RUNNING,
                AdapterState.STOPPED,
            ),
            sampling=self._state is AdapterState.RUNNING,
            queue_depth=self._queue_depth(),
            queue_capacity=self._queue_capacity(),
            last_data_host_monotonic_ns=self._last_data_ns,
            actual_sample_rate_hz=(
                self._descriptor.nominal_rate_hz
                if self._state in (AdapterState.RUNNING, AdapterState.STOPPED)
                else 0.0
            ),
            nominal_sample_rate_hz=self._descriptor.nominal_rate_hz,
            dropped_packets=0,
            message=self._fault or "persistent preview recording stream",
            metrics={
                "batches_emitted": self._batches_consumed,
                "samples_emitted": self._items_consumed,
                "stream_end_seen": self.stream_ended,
            },
        )

    def raise_if_faulted(self) -> None:
        if self._fault is not None or self._state is AdapterState.FAULTED:
            raise RecordingStreamFault(self._fault or "recording stream fault")

    def close(self) -> None:
        self._state = AdapterState.CLOSED

    def _validate_boundary(self, boundary: RecordingBoundary) -> None:
        if boundary.trial_uuid != self._trial_uuid:
            self._fail(
                f"recording boundary trial mismatch: {boundary.trial_uuid} != "
                f"{self._trial_uuid}"
            )
        if boundary.modality != self._descriptor.modality:
            self._fail("recording boundary modality mismatch")
        if boundary.device_id != self._descriptor.device_id:
            self._fail("recording boundary device mismatch")

    def _validate_boundary_metadata(self, boundary: RecordingBoundary) -> None:
        if dict(boundary.descriptor) != dict(self._endpoint.descriptor):
            self._fail("START descriptor differs from endpoint descriptor")
        if dict(boundary.configuration_snapshot) != self._configuration_snapshot:
            self._fail("START configuration differs from endpoint configuration")

    def _validate_raw_wrapper(self, wrapped: RecordedRawEvent) -> None:
        if wrapped.trial_uuid != self._trial_uuid:
            self._fail("raw event trial mismatch")
        if wrapped.modality != self._descriptor.modality:
            self._fail("raw event modality mismatch")
        if wrapped.device_id != self._descriptor.device_id:
            self._fail("raw event device mismatch")
        if not isinstance(wrapped.event, (FrameBatch, SampleBatch, SyncPulseEvent)):
            self._fail("recording wrapper contains an unsupported raw event")

    def _stop_report(self) -> StopReport:
        return StopReport(
            device_id=self._descriptor.device_id,
            modality=self._descriptor.modality,
            batches_emitted=self._batches_consumed,
            samples_emitted=self._items_consumed,
            injected_dropped_batches=0,
            raw_queue_overflows=0,
            first_data_monotonic_ns=self._first_data_ns,
            last_data_monotonic_ns=self._last_data_ns,
            fault=self._fault,
        )

    def _fail(self, message: str) -> None:
        self._fault = message
        self._state = AdapterState.FAULTED
        raise RecordingStreamProtocolError(message)

    def _raise_fault(self, message: str) -> None:
        self._fault = message
        self._state = AdapterState.FAULTED
        raise RecordingStreamFault(message)

    def _queue_depth(self) -> int:
        try:
            return max(0, int(self._endpoint.queue.qsize()))
        except (AttributeError, NotImplementedError, OSError):
            return 0

    def _queue_capacity(self) -> int:
        value = getattr(
            self._endpoint.queue,
            "_maxsize",
            getattr(self._endpoint.queue, "maxsize", 0),
        )
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1


__all__ = [
    "RecordingStreamFault",
    "RecordingStreamProtocolError",
    "StreamProxyAdapter",
]
