"""Shared lifecycle and queue mechanics for callback-driven hardware adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Empty, Full, Queue
from threading import Lock
from time import perf_counter, perf_counter_ns
from typing import Any, Mapping

from exo_collection.domain.events import (
    DeviceStatus,
    DeviceStatusEvent,
    HealthSnapshot,
    HealthStatus,
)

from .base import (
    AdapterError,
    AdapterLifecycleError,
    AdapterState,
    ModalityDescriptor,
    PreparedInfo,
    RawQueueOverflowError,
    StartToken,
    StopReport,
    TrialContext,
)


class QueuedHardwareAdapter(ABC):
    """Loss-intolerant queue and lifecycle shared by real device adapters.

    Vendor callbacks may call :meth:`_publish_raw` from their own threads.  A
    full raw queue is made visible as a fatal adapter fault; control/status
    telemetry remains a separate, lossy queue so it can never block raw data.
    """

    def __init__(self, *, queue_capacity: int) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self._queue_capacity = int(queue_capacity)
        self._raw_queue: Queue[Any] = Queue(maxsize=self._queue_capacity)
        self._control_queue: Queue[Any] = Queue(maxsize=128)
        self._state = AdapterState.DISCONNECTED
        self._state_lock = Lock()
        self._trial: TrialContext | None = None
        self._start_token: StartToken | None = None
        self._last_error: BaseException | None = None
        self._batches_emitted = 0
        self._samples_emitted = 0
        self._raw_queue_overflows = 0
        self._first_data_ns: int | None = None
        self._last_data_ns: int | None = None
        self._rate_started_at: float | None = None
        self._last_device_status = DeviceStatus.DISCONNECTED

    @property
    def state(self) -> AdapterState:
        with self._state_lock:
            return self._state

    @property
    def raw_queue(self) -> Queue[Any]:
        return self._raw_queue

    @property
    def control_queue(self) -> Queue[Any]:
        return self._control_queue

    @abstractmethod
    def descriptor(self) -> ModalityDescriptor:
        raise NotImplementedError

    @abstractmethod
    def configuration_snapshot(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _connect_hardware(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _start_hardware(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _stop_hardware(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _close_hardware(self) -> None:
        raise NotImplementedError

    def connect(self, config: Any = None) -> None:
        if config is not None:
            raise TypeError("hardware adapter configuration is fixed at construction")
        with self._state_lock:
            if self._state is not AdapterState.DISCONNECTED:
                raise AdapterLifecycleError(f"connect not allowed from {self._state.value}")
        self._emit_status(DeviceStatus.CONNECTING, "connecting hardware")
        try:
            self._connect_hardware()
        except BaseException as exc:
            try:
                self._close_hardware()
            except BaseException:
                pass
            self._set_fault(exc)
            raise
        with self._state_lock:
            self._state = AdapterState.CONNECTED
        self._emit_status(DeviceStatus.CONNECTED, "hardware connected")

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        with self._state_lock:
            if self._state not in (AdapterState.CONNECTED, AdapterState.STOPPED):
                raise AdapterLifecycleError(f"prepare not allowed from {self._state.value}")
            if not self._raw_queue.empty():
                raise AdapterLifecycleError("raw queue must be drained before prepare")
            self._trial = trial
            self._start_token = None
            self._last_error = None
            self._batches_emitted = 0
            self._samples_emitted = 0
            self._raw_queue_overflows = 0
            self._first_data_ns = None
            self._last_data_ns = None
            self._rate_started_at = None
            self._reset_trial_state()
            self._state = AdapterState.PREPARED
        descriptor = self.descriptor()
        self._emit_status(DeviceStatus.READY, "hardware prepared")
        return PreparedInfo(
            device_id=descriptor.device_id,
            modality=descriptor.modality,
            trial_uuid=str(trial.trial_uuid),
            clock_domain=descriptor.clock_domain,
            nominal_rate_hz=descriptor.nominal_rate_hz,
            channels=descriptor.channels,
            units=descriptor.units,
            queue_capacity=self._queue_capacity,
            metadata=dict(descriptor.metadata),
        )

    def _reset_trial_state(self) -> None:
        """Subclass hook called while the lifecycle lock is held."""

    def start(self, start_token: StartToken | Mapping[str, Any] | None = None) -> None:
        if start_token is None:
            start_token = StartToken()
        elif isinstance(start_token, Mapping):
            start_token = StartToken(**dict(start_token))
        if not isinstance(start_token, StartToken):
            raise TypeError("start_token must be StartToken, a mapping, or None")
        with self._state_lock:
            if self._state is not AdapterState.PREPARED:
                raise AdapterLifecycleError(f"start not allowed from {self._state.value}")
            self._start_token = start_token
            self._rate_started_at = perf_counter()
            self._state = AdapterState.RUNNING
        try:
            self._start_hardware()
        except BaseException as exc:
            self._set_fault(exc)
            try:
                self._stop_hardware()
            except BaseException:
                pass
            raise
        self._emit_status(DeviceStatus.RECORDING, "hardware acquisition started")

    def stop(self) -> StopReport:
        state = self.state
        if state is AdapterState.CLOSED:
            raise AdapterLifecycleError("stop not allowed after close")
        if state in (AdapterState.DISCONNECTED, AdapterState.CONNECTED):
            raise AdapterLifecycleError(f"stop not allowed from {state.value}")
        if state is AdapterState.STOPPED:
            return self._stop_report()
        self._emit_status(DeviceStatus.STOPPING, "stopping hardware acquisition")
        try:
            self._stop_hardware()
        except BaseException as exc:
            self._set_fault(exc)
        with self._state_lock:
            if self._state is not AdapterState.FAULTED:
                self._state = AdapterState.STOPPED
        self._emit_status(
            DeviceStatus.FAULT if self._last_error else DeviceStatus.CONNECTED,
            str(self._last_error) if self._last_error else "hardware acquisition stopped",
        )
        return self._stop_report()

    def close(self) -> None:
        if self.state is AdapterState.CLOSED:
            return
        if self.state in (AdapterState.PREPARED, AdapterState.RUNNING, AdapterState.FAULTED):
            try:
                self.stop()
            except BaseException:
                pass
        try:
            self._close_hardware()
        finally:
            with self._state_lock:
                self._state = AdapterState.CLOSED
            self._emit_status(DeviceStatus.CLOSED, "hardware closed")

    def get_event(self, timeout: float | None = None) -> Any | None:
        try:
            return self._raw_queue.get(timeout=timeout)
        except Empty:
            return None

    poll_event = get_event

    def get_control_event(self, timeout: float | None = None) -> Any | None:
        try:
            return self._control_queue.get(timeout=timeout)
        except Empty:
            return None

    def raise_if_faulted(self) -> None:
        if self._last_error is not None:
            raise AdapterError(str(self._last_error)) from self._last_error

    def _publish_raw(self, event: Any, *, item_count: int, host_monotonic_ns: int) -> None:
        if self.state is not AdapterState.RUNNING:
            return
        try:
            self._raw_queue.put_nowait(event)
        except Full as exc:
            self._raw_queue_overflows += 1
            error = RawQueueOverflowError(
                f"raw queue overflow for {self.descriptor().device_id} "
                f"(capacity={self._queue_capacity})"
            )
            self._set_fault(error)
            raise error from exc
        self._batches_emitted += 1
        self._samples_emitted += int(item_count)
        if self._first_data_ns is None:
            self._first_data_ns = host_monotonic_ns
        self._last_data_ns = host_monotonic_ns

    def _set_fault(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._last_error is None:
                self._last_error = exc
            if self._state is not AdapterState.CLOSED:
                self._state = AdapterState.FAULTED
        self._emit_status(DeviceStatus.FAULT, str(exc), error_code=type(exc).__name__)

    def _stop_report(self) -> StopReport:
        descriptor = self.descriptor()
        return StopReport(
            device_id=descriptor.device_id,
            modality=descriptor.modality,
            batches_emitted=self._batches_emitted,
            samples_emitted=self._samples_emitted,
            injected_dropped_batches=0,
            raw_queue_overflows=self._raw_queue_overflows,
            first_data_monotonic_ns=self._first_data_ns,
            last_data_monotonic_ns=self._last_data_ns,
            fault=str(self._last_error) if self._last_error else None,
        )

    def health(self) -> HealthSnapshot:
        descriptor = self.descriptor()
        state = self.state
        depth = self._raw_queue.qsize()
        elapsed = perf_counter() - self._rate_started_at if self._rate_started_at else 0.0
        actual_rate = self._samples_emitted / elapsed if elapsed > 0 else 0.0
        connected = state in {
            AdapterState.CONNECTED,
            AdapterState.PREPARED,
            AdapterState.RUNNING,
            AdapterState.STOPPED,
        }
        return HealthSnapshot(
            device_id=descriptor.device_id,
            modality=descriptor.modality,
            status=(
                HealthStatus.UNHEALTHY
                if self._last_error is not None or state is AdapterState.FAULTED
                else HealthStatus.HEALTHY if connected else HealthStatus.UNKNOWN
            ),
            device_status=self._device_status_for_state(state),
            connected=connected,
            ready=state in {AdapterState.PREPARED, AdapterState.RUNNING},
            sampling=state is AdapterState.RUNNING,
            queue_depth=depth,
            queue_capacity=self._queue_capacity,
            last_data_host_monotonic_ns=self._last_data_ns,
            actual_sample_rate_hz=actual_rate,
            nominal_sample_rate_hz=descriptor.nominal_rate_hz,
            dropped_packets=self._dropped_packets(),
            sequence_gaps=self._sequence_gaps(),
            message=str(self._last_error) if self._last_error else "ok",
            metrics={
                "batches_emitted": self._batches_emitted,
                "samples_emitted": self._samples_emitted,
                "raw_queue_overflows": self._raw_queue_overflows,
                "queue_fill_ratio": depth / self._queue_capacity,
                **self._health_metrics(),
            },
        )

    def _dropped_packets(self) -> int:
        return 0

    def _sequence_gaps(self) -> int:
        return 0

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {}

    def _emit_status(
        self,
        status: DeviceStatus,
        message: str,
        *,
        error_code: str | None = None,
    ) -> None:
        descriptor = self.descriptor()
        try:
            event = DeviceStatusEvent(
                session_uuid=(
                    str(self._trial.session_uuid)
                    if self._trial is not None and self._trial.session_uuid is not None
                    else None
                ),
                trial_uuid=str(self._trial.trial_uuid) if self._trial is not None else None,
                device_id=descriptor.device_id,
                modality=descriptor.modality,
                clock_domain=descriptor.clock_domain,
                host_monotonic_ns=perf_counter_ns(),
                status=status,
                previous_status=self._last_device_status,
                message=message,
                error_code=error_code,
            )
        except (TypeError, ValueError):
            return
        self._last_device_status = status
        try:
            self._control_queue.put_nowait(event)
        except Full:
            pass

    @staticmethod
    def _device_status_for_state(state: AdapterState) -> DeviceStatus:
        return {
            AdapterState.DISCONNECTED: DeviceStatus.DISCONNECTED,
            AdapterState.CONNECTED: DeviceStatus.CONNECTED,
            AdapterState.PREPARED: DeviceStatus.READY,
            AdapterState.RUNNING: DeviceStatus.RECORDING,
            AdapterState.STOPPED: DeviceStatus.CONNECTED,
            AdapterState.FAULTED: DeviceStatus.FAULT,
            AdapterState.CLOSED: DeviceStatus.CLOSED,
        }[state]


__all__ = ["QueuedHardwareAdapter"]
