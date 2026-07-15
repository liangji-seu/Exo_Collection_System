"""Common modality-adapter contracts and a safe simulated-worker base.

Adapters deliberately keep the control plane separate from the bounded raw-data
queue.  A full raw queue is a fatal acquisition condition: the adapter records
an explicit fault and stops instead of silently discarding samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
from time import perf_counter, perf_counter_ns, time_ns
from typing import Any, Generic, Iterator, Mapping, Protocol, TypeVar, runtime_checkable
from uuid import UUID, uuid4

import numpy as np

from exo_collection.domain.events import (
    DeviceStatus,
    DeviceStatusEvent,
    HealthSnapshot,
    HealthStatus,
)


class AdapterError(RuntimeError):
    """Base class for adapter failures."""


class AdapterLifecycleError(AdapterError):
    """Raised when a lifecycle method is called out of order."""


class RawQueueOverflowError(AdapterError):
    """Fatal error raised/reported when the raw acquisition queue is full."""


class SimulatedDisconnectError(AdapterError):
    """Fault injected by a simulator to model a device disconnect."""


class AdapterState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    PREPARED = "prepared"
    RUNNING = "running"
    STOPPED = "stopped"
    FAULTED = "faulted"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ModalityDescriptor:
    """Static and configured capabilities of one adapter instance."""

    device_id: str
    modality: str
    display_name: str
    clock_domain: str
    event_kind: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    nominal_rate_hz: float
    sample_shape: tuple[int, ...]
    dtype: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id must not be empty")
        if self.nominal_rate_hz <= 0:
            raise ValueError("nominal_rate_hz must be positive")
        if len(self.channels) != len(self.units):
            raise ValueError("channels and units must have the same length")
        if not self.sample_shape or any(int(v) <= 0 for v in self.sample_shape):
            raise ValueError("sample_shape dimensions must be positive")


@dataclass(frozen=True, slots=True)
class TrialContext:
    """Immutable information handed to an adapter during ``prepare``."""

    trial_uuid: UUID | str
    session_uuid: UUID | str | None = None
    condition: Mapping[str, Any] = field(default_factory=dict)
    recording_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Minimal generic wrapper accepted by ``connect``.

    Project configuration models may pass the architecture's ``id`` plus an
    adapter-specific ``parameters`` mapping.  Simulators also accept their
    strongly typed configuration dataclass directly.
    """

    id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("device config id must not be empty")


@dataclass(frozen=True, slots=True)
class StartToken:
    """Shared start anchor distributed by the orchestrator."""

    token_uuid: UUID = field(default_factory=uuid4)
    host_monotonic_ns: int = field(default_factory=perf_counter_ns)
    host_utc_ns: int = field(default_factory=time_ns)

    def __post_init__(self) -> None:
        if self.host_monotonic_ns < 0 or self.host_utc_ns < 0:
            raise ValueError("start timestamps must be non-negative")


@dataclass(frozen=True, slots=True)
class PreparedInfo:
    device_id: str
    modality: str
    trial_uuid: str
    clock_domain: str
    nominal_rate_hz: float
    channels: tuple[str, ...]
    units: tuple[str, ...]
    queue_capacity: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StopReport:
    device_id: str
    modality: str
    batches_emitted: int
    samples_emitted: int
    injected_dropped_batches: int
    raw_queue_overflows: int
    first_data_monotonic_ns: int | None
    last_data_monotonic_ns: int | None
    fault: str | None


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Fault/timing controls shared by every simulated device."""

    queue_capacity: int = 64
    seed: int = 0
    clock_drift_ppm: float = 0.0
    timestamp_jitter_ns: int = 0
    drop_every_n_batches: int = 0
    drop_probability: float = 0.0
    disconnect_after_batches: int | None = None
    realtime: bool = True

    def __post_init__(self) -> None:
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if self.timestamp_jitter_ns < 0:
            raise ValueError("timestamp_jitter_ns must be non-negative")
        if self.drop_every_n_batches < 0:
            raise ValueError("drop_every_n_batches must be non-negative")
        if not 0.0 <= self.drop_probability <= 1.0:
            raise ValueError("drop_probability must be in [0, 1]")
        if self.disconnect_after_batches is not None and self.disconnect_after_batches < 0:
            raise ValueError("disconnect_after_batches must be non-negative")


ConfigT = TypeVar("ConfigT", bound=SimulationConfig)


@runtime_checkable
class ModalityAdapter(Protocol):
    """Lifecycle contract shared by hardware and simulated adapters."""

    def descriptor(self) -> ModalityDescriptor: ...

    def configuration_snapshot(self) -> Mapping[str, Any]: ...

    def connect(self, config: Any = None) -> None: ...

    def prepare(self, trial: TrialContext) -> PreparedInfo: ...

    def start(self, start_token: StartToken) -> None: ...

    def stop(self) -> StopReport: ...

    def health(self) -> HealthSnapshot: ...

    def close(self) -> None: ...


def coerce_config(config_type: type[ConfigT], value: Any) -> ConfigT:
    """Coerce dataclass, Pydantic model, or mapping configuration safely."""

    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected {config_type.__name__} or a mapping")

    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw["id"]
    allowed = {item.name for item in fields(config_type)}
    return config_type(**{key: val for key, val in raw.items() if key in allowed})


class QueuedSimulatedAdapter(ABC, Generic[ConfigT]):
    """Threaded simulator with deterministic values and fatal raw overflow."""

    config_type: type[ConfigT]

    def __init__(self, config: ConfigT | Mapping[str, Any] | None = None) -> None:
        self._initial_config = config
        self._config = coerce_config(self.config_type, config)
        self._state = AdapterState.DISCONNECTED
        self._state_lock = Lock()
        self._raw_queue: Queue[Any] = Queue(maxsize=self._config.queue_capacity)
        self._control_queue: Queue[Any] = Queue(maxsize=128)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._trial: TrialContext | None = None
        self._start_token: StartToken | None = None
        self._last_error: BaseException | None = None
        self._sequence = 0
        self._next_item_index = 0
        self._attempted_batches = 0
        self._batches_emitted = 0
        self._samples_emitted = 0
        self._injected_dropped_batches = 0
        self._raw_queue_overflows = 0
        self._first_data_ns: int | None = None
        self._last_data_ns: int | None = None
        self._rate_started_at: float | None = None
        self._last_jittered_ns: int | None = None
        self._rng_values = np.random.default_rng(self._config.seed)
        self._rng_faults = np.random.default_rng(self._config.seed ^ 0x5A17_2EED)
        self._rng_timing = np.random.default_rng(self._config.seed ^ 0x19C0_FFEE)
        self._last_device_status = DeviceStatus.DISCONNECTED

    @property
    def state(self) -> AdapterState:
        with self._state_lock:
            return self._state

    @property
    def raw_queue(self) -> Queue[Any]:
        """The bounded, loss-intolerant raw event queue."""

        return self._raw_queue

    @property
    def control_queue(self) -> Queue[Any]:
        """Low-volume status/metric events, separate from raw data."""

        return self._control_queue

    def configuration_snapshot(self) -> Mapping[str, Any]:
        """Return the fully resolved simulator configuration for provenance."""

        return asdict(self._config)

    @abstractmethod
    def descriptor(self) -> ModalityDescriptor:
        raise NotImplementedError

    @property
    @abstractmethod
    def _rate_hz(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def _items_per_batch(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _make_events(
        self,
        *,
        sequence: int,
        first_item_index: int,
        host_monotonic_ns: int,
    ) -> list[Any]:
        raise NotImplementedError

    def connect(self, config: ConfigT | Mapping[str, Any] | None = None) -> None:
        with self._state_lock:
            if self._state is not AdapterState.DISCONNECTED:
                raise AdapterLifecycleError(f"connect not allowed from {self._state.value}")
            self._config = coerce_config(
                self.config_type, self._initial_config if config is None else config
            )
            self._raw_queue = Queue(maxsize=self._config.queue_capacity)
            self._control_queue = Queue(maxsize=128)
            self._reset_random_generators()
            self._state = AdapterState.CONNECTED
        self._emit_status("connected", "simulated device connected")

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        self._emit_status("preparing", f"preparing for trial {trial.trial_uuid}")
        with self._state_lock:
            if self._state not in (AdapterState.CONNECTED, AdapterState.STOPPED):
                raise AdapterLifecycleError(f"prepare not allowed from {self._state.value}")
            if self._raw_queue.qsize():
                raise AdapterLifecycleError("raw queue must be drained before preparing another trial")
            self._trial = trial
            self._start_token = None
            self._last_error = None
            self._sequence = 0
            self._next_item_index = 0
            self._attempted_batches = 0
            self._batches_emitted = 0
            self._samples_emitted = 0
            self._injected_dropped_batches = 0
            self._raw_queue_overflows = 0
            self._first_data_ns = None
            self._last_data_ns = None
            self._rate_started_at = None
            self._last_jittered_ns = None
            self._stop_event.clear()
            self._reset_random_generators()
            self._state = AdapterState.PREPARED

        desc = self.descriptor()
        info = PreparedInfo(
            device_id=desc.device_id,
            modality=desc.modality,
            trial_uuid=str(trial.trial_uuid),
            clock_domain=desc.clock_domain,
            nominal_rate_hz=desc.nominal_rate_hz,
            channels=desc.channels,
            units=desc.units,
            queue_capacity=self._config.queue_capacity,
            metadata=dict(desc.metadata),
        )
        self._emit_status("prepared", f"prepared for trial {trial.trial_uuid}")
        return info

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
            self._stop_event.clear()
            self._rate_started_at = perf_counter()
            self._state = AdapterState.RUNNING
            self._thread = Thread(
                target=self._run_guarded,
                name=f"sim-{self.descriptor().device_id}",
                daemon=True,
            )
            self._thread.start()
        self._emit_status("recording", "simulated acquisition started")

    def stop(self) -> StopReport:
        state = self.state
        if state is AdapterState.CLOSED:
            raise AdapterLifecycleError("stop not allowed after close")
        if state in (AdapterState.DISCONNECTED, AdapterState.CONNECTED):
            raise AdapterLifecycleError(f"stop not allowed from {state.value}")

        self._emit_status("stopping", "stopping simulated acquisition")
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                self._set_fault(AdapterError("simulator worker did not stop within 5 seconds"))
        self._thread = None

        with self._state_lock:
            if self._state is not AdapterState.FAULTED:
                self._state = AdapterState.STOPPED
        self._emit_status(
            "faulted" if self._last_error else "stopped",
            str(self._last_error) if self._last_error else "simulated acquisition stopped",
        )
        return self._stop_report()

    def close(self) -> None:
        state = self.state
        if state is AdapterState.CLOSED:
            return
        if state in (AdapterState.RUNNING, AdapterState.PREPARED, AdapterState.FAULTED):
            try:
                self.stop()
            except AdapterLifecycleError:
                pass
        with self._state_lock:
            self._state = AdapterState.CLOSED
        self._emit_status("closed", "simulated device closed")

    def health(self) -> HealthSnapshot:
        desc = self.descriptor()
        depth = self._raw_queue.qsize()
        elapsed = perf_counter() - self._rate_started_at if self._rate_started_at else 0.0
        rate = self._samples_emitted / elapsed if elapsed > 0 else 0.0
        state = self.state
        device_status = self._device_status_for_state(state)
        if self._last_error is not None or state is AdapterState.FAULTED:
            health_status = HealthStatus.UNHEALTHY
        elif state in (AdapterState.CONNECTED, AdapterState.PREPARED, AdapterState.RUNNING, AdapterState.STOPPED):
            health_status = HealthStatus.HEALTHY
        else:
            health_status = HealthStatus.UNKNOWN
        return HealthSnapshot(
            device_id=desc.device_id,
            modality=desc.modality,
            status=health_status,
            device_status=device_status,
            connected=state in (
                AdapterState.CONNECTED,
                AdapterState.PREPARED,
                AdapterState.RUNNING,
                AdapterState.STOPPED,
            ),
            ready=state in (AdapterState.PREPARED, AdapterState.RUNNING),
            sampling=state is AdapterState.RUNNING,
            queue_depth=depth,
            queue_capacity=self._config.queue_capacity,
            last_data_host_monotonic_ns=self._last_data_ns,
            actual_sample_rate_hz=rate,
            nominal_sample_rate_hz=desc.nominal_rate_hz,
            dropped_packets=self._injected_dropped_batches * self._items_per_batch,
            message=str(self._last_error) if self._last_error else "ok",
            metrics={
                "batches_emitted": self._batches_emitted,
                "samples_emitted": self._samples_emitted,
                "injected_dropped_batches": self._injected_dropped_batches,
                "raw_queue_overflows": self._raw_queue_overflows,
                "queue_fill_ratio": depth / self._config.queue_capacity,
            },
        )

    def get_event(self, timeout: float | None = None) -> Any | None:
        """Read one raw event; return ``None`` on timeout."""

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

    def iter_events(self, timeout: float = 0.1) -> Iterator[Any]:
        while self.state is AdapterState.RUNNING or not self._raw_queue.empty():
            event = self.get_event(timeout=timeout)
            if event is not None:
                yield event

    def raise_if_faulted(self) -> None:
        if self._last_error is not None:
            raise AdapterError(str(self._last_error)) from self._last_error

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # worker faults must become observable health
            self._set_fault(exc)

    def _run(self) -> None:
        assert self._start_token is not None
        batch_period_ns = max(1, round(1_000_000_000 * self._items_per_batch / self._rate_hz))
        scheduled_ns = self._start_token.host_monotonic_ns

        while not self._stop_event.is_set():
            now_ns = perf_counter_ns()
            if self._config.realtime and scheduled_ns > now_ns:
                if self._stop_event.wait((scheduled_ns - now_ns) / 1_000_000_000):
                    break

            if (
                self._config.disconnect_after_batches is not None
                and self._attempted_batches >= self._config.disconnect_after_batches
            ):
                raise SimulatedDisconnectError(
                    f"injected disconnect after {self._attempted_batches} batches"
                )

            sequence = self._sequence
            first_index = self._next_item_index
            host_ns = self._jitter_timestamp(scheduled_ns)
            should_drop = self._should_drop_batch()

            self._attempted_batches += 1
            self._sequence += 1
            self._next_item_index += self._items_per_batch
            scheduled_ns += batch_period_ns

            if should_drop:
                self._injected_dropped_batches += 1
                continue

            events = self._make_events(
                sequence=sequence,
                first_item_index=first_index,
                host_monotonic_ns=host_ns,
            )
            if not events:
                raise AdapterError("simulator generated an empty event list")
            for event in events:
                self._enqueue_raw(event)
                if self._stop_event.is_set():
                    break
            if self._last_error is not None:
                break

            self._batches_emitted += 1
            self._samples_emitted += self._items_per_batch
            self._first_data_ns = host_ns if self._first_data_ns is None else self._first_data_ns
            self._last_data_ns = host_ns

            if not self._config.realtime:
                # Yield to the consumer without introducing nondeterministic data values.
                self._stop_event.wait(0.0001)

    def _enqueue_raw(self, event: Any) -> None:
        try:
            self._raw_queue.put_nowait(event)
        except Full as exc:
            self._raw_queue_overflows += 1
            error = RawQueueOverflowError(
                f"raw queue overflow for {self.descriptor().device_id} "
                f"(capacity={self._config.queue_capacity})"
            )
            self._set_fault(error)
            raise error from exc

    def _should_drop_batch(self) -> bool:
        ordinal = self._attempted_batches + 1
        periodic = bool(
            self._config.drop_every_n_batches
            and ordinal % self._config.drop_every_n_batches == 0
        )
        random_drop = bool(
            self._config.drop_probability
            and self._rng_faults.random() < self._config.drop_probability
        )
        return periodic or random_drop

    def _jitter_timestamp(self, scheduled_ns: int) -> int:
        jitter = 0
        if self._config.timestamp_jitter_ns:
            jitter = int(
                round(self._rng_timing.normal(0.0, self._config.timestamp_jitter_ns))
            )
        value = max(0, scheduled_ns + jitter)
        if self._last_jittered_ns is not None:
            value = max(value, self._last_jittered_ns + 1)
        self._last_jittered_ns = value
        return value

    def device_time_ns(self, item_index: int, rate_hz: float | None = None) -> int:
        """Deterministic simulated device clock including configured drift."""

        rate = self._rate_hz if rate_hz is None else rate_hz
        nominal = item_index * 1_000_000_000 / rate
        return int(round(nominal * (1.0 + self._config.clock_drift_ppm * 1e-6)))

    def _set_fault(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._last_error is None:
                self._last_error = exc
            if self._state is not AdapterState.CLOSED:
                self._state = AdapterState.FAULTED
        self._stop_event.set()
        self._emit_status("faulted", str(exc), error_code=type(exc).__name__)

    def _stop_report(self) -> StopReport:
        desc = self.descriptor()
        return StopReport(
            device_id=desc.device_id,
            modality=desc.modality,
            batches_emitted=self._batches_emitted,
            samples_emitted=self._samples_emitted,
            injected_dropped_batches=self._injected_dropped_batches,
            raw_queue_overflows=self._raw_queue_overflows,
            first_data_monotonic_ns=self._first_data_ns,
            last_data_monotonic_ns=self._last_data_ns,
            fault=str(self._last_error) if self._last_error else None,
        )

    def _reset_random_generators(self) -> None:
        self._rng_values = np.random.default_rng(self._config.seed)
        self._rng_faults = np.random.default_rng(self._config.seed ^ 0x5A17_2EED)
        self._rng_timing = np.random.default_rng(self._config.seed ^ 0x19C0_FFEE)

    def _event_common(self, host_monotonic_ns: int) -> dict[str, Any]:
        desc = self.descriptor()
        return {
            "session_uuid": str(self._trial.session_uuid)
            if self._trial and self._trial.session_uuid is not None
            else None,
            "trial_uuid": str(self._trial.trial_uuid) if self._trial else None,
            "device_id": desc.device_id,
            "modality": desc.modality,
            "clock_domain": desc.clock_domain,
            "host_monotonic_ns": host_monotonic_ns,
            "host_utc_ns": time_ns(),
        }

    def _emit_status(self, status: str, message: str, error_code: str | None = None) -> None:
        """Best-effort control event; health state remains the source of truth."""

        status_map = {
            "connected": DeviceStatus.CONNECTED,
            "preparing": DeviceStatus.PREPARING,
            "prepared": DeviceStatus.READY,
            "recording": DeviceStatus.RECORDING,
            "stopping": DeviceStatus.STOPPING,
            "stopped": DeviceStatus.CONNECTED,
            "faulted": DeviceStatus.FAULT,
            "closed": DeviceStatus.CLOSED,
        }
        device_status = status_map.get(status)
        if device_status is None:
            return
        try:
            kwargs = self._event_common(perf_counter_ns())
            kwargs.update(
                status=device_status,
                previous_status=self._last_device_status,
                message=message,
                error_code=error_code,
            )
            event = DeviceStatusEvent(**kwargs)
        except (TypeError, ValueError):
            return
        self._last_device_status = device_status
        try:
            self._control_queue.put_nowait(event)
        except Full:
            # Control events never displace or mask raw samples; health() remains available.
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


__all__ = [
    "AdapterError",
    "AdapterLifecycleError",
    "AdapterState",
    "DeviceConfig",
    "HealthSnapshot",
    "ModalityAdapter",
    "ModalityDescriptor",
    "PreparedInfo",
    "QueuedSimulatedAdapter",
    "RawQueueOverflowError",
    "SimulatedDisconnectError",
    "SimulationConfig",
    "StartToken",
    "StopReport",
    "TrialContext",
    "coerce_config",
]
