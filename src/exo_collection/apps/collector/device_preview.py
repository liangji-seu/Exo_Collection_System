"""Single-modality real-time preview worker.

Each ``ModalityPreviewWorker`` runs in a dedicated ``spawn`` subprocess and
manages exactly **one** adapter.  It publishes ``WorkerEvent``-format messages
to the parent process so the GUI can display live signal previews and health
without ever instantiating a Writer, Catalog, TrialPackageBuilder, or creating
any Session/Trial/Manifest/H5/bin files on disk.
"""

from __future__ import annotations

import logging
import multiprocessing
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Full
from time import perf_counter
from typing import Any

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.acquisition.preview import build_preview_event
from exo_collection.adapters.base import (
    AdapterState,
    ModalityAdapter,
    ModalityDescriptor,
    StartToken,
    TrialContext,
)
from exo_collection.domain.events import (
    DeviceStatusEvent,
    FrameBatch,
    SampleBatch,
    SyncPulseEvent,
)

_log = logging.getLogger(__name__)

DEFAULT_PREVIEW_QUEUE_SIZE = 128
DEFAULT_HEALTH_POLL_INTERVAL_S = 0.5
DEFAULT_PREVIEW_DOWNSAMPLE_MAX_S = 1.0 / 15.0  # ~15 fps max


def _preview_rate_limit_key(event: WorkerEvent) -> tuple[str, int | None]:
    """Return the independent UI stream represented by a preview event.

    Packet-per-channel ultrasound produces four independent events.  Sharing
    one limiter across them discards three interleaved channels, so its channel
    index is part of the key.  Batched ultrasound, IMU and encoder previews
    continue to use one key per modality.
    """

    modality = str(event.modality or event.payload.get("modality") or "")
    if modality == "ultrasound":
        raw_channel = event.payload.get("channel_index")
        if raw_channel is not None:
            try:
                channel_index = int(raw_channel)
            except (TypeError, ValueError):
                channel_index = None
            if channel_index is not None and 0 <= channel_index <= 3:
                return modality, channel_index
    return modality, None


def _preview_is_due(
    event: WorkerEvent,
    *,
    now: float,
    last_sent_by_stream: dict[tuple[str, int | None], float],
    interval_s: float = DEFAULT_PREVIEW_DOWNSAMPLE_MAX_S,
) -> bool:
    """Apply the best-effort preview cap independently to each UI stream."""

    key = _preview_rate_limit_key(event)
    last_sent = last_sent_by_stream.get(key)
    if last_sent is not None and now - last_sent < interval_s:
        return False
    last_sent_by_stream[key] = now
    return True


# ── Public types ───────────────────────────────────────────────────────────


class ModalityPreviewOutput:
    """Snapshot of a preview worker's event queue after one poll."""

    __slots__ = ("events", "alive", "exitcode")

    def __init__(
        self,
        events: list[WorkerEvent],
        *,
        alive: bool,
        exitcode: int | None = None,
    ) -> None:
        self.events = events
        self.alive = alive
        self.exitcode = exitcode


class ModalityPreviewHandle(ABC):
    """Abstract handle for a single-modality preview subprocess."""

    @property
    @abstractmethod
    def is_alive(self) -> bool: ...

    @property
    @abstractmethod
    def exitcode(self) -> int | None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def request_stop(self) -> None: ...

    @abstractmethod
    def poll_events(self, limit: int = 100) -> list[WorkerEvent]: ...

    @abstractmethod
    def join(self, timeout: float | None = None) -> int | None: ...

    @abstractmethod
    def terminate(self, timeout: float = 5.0) -> int | None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def modality(self) -> str: ...

    @property
    @abstractmethod
    def device_id(self) -> str: ...

    @property
    @abstractmethod
    def simulated(self) -> bool: ...


# ── Adapter factory protocol ───────────────────────────────────────────────


AdapterFactory = Callable[[], ModalityAdapter]


@dataclass(frozen=True, slots=True)
class ProfileModalityAdapterFactory:
    """Pickle-safe factory used by Windows ``spawn`` preview processes."""

    profile_key: str
    modality: str
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __call__(self) -> ModalityAdapter:
        from exo_collection.configuration import build_adapter, load_device_profile

        return build_adapter(
            load_device_profile(self.profile_key),
            self.modality,
            self.overrides if self.profile_key == "hardware" else {},
        )


# ── Subprocess target ──────────────────────────────────────────────────────


def _preview_runner_target(
    event_queue: multiprocessing.Queue,
    stop_pipe: multiprocessing.connection.Connection,
    adapter_factory: AdapterFactory,
    device_id: str,
    modality: str,
    simulated: bool,
    health_poll_interval_s: float = DEFAULT_HEALTH_POLL_INTERVAL_S,
) -> None:
    """Entry point executed in the spawned subprocess.

    Lifecycle (mandatory, non-skippable):
        connect -> start -> loop (read_batch/read_frames + health) ->
        stop -> disconnect

    This function NEVER instantiates Writer, Catalog, TrialPackageBuilder,
    Session, Trial, Manifest, H5, or bin files.
    """
    # Suppress __del__-based ResourceWarning inside the subprocess.
    adapter: ModalityAdapter | None = None
    try:
        from exo_collection.logging_setup import configure_subprocess_logging
        configure_subprocess_logging()

        adapter = adapter_factory()
        # ---- connect ----
        _send_event(
            event_queue,
            WorkerEventType.STATE,
            modality=modality,
            device_id=device_id,
            payload={"state": "CONNECTING", "modality": modality, "device_id": device_id, "simulated": simulated},
            message=f"Preview worker connecting {modality} ({device_id})",
        )
        adapter.connect()

        descriptor = adapter.descriptor()
        get_event = getattr(adapter, "get_event", None)
        if not callable(get_event):
            raise TypeError(
                f"{type(adapter).__name__} does not expose the raw get_event API"
            )

        # ---- prepare/start with an in-memory preview-only context ----
        _send_event(
            event_queue,
            WorkerEventType.STATE,
            modality=modality,
            device_id=device_id,
            payload={"state": "PREVIEW_STARTING", "modality": modality, "device_id": device_id, "simulated": simulated},
            message=f"Preview worker starting {modality} ({device_id})",
        )
        dummy_context = TrialContext(
            trial_uuid="00000000-0000-0000-0000-000000000000",
            session_uuid="00000000-0000-0000-0000-000000000000",
            condition={"purpose": "preview_only"},
            recording_dir=None,
        )
        adapter.prepare(dummy_context)
        adapter.start(StartToken())

        # ---- preview loop ----
        last_health = perf_counter()
        last_preview_send_by_stream: dict[tuple[str, int | None], float] = {}
        ready_sent = False
        while not stop_pipe.poll():
            # Drain raw events but only keep the *latest* preview per independent
            # UI stream (ultrasound channel, IMU modality, …).  This prevents stale
            # frames from accumulating across the three internal queues and ensures
            # the display always reflects the freshest data.
            latest_by_key: dict[tuple[str, int | None], WorkerEvent] = {}
            for _ in range(32):
                if stop_pipe.poll():
                    break
                raw = get_event(timeout=0.01 if not ready_sent else 0.0)
                if raw is None:
                    break
                preview = _build_preview_event(
                    raw, modality, descriptor.device_id, descriptor, simulated
                )
                if preview is not None:
                    observed_raw_data = isinstance(
                        raw, (FrameBatch, SampleBatch, SyncPulseEvent)
                    )
                    if observed_raw_data and not ready_sent:
                        ready_sent = True
                        _send_event(
                            event_queue,
                            WorkerEventType.STATE,
                            modality=modality,
                            device_id=descriptor.device_id,
                            payload={
                                "state": "READY",
                                "modality": modality,
                                "device_id": descriptor.device_id,
                                "simulated": simulated,
                                "descriptor": _descriptor_dict(descriptor),
                                "observed_raw_data": True,
                            },
                            message=(
                                f"Preview {modality} ({descriptor.device_id}) READY"
                            ),
                        )
                    adapter_ready_event = (
                        preview.event_type is WorkerEventType.STATE
                        and preview.payload.get("state") == "READY"
                    )
                    if adapter_ready_event:
                        continue
                    if preview.event_type is not WorkerEventType.PREVIEW:
                        _send_event_raw(event_queue, preview)
                    else:
                        key = _preview_rate_limit_key(preview)
                        latest_by_key[key] = preview

            # Send only the latest preview per stream, subject to the rate cap.
            now = perf_counter()
            for preview in latest_by_key.values():
                if _preview_is_due(
                    preview,
                    now=now,
                    last_sent_by_stream=last_preview_send_by_stream,
                ):
                    _send_event_raw(event_queue, preview)

            # Health telemetry
            now = perf_counter()
            if now - last_health >= health_poll_interval_s:
                last_health = now
                health = adapter.health()
                _send_event(
                    event_queue,
                    WorkerEventType.HEALTH,
                    modality=modality,
                    device_id=device_id,
                    payload={
                        "modality": modality,
                        "device_id": descriptor.device_id,
                        "simulated": simulated,
                        "status": health.device_status.value,
                        "health_status": health.status.value,
                        "connected": health.connected,
                        "ready": health.ready,
                        "sampling": health.sampling,
                        "sample_count": health.metrics.get("samples_emitted", 0),
                        "actual_sample_rate_hz": health.actual_sample_rate_hz,
                        "nominal_sample_rate_hz": health.nominal_sample_rate_hz,
                        "queue_depth": health.queue_depth,
                        "queue_capacity": health.queue_capacity,
                        "dropped_packets": health.dropped_packets,
                        "message": health.message,
                        "sampled_at_utc": health.sampled_at_utc.isoformat(),
                    },
                )

            # Brief sleep to prevent tight-loop CPU burn
            import time as _time
            _time.sleep(0.002)

        # ---- stop + close ----
        _send_event(
            event_queue,
            WorkerEventType.STATE,
            modality=modality,
            device_id=device_id,
            payload={"state": "STOPPING", "modality": modality, "device_id": device_id},
            message=f"Preview worker stopping {modality} ({device_id})",
        )
        adapter.stop()
        adapter.close()
        _send_event(
            event_queue,
            WorkerEventType.STATE,
            modality=modality,
            device_id=device_id,
            payload={"state": "DISCONNECTED", "modality": modality, "device_id": device_id},
            message=f"Preview worker disconnected {modality} ({device_id})",
        )

    except Exception:
        tb = traceback.format_exc()
        _send_event(
            event_queue,
            WorkerEventType.FAILED,
            modality=modality,
            device_id=device_id,
            payload={
                "state": "FAILED",
                "modality": modality,
                "device_id": device_id,
                "simulated": simulated,
                "traceback": tb,
            },
            message=f"Preview worker failed: {tb.splitlines()[-1] if tb else 'unknown'}",
        )
        _log.error("Preview worker %s (%s) crashed: %s", modality, device_id, tb)
    finally:
        if adapter is not None:
            try:
                if adapter.state not in (AdapterState.DISCONNECTED, AdapterState.CLOSED):
                    try:
                        if adapter.state in (AdapterState.RUNNING, AdapterState.PREPARED, AdapterState.STOPPED, AdapterState.FAULTED):
                            adapter.stop()
                    except Exception:
                        pass
                    try:
                        adapter.close()
                    except Exception:
                        pass
            except Exception:
                pass


def _descriptor_dict(descriptor: ModalityDescriptor) -> dict[str, Any]:
    return {
        "device_id": descriptor.device_id,
        "modality": descriptor.modality,
        "clock_domain": descriptor.clock_domain,
        "nominal_rate_hz": descriptor.nominal_rate_hz,
        "channels": descriptor.channels,
        "units": descriptor.units,
        "metadata": dict(descriptor.metadata),
    }


def _build_preview_event(
    raw: Any,
    modality: str,
    device_id: str,
    descriptor: ModalityDescriptor,
    simulated: bool,
) -> WorkerEvent | None:
    if isinstance(raw, (FrameBatch, SampleBatch)):
        if raw.modality != modality:
            raise ValueError(
                f"preview worker for {modality!r} received {raw.modality!r} data"
            )
        return build_preview_event(
            raw,
            None,
            extra_payload={
                "device_id": device_id,
                "simulated": simulated,
                "descriptor_device_id": descriptor.device_id,
            },
        )
    if isinstance(raw, SyncPulseEvent):
        return WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality=modality,
            trial_uuid=None,
            payload={
                "modality": modality,
                "device_id": device_id,
                "simulated": simulated,
                "edge_type": raw.edge_type.value,
                "amplitude": raw.amplitude,
                "pulse_id": raw.pulse_id,
                "host_monotonic_ns": raw.host_monotonic_ns,
            },
        )
    if isinstance(raw, DeviceStatusEvent):
        return WorkerEvent(
            event_type=WorkerEventType.STATE,
            modality=modality,
            trial_uuid=None,
            payload={"state": raw.status.value, "modality": modality,
                     "device_id": device_id, "simulated": simulated},
            message=raw.message or "",
        )
    return None


def _send_event(
    queue: multiprocessing.Queue,
    event_type: WorkerEventType,
    *,
    modality: str,
    device_id: str,
    payload: dict[str, Any],
    message: str = "",
) -> None:
    _send_event_raw(
        queue,
        WorkerEvent(
            event_type=event_type,
            modality=modality,
            trial_uuid=None,
            payload=payload,
            message=message,
        ),
    )


def _send_event_raw(queue: multiprocessing.Queue, event: WorkerEvent) -> None:
    if event.event_type is WorkerEventType.PREVIEW:
        try:
            queue.put_nowait(event)
        except Full:
            # Live preview is intentionally best-effort; raw acquisition stays
            # inside the adapter queue and is never represented by this queue.
            return
        return
    try:
        queue.put(event, timeout=0.5)
    except Full:
        _log.error("preview control event queue is full: %s", event.event_type.value)


# ── Production handle ──────────────────────────────────────────────────────


class ModalityPreviewProcessHandle(ModalityPreviewHandle):
    """Production handle that wraps a multiprocessing.Process."""

    def __init__(
        self,
        adapter_factory: AdapterFactory,
        *,
        device_id: str = "",
        modality: str = "",
        simulated: bool = True,
        health_poll_interval_s: float = DEFAULT_HEALTH_POLL_INTERVAL_S,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._device_id = device_id
        self._modality = modality
        self._simulated = simulated
        self._health_poll_interval_s = health_poll_interval_s

        ctx = multiprocessing.get_context("spawn")
        self._event_queue: multiprocessing.Queue = ctx.Queue(
            maxsize=DEFAULT_PREVIEW_QUEUE_SIZE
        )
        self._stop_pipe_recv, self._stop_pipe_send = ctx.Pipe(duplex=False)
        self._process = ctx.Process(
            target=_preview_runner_target,
            args=(
                self._event_queue,
                self._stop_pipe_recv,
                self._adapter_factory,
                self._device_id,
                self._modality,
                self._simulated,
                self._health_poll_interval_s,
            ),
            name=f"preview-{self._modality}-{self._device_id}",
            daemon=True,
        )
        self._started = False
        self._stopped = False
        self._closed = False

    @property
    def is_alive(self) -> bool:
        if not self._started:
            return False
        return self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def start(self) -> None:
        if self._started:
            raise RuntimeError("preview worker can only be started once")
        if self._closed:
            raise RuntimeError("preview worker is closed")
        self._process.start()
        self._started = True

    def request_stop(self) -> None:
        if self._stopped:
            return
        if not self._started:
            self._stopped = True
            return
        try:
            self._stop_pipe_send.send("stop")
        except (BrokenPipeError, OSError):
            pass
        self._stopped = True

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        events: list[WorkerEvent] = []
        for _ in range(limit):
            try:
                event = self._event_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
            events.append(event)
        return events

    def join(self, timeout: float | None = None) -> int | None:
        if not self._started:
            return None
        self._process.join(timeout=timeout)
        return self._process.exitcode

    def terminate(self, timeout: float = 5.0) -> int | None:
        if not self._started:
            return None
        if not self._process.is_alive():
            return self._process.exitcode
        self._process.terminate()
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1.0)
        return self._process.exitcode

    def close(self) -> None:
        if self._closed:
            return
        if self._started:
            self.request_stop()
            self.join(timeout=0.25)
            if self._process.is_alive():
                raise RuntimeError("cannot close a running preview worker")
        try:
            self._event_queue.close()
        except Exception:
            pass
        try:
            self._stop_pipe_send.close()
        except Exception:
            pass
        try:
            self._stop_pipe_recv.close()
        except Exception:
            pass
        self._closed = True

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def simulated(self) -> bool:
        return self._simulated


# ── Synchronous test runner ────────────────────────────────────────────────


class InProcessPreviewRunner:
    """Synchronous, in-process preview runner for unit tests.

    Provides a deterministic way to exercise the preview lifecycle without
    spawning subprocesses or importing real device SDKs.
    """

    def __init__(
        self,
        adapter_factory: AdapterFactory | None = None,
        *,
        device_id: str = "test_device",
        modality: str = "imu",
        simulated: bool = True,
        health_poll_interval_s: float = DEFAULT_HEALTH_POLL_INTERVAL_S,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._device_id = device_id
        self._modality = modality
        self._simulated = simulated
        self._health_poll_interval_s = health_poll_interval_s
        self._events: list[WorkerEvent] = []
        self._adapter: ModalityAdapter | None = None
        self._alive = False
        self._stopped = False
        self._ready = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def exitcode(self) -> int | None:
        return 0 if not self._alive else None

    def start(self) -> None:
        if self._alive or self._adapter is not None:
            raise RuntimeError("InProcessPreviewRunner can only be started once")
        if self._adapter_factory is None:
            raise RuntimeError("InProcessPreviewRunner requires an adapter_factory")
        self._adapter = self._adapter_factory()
        self._alive = True
        self._events.append(
            WorkerEvent(
                event_type=WorkerEventType.STATE,
                modality=self._modality,
                trial_uuid=None,
                payload={"state": "CONNECTING", "modality": self._modality,
                         "device_id": self._device_id, "simulated": self._simulated},
                message=f"Preview worker connecting {self._modality} ({self._device_id})",
            )
        )
        try:
            self._adapter.connect()
            self._adapter.prepare(
                TrialContext(
                    trial_uuid="00000000-0000-0000-0000-000000000000",
                    session_uuid="00000000-0000-0000-0000-000000000000",
                    condition={"purpose": "preview_only"},
                    recording_dir=None,
                )
            )
            self._adapter.start(StartToken())
        except Exception:
            self._alive = False
            try:
                self._adapter.close()
            except Exception:
                pass
            raise

    def request_stop(self) -> None:
        self._stopped = True

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        if self._adapter is None:
            return []
        if not self._stopped and limit > 0:
            raw = self._adapter.get_event(timeout=0.01)
            if raw is not None:
                descriptor = self._adapter.descriptor()
                event = _build_preview_event(
                    raw, self._modality, self._device_id, descriptor, self._simulated
                )
                if event is not None:
                    if not self._ready and isinstance(
                        raw, (FrameBatch, SampleBatch, SyncPulseEvent)
                    ):
                        self._ready = True
                        self._events.append(
                            WorkerEvent(
                                event_type=WorkerEventType.STATE,
                                modality=self._modality,
                                trial_uuid=None,
                                payload={
                                    "state": "READY",
                                    "modality": self._modality,
                                    "device_id": self._device_id,
                                    "simulated": self._simulated,
                                    "observed_raw_data": True,
                                },
                                message=(
                                    f"Preview {self._modality} "
                                    f"({self._device_id}) READY"
                                ),
                            )
                        )
                    adapter_ready_event = (
                        event.event_type is WorkerEventType.STATE
                        and event.payload.get("state") == "READY"
                    )
                    if not adapter_ready_event:
                        self._events.append(event)
        result = self._events[:limit]
        self._events = self._events[limit:]
        return result

    def join(self, timeout: float | None = None) -> int | None:
        if self._adapter is not None:
            try:
                self._adapter.stop()
            except Exception:
                pass
            try:
                self._adapter.close()
            except Exception:
                pass
        self._alive = False
        return 0

    def terminate(self, timeout: float = 5.0) -> int | None:
        self._alive = False
        return self.join()

    def close(self) -> None:
        self.request_stop()
        self.join()
        self._events.clear()

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def simulated(self) -> bool:
        return self._simulated


__all__ = [
    "ModalityPreviewOutput",
    "ModalityPreviewHandle",
    "ModalityPreviewProcessHandle",
    "InProcessPreviewRunner",
    "AdapterFactory",
    "ProfileModalityAdapterFactory",
    "DEFAULT_PREVIEW_QUEUE_SIZE",
]
