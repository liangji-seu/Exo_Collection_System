"""Single-modality real-time preview worker.

Each ``ModalityPreviewWorker`` runs in a dedicated ``spawn`` subprocess and
manages exactly **one** adapter.  It publishes ``WorkerEvent``-format messages
to the parent process so the GUI can display live signal previews and health
without ever instantiating a Writer, Catalog, TrialPackageBuilder, or creating
any Session/Trial/Manifest/H5/bin files on disk.

When the UI issues a recording command, the preview worker forwards raw domain
events into a bounded recording queue.  The CollectorWorker drains that queue
through a ``StreamProxyAdapter`` so recording never stops or reconnects the
real hardware Adapter.
"""

from __future__ import annotations

import logging
import multiprocessing
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from time import perf_counter
from typing import Any

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.acquisition.preview import build_preview_event
from exo_collection.acquisition.recording_stream import (
    RecordingCommand,
    RecordingCommandKind,
    RecordingStreamEndpoint,
    RecordingStreamError,
    RecordingStreamOverflow,
    RecordingStreamProducer,
    normalize_trial_uuid,
)
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
DEFAULT_PREVIEW_DOWNSAMPLE_MAX_S = 1.0 / 30.0  # ~30 fps max


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
    def begin_recording(self, trial_uuid: str) -> None: ...

    @abstractmethod
    def end_recording(self, trial_uuid: str) -> None: ...

    @abstractmethod
    def discard_recording_backlog(self) -> int:
        """Discard stale queued recording messages while no Trial is active."""
        ...

    @property
    @abstractmethod
    def recording_endpoint(self) -> RecordingStreamEndpoint | None: ...

    def request_start_recording(self, trial_uuid: str) -> None:
        self.begin_recording(trial_uuid)

    def request_stop_recording(self, trial_uuid: str) -> None:
        self.end_recording(trial_uuid)

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
    raw_recording_queue: multiprocessing.Queue | None = None,
    control_pipe: multiprocessing.connection.Connection | None = None,
) -> None:
    """Entry point executed in the spawned subprocess.

    Lifecycle (mandatory, non-skippable):
        connect -> start -> loop (read_batch/read_frames + health) ->
        stop -> disconnect

    This function NEVER instantiates Writer, Catalog, TrialPackageBuilder,
    Session, Trial, Manifest, H5, or bin files.

    If ``raw_recording_queue`` and ``control_pipe`` are provided, the worker
    supports recording commands:
    - START_RECORDING(trial_uuid):  write START boundary, forward raw events
    - STOP_RECORDING(trial_uuid):   stop forwarding, write END boundary
    - SHUTDOWN:                     exit the preview loop

    START, END, and raw events are written by the same producer process on
    the same queue, guaranteeing ordering.
    """
    adapter: ModalityAdapter | None = None
    try:
        from exo_collection.logging_setup import configure_subprocess_logging

        configure_subprocess_logging()
        adapter = adapter_factory()
        _send_event(
            event_queue,
            WorkerEventType.STATE,
            modality=modality,
            device_id=device_id,
            payload={
                "state": "CONNECTING",
                "modality": modality,
                "device_id": device_id,
                "simulated": simulated,
            },
            message=f"Preview worker connecting {modality} ({device_id})",
        )
        adapter.connect()
        descriptor = adapter.descriptor()
        descriptor_payload = _descriptor_dict(descriptor)
        config_snapshot = dict(adapter.configuration_snapshot())
        get_event = getattr(adapter, "get_event", None)
        if not callable(get_event):
            raise TypeError(
                f"{type(adapter).__name__} does not expose the raw get_event API"
            )

        _send_event(
            event_queue,
            WorkerEventType.STATE,
            modality=modality,
            device_id=device_id,
            payload={
                "state": "PREVIEW_STARTING",
                "modality": modality,
                "device_id": device_id,
                "simulated": simulated,
            },
            message=f"Preview worker starting {modality} ({device_id})",
        )
        adapter.prepare(
            TrialContext(
                trial_uuid="00000000-0000-0000-0000-000000000000",
                session_uuid="00000000-0000-0000-0000-000000000000",
                condition={"purpose": "preview_only"},
                recording_dir=None,
            )
        )
        adapter.start(StartToken())

        producer = (
            RecordingStreamProducer(
                raw_recording_queue,
                device_id=descriptor.device_id,
                modality=modality,
                descriptor=descriptor_payload,
                configuration_snapshot=config_snapshot,
            )
            if raw_recording_queue is not None
            else None
        )
        last_health = perf_counter()
        last_preview_send_by_stream: dict[tuple[str, int | None], float] = {}
        preview_round_robin_cursor = 0
        ready_sent = False
        shutdown_requested = False

        def send_ack(status: str, trial_uuid: str | None, message: str = "") -> None:
            if control_pipe is None:
                return
            try:
                control_pipe.send(
                    {
                        "status": status,
                        "trial_uuid": trial_uuid,
                        "modality": modality,
                        "message": message,
                    }
                )
            except (BrokenPipeError, OSError):
                pass

        def fail_recording(trial_uuid: str | None, message: str) -> None:
            if producer is not None:
                producer.abort(message)
            _send_event(
                event_queue,
                WorkerEventType.FAILED,
                modality=modality,
                device_id=descriptor.device_id,
                trial_uuid=trial_uuid,
                payload={
                    "state": "FAULT",
                    "modality": modality,
                    "device_id": descriptor.device_id,
                    "simulated": simulated,
                    "fault": message,
                    "trial_uuid": trial_uuid,
                },
                message=message,
            )
            send_ack("FAULT", trial_uuid, message)

        while not shutdown_requested and not stop_pipe.poll():
            while control_pipe is not None and control_pipe.poll():
                try:
                    command = control_pipe.recv()
                except (EOFError, OSError):
                    shutdown_requested = True
                    break
                if not isinstance(command, RecordingCommand):
                    fail_recording(None, "invalid recording control command")
                    continue
                try:
                    if command.kind is RecordingCommandKind.START_RECORDING:
                        if producer is None:
                            raise RecordingStreamError("recording stream unavailable")
                        producer.begin(command.trial_uuid or "")
                        send_ack("STARTED", command.trial_uuid)
                    elif command.kind is RecordingCommandKind.STOP_RECORDING:
                        if producer is None:
                            raise RecordingStreamError("recording stream unavailable")
                        producer.end(command.trial_uuid or "")
                        send_ack("STOPPED", command.trial_uuid)
                    elif command.kind is RecordingCommandKind.SHUTDOWN:
                        if producer is not None and producer.recording:
                            producer.abort("preview worker shutdown during recording")
                        send_ack("SHUTDOWN", command.trial_uuid)
                        shutdown_requested = True
                        break
                except (RecordingStreamError, ValueError) as exc:
                    fail_recording(command.trial_uuid, str(exc))

            if shutdown_requested:
                break

            latest_by_key: dict[tuple[str, int | None], WorkerEvent] = {}
            for _ in range(32):
                if stop_pipe.poll() or (
                    control_pipe is not None and control_pipe.poll()
                ):
                    break
                raw = get_event(timeout=0.01 if not ready_sent else 0.0)
                if raw is None:
                    break

                if producer is not None and producer.recording:
                    try:
                        producer.forward(raw)
                    except RecordingStreamOverflow as exc:
                        failed_trial_uuid = producer.active_trial_uuid
                        fail_recording(failed_trial_uuid, str(exc))

                preview = _build_preview_event(
                    raw, modality, descriptor.device_id, descriptor, simulated
                )
                if preview is None:
                    continue
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
                            "descriptor": descriptor_payload,
                            "configuration_snapshot": config_snapshot,
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
                    latest_by_key[_preview_rate_limit_key(preview)] = preview

            now = perf_counter()
            preview_round_robin_cursor = _send_latest_previews_fairly(
                event_queue,
                latest_by_key,
                now=now,
                last_sent_by_stream=last_preview_send_by_stream,
                cursor=preview_round_robin_cursor,
            )

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
        "display_name": descriptor.display_name,
        "clock_domain": descriptor.clock_domain,
        "event_kind": descriptor.event_kind,
        "nominal_rate_hz": descriptor.nominal_rate_hz,
        "channels": list(descriptor.channels),
        "units": list(descriptor.units),
        "sample_shape": list(descriptor.sample_shape),
        "dtype": descriptor.dtype,
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
                "preview_labels": list(
                    descriptor.metadata.get("preview_labels") or []
                ),
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
    trial_uuid: str | None = None,
) -> None:
    _send_event_raw(
        queue,
        WorkerEvent(
            event_type=event_type,
            modality=modality,
            trial_uuid=trial_uuid,
            payload=payload,
            message=message,
        ),
    )


def _send_event_raw(queue: multiprocessing.Queue, event: WorkerEvent) -> bool:
    """Best-effort enqueue and report whether the event was accepted.

    Preview delivery must never block acquisition.  Returning the result lets
    the fair scheduler advance only after a stream actually obtained queue
    capacity, instead of treating a dropped attempt as a delivered frame.
    """
    if event.event_type is WorkerEventType.PREVIEW:
        try:
            queue.put_nowait(event)
        except Full:
            # Live preview is intentionally best-effort; raw acquisition stays
            # inside the adapter queue and is never represented by this queue.
            return False
        return True
    try:
        queue.put(event, timeout=0.5)
    except Full:
        _log.error("preview control event queue is full: %s", event.event_type.value)
        return False
    return True


def _send_latest_previews_fairly(
    queue: multiprocessing.Queue,
    latest_by_key: dict[tuple[str, int | None], WorkerEvent],
    *,
    now: float,
    last_sent_by_stream: dict[tuple[str, int | None], float],
    cursor: int,
    interval_s: float = DEFAULT_PREVIEW_DOWNSAMPLE_MAX_S,
) -> int:
    """Non-blocking round-robin delivery of the latest independent streams.

    A bounded queue may have only one free slot per acquisition-loop pass.  A
    fixed ch0..ch3 iteration order would let channel 0 repeatedly claim that
    slot and starve later channels.  This routine starts after the last stream
    that *successfully* enqueued, so recurring pressure still gives every
    channel a turn.  Failed enqueue attempts do not consume the rate limit.
    """

    keys = sorted(
        latest_by_key,
        key=lambda item: (item[0], -1 if item[1] is None else item[1]),
    )
    if not keys:
        return cursor

    start = cursor % len(keys)
    next_cursor = start
    for offset in range(len(keys)):
        index = (start + offset) % len(keys)
        key = keys[index]
        last_sent = last_sent_by_stream.get(key)
        if last_sent is not None and now - last_sent < interval_s:
            continue
        if _send_event_raw(queue, latest_by_key[key]):
            last_sent_by_stream[key] = now
            next_cursor = (index + 1) % len(keys)
    return next_cursor


# ── Production handle ──────────────────────────────────────────────────────


DEFAULT_RECORDING_QUEUE_SIZE = 2048


class ModalityPreviewProcessHandle(ModalityPreviewHandle):
    """Production handle that wraps a multiprocessing.Process.

    Owns a bounded raw recording queue and duplex control pipe so the
    CollectorWorker can drain raw events without stopping the adapter.
    """

    def __init__(
        self,
        adapter_factory: AdapterFactory,
        *,
        device_id: str = "",
        modality: str = "",
        simulated: bool = True,
        health_poll_interval_s: float = DEFAULT_HEALTH_POLL_INTERVAL_S,
        recording_queue_size: int = DEFAULT_RECORDING_QUEUE_SIZE,
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

        # Duplex control pipe: UI sends commands, worker sends ACK
        self._control_pipe_local, self._control_pipe_remote = ctx.Pipe(duplex=True)

        # Bounded raw recording queue for forwarding domain events
        self._raw_recording_queue: multiprocessing.Queue = ctx.Queue(
            maxsize=recording_queue_size
        )

        # Pass the receiving end of stop_pipe (recv) and the remote end of
        # control_pipe to the subprocess.
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
                self._raw_recording_queue,
                self._control_pipe_remote,
            ),
            name=f"preview-{self._modality}-{self._device_id}",
            daemon=True,
        )
        self._started = False
        self._stopped = False
        self._closed = False
        self._process_closed = False
        self._final_exitcode: int | None = None

        # Recording state tracking
        self._recording_active = False
        self._active_trial_uuid: str | None = None
        self._recording_ack_received = False

        # Descriptor saved after preview worker reports READY
        self._cached_descriptor: dict[str, Any] | None = None
        self._cached_config_snapshot: dict[str, Any] | None = None

    @property
    def is_alive(self) -> bool:
        if not self._started or self._process_closed:
            return False
        return self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        if self._process_closed:
            return self._final_exitcode
        return self._process.exitcode

    def start(self) -> None:
        if self._started:
            raise RuntimeError("preview worker can only be started once")
        if self._closed:
            raise RuntimeError("preview worker is closed")
        try:
            self._process.start()
        except BaseException:
            # A rare late spawn failure can occur after ``_popen`` is
            # installed.  Remember that partial start so close() can still
            # terminate/join the child instead of leaking it.
            self._started = self._process.pid is not None
            raise
        self._started = True
        # The spawned child has duplicated these endpoints.  Keeping the
        # child-only copies open in the parent prevents EOF detection and
        # leaks Windows kernel handles across repeated connect/disconnects.
        self._stop_pipe_recv.close()
        self._control_pipe_remote.close()

    def request_stop(self) -> None:
        if self._stopped:
            return
        if not self._started:
            self._stopped = True
            return
        try:
            self._control_pipe_local.send(
                RecordingCommand(
                    RecordingCommandKind.SHUTDOWN,
                    self._active_trial_uuid,
                )
            )
        except (BrokenPipeError, OSError):
            pass
        self._stopped = True

    def begin_recording(self, trial_uuid: str) -> None:
        """Begin one UUID-isolated raw stream without reconnecting hardware."""
        if not self._started:
            raise RuntimeError("preview worker is not started")
        if self._closed or self._stopped or not self.is_alive:
            raise RuntimeError("preview worker is not available")
        if self._cached_descriptor is None:
            raise RuntimeError("preview worker is not READY")
        normalized = normalize_trial_uuid(trial_uuid)
        if self._active_trial_uuid is not None:
            raise RuntimeError(
                f"recording already active for {self._active_trial_uuid}"
            )
        try:
            self._control_pipe_local.send(
                RecordingCommand(RecordingCommandKind.START_RECORDING, normalized)
            )
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("failed to send START_RECORDING") from exc
        self._recording_active = True
        self._active_trial_uuid = normalized

    def end_recording(self, trial_uuid: str) -> None:
        """End the active raw stream and enqueue its ordered END boundary."""
        if not self._started:
            raise RuntimeError("preview worker is not started")
        normalized = normalize_trial_uuid(trial_uuid)
        if self._active_trial_uuid != normalized:
            raise RuntimeError(
                f"cannot stop trial {normalized}; active trial is "
                f"{self._active_trial_uuid or 'none'}"
            )
        try:
            self._control_pipe_local.send(
                RecordingCommand(RecordingCommandKind.STOP_RECORDING, normalized)
            )
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("failed to send STOP_RECORDING") from exc
        # Pipe ordering guarantees a later START cannot overtake this STOP.
        # Clear the parent-side gate immediately so callers need not drain ACKs
        # before starting the next Trial.
        self._recording_active = False
        self._active_trial_uuid = None

    def discard_recording_backlog(self) -> int:
        """Non-blockingly remove messages left by a completed/failed Trial.

        The method is deliberately forbidden during recording: dropping even
        one active raw event would violate the loss-intolerant recording
        contract.  It is intended as a defensive cleanup immediately before a
        new Trial is armed, after the previous consumer has exited.
        """
        if self._recording_active or self._active_trial_uuid is not None:
            raise RuntimeError("cannot discard backlog while recording is active")
        discarded = 0
        while True:
            try:
                self._raw_recording_queue.get_nowait()
            except Empty:
                break
            discarded += 1
        return discarded

    def drain_control_ack(self) -> list[dict[str, Any]]:
        """Non-blocking drain of control pipe ACK messages from the worker."""
        acks: list[dict[str, Any]] = []
        while self._control_pipe_local.poll():
            try:
                msg = self._control_pipe_local.recv()
            except (EOFError, OSError):
                break
            if isinstance(msg, dict):
                acks.append(msg)
                status = str(msg.get("status") or "")
                trial_uuid = msg.get("trial_uuid")
                if status in ("STOPPED", "FAULT", "SHUTDOWN"):
                    if trial_uuid is None or trial_uuid == self._active_trial_uuid:
                        self._recording_active = False
                        self._active_trial_uuid = None
        return acks

    @property
    def recording_endpoint(self) -> RecordingStreamEndpoint | None:
        """Return the loss-intolerant raw queue after the worker is READY."""
        if self._cached_descriptor is None:
            return None
        return RecordingStreamEndpoint(
            queue=self._raw_recording_queue,
            device_id=str(
                self._cached_descriptor.get("device_id") or self._device_id
            ),
            modality=self._modality,
            descriptor=dict(self._cached_descriptor),
            configuration_snapshot=dict(self._cached_config_snapshot or {}),
        )

    # Compatibility aliases for code migrated incrementally to the public API.
    request_start_recording = begin_recording
    request_stop_recording = end_recording

    @property
    def recording_active(self) -> bool:
        return self._recording_active

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
            if event.event_type is WorkerEventType.STATE:
                state = str(event.payload.get("state") or "")
                if state == "READY":
                    desc = event.payload.get("descriptor")
                    if isinstance(desc, dict):
                        self._cached_descriptor = dict(desc)
                        snapshot = event.payload.get("configuration_snapshot")
                        self._cached_config_snapshot = (
                            dict(snapshot) if isinstance(snapshot, dict) else {}
                        )
        return events

    def join(self, timeout: float | None = None) -> int | None:
        if not self._started or self._process_closed:
            return None
        self._process.join(timeout=timeout)
        return self._process.exitcode

    def terminate(self, timeout: float = 5.0) -> int | None:
        if not self._started or self._process_closed:
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
        if self._started and not self._process_closed:
            self.request_stop()
            self.join(timeout=0.25)
            if self._process.is_alive():
                raise RuntimeError("cannot close a running preview worker")
            self._final_exitcode = self._process.exitcode
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
        try:
            self._control_pipe_local.close()
        except Exception:
            pass
        try:
            self._control_pipe_remote.close()
        except Exception:
            pass
        try:
            self._raw_recording_queue.close()
            self._raw_recording_queue.join_thread()
        except Exception:
            pass
        if not self._process_closed:
            self._process.close()
            self._process_closed = True
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
        recording_queue_size: int = DEFAULT_RECORDING_QUEUE_SIZE,
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
        self._finalized = False
        self._raw_recording_queue: Queue[Any] = Queue(maxsize=recording_queue_size)
        self._recording_producer: RecordingStreamProducer | None = None
        self._descriptor_payload: dict[str, Any] | None = None
        self._configuration_snapshot: dict[str, Any] = {}

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
            descriptor = self._adapter.descriptor()
            self._descriptor_payload = _descriptor_dict(descriptor)
            self._configuration_snapshot = dict(
                self._adapter.configuration_snapshot()
            )
            self._recording_producer = RecordingStreamProducer(
                self._raw_recording_queue,
                device_id=descriptor.device_id,
                modality=self._modality,
                descriptor=self._descriptor_payload,
                configuration_snapshot=self._configuration_snapshot,
            )
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
        if self._recording_producer is not None:
            self._recording_producer.abort(
                "preview worker shutdown during recording"
            )
        self._stopped = True

    def begin_recording(self, trial_uuid: str) -> None:
        if not self._alive or self._stopped:
            raise RuntimeError("preview worker is not available")
        if not self._ready or self._recording_producer is None:
            raise RuntimeError("preview worker is not READY")
        self._recording_producer.begin(trial_uuid)

    def end_recording(self, trial_uuid: str) -> None:
        if self._recording_producer is None:
            raise RuntimeError("recording stream unavailable")
        self._recording_producer.end(trial_uuid)

    def discard_recording_backlog(self) -> int:
        if self.recording_active:
            raise RuntimeError("cannot discard backlog while recording is active")
        discarded = 0
        while True:
            try:
                self._raw_recording_queue.get_nowait()
            except Empty:
                break
            discarded += 1
        return discarded

    @property
    def recording_endpoint(self) -> RecordingStreamEndpoint | None:
        if not self._ready or self._descriptor_payload is None:
            return None
        return RecordingStreamEndpoint(
            queue=self._raw_recording_queue,
            device_id=str(
                self._descriptor_payload.get("device_id") or self._device_id
            ),
            modality=self._modality,
            descriptor=dict(self._descriptor_payload),
            configuration_snapshot=dict(self._configuration_snapshot),
        )

    request_start_recording = begin_recording
    request_stop_recording = end_recording

    @property
    def recording_active(self) -> bool:
        return bool(
            self._recording_producer is not None
            and self._recording_producer.recording
        )

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        if self._adapter is None:
            return []
        if not self._stopped and limit > 0:
            raw = self._adapter.get_event(timeout=0.01)
            if raw is not None:
                descriptor = self._adapter.descriptor()
                if (
                    self._recording_producer is not None
                    and self._recording_producer.recording
                ):
                    try:
                        self._recording_producer.forward(raw)
                    except RecordingStreamOverflow as exc:
                        failed_trial_uuid = (
                            self._recording_producer.active_trial_uuid
                        )
                        self._recording_producer.abort(str(exc))
                        self._events.append(
                            WorkerEvent(
                                event_type=WorkerEventType.FAILED,
                                modality=self._modality,
                                trial_uuid=failed_trial_uuid,
                                payload={
                                    "state": "FAULT",
                                    "modality": self._modality,
                                    "device_id": self._device_id,
                                    "simulated": self._simulated,
                                    "fault": str(exc),
                                    "trial_uuid": failed_trial_uuid,
                                },
                                message=str(exc),
                            )
                        )
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
                                    "descriptor": dict(
                                        self._descriptor_payload or {}
                                    ),
                                    "configuration_snapshot": dict(
                                        self._configuration_snapshot
                                    ),
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
        if self._adapter is not None and not self._finalized:
            try:
                self._adapter.stop()
            except Exception:
                pass
            try:
                self._adapter.close()
            except Exception:
                pass
            self._finalized = True
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
    "DEFAULT_RECORDING_QUEUE_SIZE",
]
