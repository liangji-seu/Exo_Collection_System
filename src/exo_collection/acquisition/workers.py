"""Spawn-safe collector worker boundary used by the desktop UI."""

from __future__ import annotations

from contextlib import suppress
import multiprocessing as mp
from multiprocessing.queues import Queue
from queue import Empty, Full
import traceback
from typing import Any

import numpy as np

from exo_collection.acquisition.buffers import (
    PreviewBufferDescriptor,
    SharedPreviewBuffer,
)
from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.orchestration.models import TrialRunRequest


LOSSY_EVENT_TYPES = {
    WorkerEventType.PREVIEW,
    WorkerEventType.HEALTH,
    WorkerEventType.METRIC,
    # The raw pulse waveform and every detected edge are persisted inside the
    # Trial. UI sync notifications are therefore replaceable telemetry; making
    # them reliable would let a frozen UI fill the control queue and block the
    # acquisition process during a long Trial.
    WorkerEventType.SYNC,
}


def _put_worker_event(
    telemetry_queue: Queue[Any],
    control_queue: Queue[Any],
    event: WorkerEvent,
) -> None:
    payload = event.model_dump(mode="json")
    if event.event_type in LOSSY_EVENT_TYPES:
        try:
            telemetry_queue.put_nowait(payload)
        except Full:
            pass  # every telemetry/preview event is replaceable and lossy
        return
    # State and terminal events have their own small queue. Their bounded count
    # cannot be exhausted by a stalled preview consumer.
    control_queue.put(payload, timeout=2.0)


def _trial_worker_entry(
    request_payload: dict[str, Any],
    telemetry_queue: Queue[Any],
    control_queue: Queue[Any],
    stop_event: Any,
    preview_descriptors: dict[str, PreviewBufferDescriptor],
) -> None:
    """Import the orchestration implementation inside the spawned process."""

    # Telemetry is explicitly lossy.  If the UI stops consuming it, a Windows
    # multiprocessing.Queue feeder can otherwise keep this process alive after
    # the Trial has already finalized.  Only the small control queue is allowed
    # to participate in the child process's exit flush.
    telemetry_queue.cancel_join_thread()
    request = TrialRunRequest.model_validate(request_payload)
    preview_buffers = {
        modality: SharedPreviewBuffer.attach(descriptor)
        for modality, descriptor in preview_descriptors.items()
    }

    def publish(event: WorkerEvent) -> None:
        if event.event_type is WorkerEventType.PREVIEW and event.modality in preview_buffers:
            payload = dict(event.payload)
            channels = payload.pop("channels", None)
            values = payload.pop("values", None)
            x_values = np.asarray(payload.pop("x", ()), dtype=np.float32).reshape(-1)
            if isinstance(channels, (list, tuple)) and channels:
                arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in channels]
                points_per_channel = min(array.size for array in arrays)
                arrays = [array[:points_per_channel] for array in arrays]
                signal_values = np.concatenate(arrays)
                channel_count = len(arrays)
            else:
                signal_values = np.asarray(values or (), dtype=np.float32).reshape(-1)
                channel_count = 1
                points_per_channel = int(signal_values.size)

            # Keep paired time/value arrays aligned if an adapter provides a
            # larger preview than the shared segment. Raw acquisition data is
            # unaffected; this branch only thins the replaceable UI view.
            capacity = preview_buffers[event.modality].capacity
            if x_values.size and x_values.size == signal_values.size:
                maximum_points = max(1, capacity // 2)
                if signal_values.size > maximum_points:
                    selection = np.linspace(
                        0,
                        signal_values.size - 1,
                        maximum_points,
                        dtype=np.int64,
                    )
                    x_values = x_values[selection]
                    signal_values = signal_values[selection]
                    points_per_channel = int(signal_values.size)
            elif x_values.size:
                # A malformed preview time axis must never be paired with a
                # different signal. Drop only the optional x values; the UI
                # will use a local sample index for this lossy preview.
                x_values = np.empty(0, dtype=np.float32)

            shared_values = np.concatenate((x_values, signal_values))
            if shared_values.size:
                generation = preview_buffers[event.modality].write(
                    shared_values,
                    host_monotonic_ns=int(
                        payload.get("host_monotonic_ns") or 0
                    ),
                )
                payload["shared_preview"] = {
                    "generation": generation,
                    "length": int(shared_values.size),
                    "x_length": int(x_values.size),
                    "channel_count": channel_count,
                    "points_per_channel": points_per_channel,
                }
                event = event.model_copy(update={"payload": payload})
        _put_worker_event(telemetry_queue, control_queue, event)

    try:
        from exo_collection.orchestration.simulated import run_trial

        result = run_trial(request, stop_requested=stop_event, publish=publish)
        publish(
            WorkerEvent(
                event_type=WorkerEventType.COMPLETED,
                trial_uuid=str(result.trial_uuid),
                message="Trial package finalized",
                payload=result.model_dump(mode="json"),
            )
        )
    except BaseException as exc:
        failure_payload: dict[str, Any] = {
            "traceback": traceback.format_exc(limit=20),
        }
        # Orchestration failures may expose a small JSON-safe audit context.
        # Keeping this generic avoids coupling the spawn boundary to one
        # concrete adapter or simulated implementation.
        structured_context = getattr(exc, "worker_payload", None)
        if callable(structured_context):
            try:
                context = structured_context()
                if isinstance(context, dict):
                    failure_payload.update(context)
            except BaseException:
                pass
        publish(
            WorkerEvent(
                event_type=WorkerEventType.FAILED,
                trial_uuid=str(request.trial_uuid),
                message=f"{type(exc).__name__}: {exc}",
                payload=failure_payload,
            )
        )
        raise
    finally:
        for buffer in preview_buffers.values():
            buffer.close()


class CollectorWorker:
    """Own one spawned collector-core process and its bounded control queue."""

    def __init__(self, request: TrialRunRequest, *, queue_capacity: int = 256) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.request = request
        self._context = mp.get_context("spawn")
        self._events = self._context.Queue(maxsize=queue_capacity)
        self._control_events = self._context.Queue(maxsize=32)
        self._stop_requested = self._context.Event()
        self._preview_buffers = {
            "ultrasound": SharedPreviewBuffer.create(16 * 512),
            "imu": SharedPreviewBuffer.create(4096),
            "encoder": SharedPreviewBuffer.create(4096),
            "sync_pulse": SharedPreviewBuffer.create(4096),
        }
        self._process = self._context.Process(
            target=_trial_worker_entry,
            args=(
                request.model_dump(mode="json"),
                self._events,
                self._control_events,
                self._stop_requested,
                {
                    modality: buffer.descriptor
                    for modality, buffer in self._preview_buffers.items()
                },
            ),
            name=f"collector-core-{str(request.trial_uuid)[:8]}",
            daemon=False,
        )
        self._closed = False
        self._exitcode_snapshot: int | None = None

    @property
    def pid(self) -> int | None:
        return None if self._closed else self._process.pid

    @property
    def is_alive(self) -> bool:
        return False if self._closed else self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._exitcode_snapshot if self._closed else self._process.exitcode

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Collector worker is closed")
        if self._process.pid is not None:
            raise RuntimeError("Collector worker can only be started once")
        try:
            self._process.start()
        except BaseException:
            # A Windows spawn failure can occur after queues/shared-memory
            # segments have been created. No window poller owns the handle in
            # that path, so release every resource before propagating it.
            self._cleanup_after_start_failure()
            raise

    def request_stop(self) -> None:
        if self._closed:
            return
        self._stop_requested.set()

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        if limit <= 0 or self._closed:
            return []
        events: list[WorkerEvent] = []
        for queue in (self._control_events, self._events):
            while len(events) < limit:
                try:
                    payload = queue.get_nowait()
                except Empty:
                    break
                event = WorkerEvent.model_validate(payload)
                events.append(self._inflate_shared_preview(event))
        return events

    def _inflate_shared_preview(self, event: WorkerEvent) -> WorkerEvent:
        if event.event_type is not WorkerEventType.PREVIEW or event.modality is None:
            return event
        buffer = self._preview_buffers.get(event.modality)
        marker = event.payload.get("shared_preview")
        if buffer is None or not isinstance(marker, dict):
            return event
        try:
            values, timestamp, observed_generation = buffer.read(retries=50)
            expected_generation = int(marker.get("generation") or 0)
            x_length = max(0, int(marker.get("x_length") or 0))
            channel_count = max(1, int(marker.get("channel_count") or 1))
            points_per_channel = max(
                0, int(marker.get("points_per_channel") or 0)
            )
        except (RuntimeError, TypeError, ValueError):
            return event
        # A later preview may overwrite shared memory before its Queue marker
        # is consumed. Never combine metadata from one generation with values
        # from another; the next replaceable preview event will carry the new
        # marker.
        if observed_generation != expected_generation:
            return event
        payload = dict(event.payload)
        payload["host_monotonic_ns"] = timestamp
        payload["shared_preview"] = {
            **marker,
            "observed_generation": observed_generation,
        }
        if x_length:
            if values.size < x_length:
                return event
            payload["x"] = values[:x_length].tolist()
            values = values[x_length:]
        if channel_count > 1 and points_per_channel > 0:
            required = channel_count * points_per_channel
            if values.size < required:
                return event
            channels = values[:required].reshape(channel_count, points_per_channel)
            payload["channels"] = channels.tolist()
            payload["values"] = channels[0].tolist()
        else:
            payload["values"] = values.tolist()
        return event.model_copy(update={"payload": payload})

    def join(self, timeout: float | None = None) -> int | None:
        if self._closed:
            return self._exitcode_snapshot
        self._process.join(timeout)
        self._exitcode_snapshot = self._process.exitcode
        return self._exitcode_snapshot

    def terminate_for_recovery(self, timeout: float = 5.0) -> int | None:
        """Last-resort shutdown after a controlled stop timeout.

        A forced process exit intentionally leaves the Trial as `.recording`;
        the normal startup recovery workflow will inspect it before publication.
        """

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._closed:
            return self._exitcode_snapshot
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
        self._exitcode_snapshot = self._process.exitcode
        return self._exitcode_snapshot

    def close(self) -> None:
        if self._closed:
            return
        if self._process.is_alive():
            raise RuntimeError("cannot close a running Collector worker; request a controlled stop first")
        self._exitcode_snapshot = self._process.exitcode
        self._events.close()
        self._events.join_thread()
        self._control_events.close()
        self._control_events.join_thread()
        self._process.close()
        for buffer in self._preview_buffers.values():
            try:
                buffer.close()
            finally:
                buffer.unlink()
        self._preview_buffers.clear()
        self._closed = True

    def _cleanup_after_start_failure(self) -> None:
        if self._process.pid is not None:
            with suppress(BaseException):
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=1.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)
        with suppress(BaseException):
            self._exitcode_snapshot = self._process.exitcode
        for queue in (self._events, self._control_events):
            with suppress(BaseException):
                queue.close()
            with suppress(BaseException):
                queue.join_thread()
        with suppress(BaseException):
            self._process.close()
        for buffer in self._preview_buffers.values():
            with suppress(BaseException):
                buffer.close()
            with suppress(BaseException):
                buffer.unlink()
        self._preview_buffers.clear()
        self._closed = True
