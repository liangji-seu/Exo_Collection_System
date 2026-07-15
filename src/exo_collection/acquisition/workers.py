"""Spawn-safe collector worker boundary used by the desktop UI."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.queues import Queue
from queue import Empty, Full
import traceback
from typing import Any

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.orchestration.models import TrialRunRequest


LOSSY_EVENT_TYPES = {
    WorkerEventType.PREVIEW,
    WorkerEventType.HEALTH,
    WorkerEventType.METRIC,
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
) -> None:
    """Import the orchestration implementation inside the spawned process."""

    # Telemetry is explicitly lossy.  If the UI stops consuming it, a Windows
    # multiprocessing.Queue feeder can otherwise keep this process alive after
    # the Trial has already finalized.  Only the small control queue is allowed
    # to participate in the child process's exit flush.
    telemetry_queue.cancel_join_thread()
    request = TrialRunRequest.model_validate(request_payload)

    def publish(event: WorkerEvent) -> None:
        _put_worker_event(telemetry_queue, control_queue, event)

    try:
        from exo_collection.orchestration.simulated import run_simulated_trial

        result = run_simulated_trial(request, stop_requested=stop_event, publish=publish)
        publish(
            WorkerEvent(
                event_type=WorkerEventType.COMPLETED,
                trial_uuid=str(result.trial_uuid),
                message="Trial package finalized",
                payload=result.model_dump(mode="json"),
            )
        )
    except BaseException as exc:
        publish(
            WorkerEvent(
                event_type=WorkerEventType.FAILED,
                trial_uuid=str(request.trial_uuid),
                message=f"{type(exc).__name__}: {exc}",
                payload={"traceback": traceback.format_exc(limit=20)},
            )
        )
        raise


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
        self._process = self._context.Process(
            target=_trial_worker_entry,
            args=(
                request.model_dump(mode="json"),
                self._events,
                self._control_events,
                self._stop_requested,
            ),
            name=f"collector-core-{str(request.trial_uuid)[:8]}",
            daemon=False,
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def start(self) -> None:
        if self._process.pid is not None:
            raise RuntimeError("Collector worker can only be started once")
        self._process.start()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        if limit <= 0:
            return []
        events: list[WorkerEvent] = []
        for queue in (self._control_events, self._events):
            while len(events) < limit:
                try:
                    payload = queue.get_nowait()
                except Empty:
                    break
                events.append(WorkerEvent.model_validate(payload))
        return events

    def join(self, timeout: float | None = None) -> int | None:
        self._process.join(timeout)
        return self._process.exitcode

    def terminate_for_recovery(self, timeout: float = 5.0) -> int | None:
        """Last-resort shutdown after a controlled stop timeout.

        A forced process exit intentionally leaves the Trial as `.recording`;
        the normal startup recovery workflow will inspect it before publication.
        """

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
        return self._process.exitcode

    def close(self) -> None:
        if self._process.is_alive():
            raise RuntimeError("cannot close a running Collector worker; request a controlled stop first")
        self._events.close()
        self._events.join_thread()
        self._control_events.close()
        self._control_events.join_thread()
        self._process.close()
