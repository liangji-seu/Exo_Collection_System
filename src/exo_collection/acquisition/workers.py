"""Spawn-safe collector worker boundary used by the desktop UI."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.queues import Queue
from queue import Empty, Full
import traceback
from typing import Any

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.orchestration.models import TrialRunRequest


def _put_worker_event(queue: Queue[Any], event: WorkerEvent) -> None:
    payload = event.model_dump(mode="json")
    try:
        queue.put(payload, timeout=0.25)
    except Full:
        if event.event_type is WorkerEventType.PREVIEW:
            return  # preview is explicitly lossy
        queue.put(payload, timeout=5.0)


def _trial_worker_entry(
    request_payload: dict[str, Any],
    event_queue: Queue[Any],
    stop_event: Any,
) -> None:
    """Import the orchestration implementation inside the spawned process."""

    request = TrialRunRequest.model_validate(request_payload)

    def publish(event: WorkerEvent) -> None:
        _put_worker_event(event_queue, event)

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
        self._stop_requested = self._context.Event()
        self._process = self._context.Process(
            target=_trial_worker_entry,
            args=(request.model_dump(mode="json"), self._events, self._stop_requested),
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
        for _ in range(limit):
            try:
                payload = self._events.get_nowait()
            except Empty:
                break
            events.append(WorkerEvent.model_validate(payload))
        return events

    def join(self, timeout: float | None = None) -> int | None:
        self._process.join(timeout)
        return self._process.exitcode

    def close(self) -> None:
        if self._process.is_alive():
            raise RuntimeError("cannot close a running Collector worker; request a controlled stop first")
        self._events.close()
        self._events.join_thread()
        self._process.close()

