"""Bounded, independent worker for 20 Hz assistance-gate decisions."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from time import perf_counter_ns
from typing import Sequence


@dataclass(frozen=True)
class GateSample:
    source_grid_ns: int
    values: tuple[float, ...]
    generation: int
    valid: bool = True


@dataclass(frozen=True)
class GateResult:
    decision: object
    computed_at_ns: int
    generation: int
    compute_ms: float


def _gate_process_main(mode, deployment_root, input_queue, result_queue,
                       status_queue, stop_event):
    """Spawn target: load and execute the gate without sharing the UI GIL."""
    engine = None
    generation = None
    try:
        if mode == "phase_ensemble":
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        try:
            from threadpoolctl import threadpool_limits
            native_thread_limit = threadpool_limits(limits=1)
        except ImportError:
            native_thread_limit = None
        from exo_collection.apps.model_runtime.gates import load_gate
        engine = load_gate(mode, deployment_root=deployment_root)
        provenance = dict(getattr(engine, "provenance", {"mode": mode}))
        engine.reset()
        status_queue.put_nowait(("ready", provenance, None))
        while not stop_event.is_set():
            try:
                sample = input_queue.get(timeout=.02)
            except Empty:
                continue
            if generation != sample.generation:
                engine.reset()
                generation = sample.generation
            started = int(perf_counter_ns())
            decision = engine.step(sample.source_grid_ns, sample.values, valid=sample.valid)
            finished = int(perf_counter_ns())
            if decision is not None:
                result = GateResult(decision, finished, sample.generation,
                                    max(0.0, (finished-started)/1e6))
                try:
                    result_queue.put_nowait(result)
                except Full:
                    status_queue.put_nowait(("fault", provenance, "gate decision queue overflow"))
                    stop_event.set()
                    return
    except Exception as exc:
        try:
            status_queue.put_nowait(("fault", getattr(engine, "provenance", {"mode": mode}),
                                     f"gate worker failure: {exc}"))
        except Exception:
            pass


class GateWorker:
    """Consumes every canonical grid sample outside the inference thread.

    Production uses a spawned process so sklearn/Torch gate execution cannot
    contend with the 100 Hz model thread for the GIL. Thread mode remains
    available for deterministic unit tests. Queue overflow and computation
    failures latch closed; the producer never waits for gate inference.
    """
    def __init__(self, mode: str, *, deployment_root=None, queue_size: int = 256,
                 clock_ns=None, process_isolated: bool = True) -> None:
        self.mode = mode
        self.deployment_root = deployment_root
        self.queue_size = max(1, int(queue_size))
        self.process_isolated = bool(process_isolated)
        self._ctx = get_context("spawn") if self.process_isolated else None
        self.queue = (self._ctx.Queue(maxsize=self.queue_size) if self._ctx
                      else Queue(maxsize=self.queue_size))
        self.clock_ns = clock_ns or perf_counter_ns
        self._stop = self._ctx.Event() if self._ctx else Event()
        self._lock = RLock()
        self._thread: Thread | None = None
        self._process = None
        self._status_queue = self._ctx.Queue(maxsize=8) if self._ctx else None
        self._engine = None
        self._engine_generation: int | None = None
        self._results = self._ctx.Queue(maxsize=64) if self._ctx else Queue(maxsize=64)
        self._closed_reason: str | None = None
        self._dropped = 0
        self._skipped_loading = 0
        self._ready = False
        self.provenance: dict = {"mode": mode}

    @property
    def closed_reason(self) -> str | None:
        self._refresh_process_state()
        with self._lock:
            return self._closed_reason

    @property
    def queue_depth(self) -> int:
        try:
            return self.queue.qsize()
        except (NotImplementedError, OSError):
            return 0

    @property
    def ready(self) -> bool:
        self._refresh_process_state()
        with self._lock:
            return self._ready and self._closed_reason is None

    @property
    def state(self) -> str:
        self._refresh_process_state()
        with self._lock:
            if self._closed_reason is not None:
                return "fault"
            if self.mode == "none":
                return "disabled"
            if self._ready:
                return "ready"
            if ((self._thread is not None and self._thread.is_alive())
                    or (self._process is not None and self._process.is_alive())):
                return "loading"
            return "stopped"

    @property
    def skipped_loading(self) -> int:
        with self._lock:
            return self._skipped_loading

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def start(self) -> None:
        if self.mode == "none":
            return
        if self.process_isolated:
            self._start_process()
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._closed_reason = None
            self._ready = False
        self._engine = None
        self._thread = Thread(target=self._run, name="assistance-gate", daemon=True)
        self._thread.start()

    def _start_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        with self._lock:
            self._closed_reason = None
            self._ready = False
        self._stop = self._ctx.Event()
        self.queue = self._ctx.Queue(maxsize=self.queue_size)
        self._results = self._ctx.Queue(maxsize=64)
        self._status_queue = self._ctx.Queue(maxsize=8)
        self._process = self._ctx.Process(target=_gate_process_main,
            args=(self.mode, self.deployment_root, self.queue, self._results,
                  self._status_queue, self._stop),
            name="assistance-gate", daemon=True)
        self._process.start()

    def _refresh_process_state(self) -> None:
        if not self.process_isolated or self._status_queue is None:
            return
        while True:
            try:
                kind, provenance, reason = self._status_queue.get_nowait()
            except (Empty, OSError, ValueError):
                break
            with self._lock:
                self.provenance = dict(provenance or {"mode": self.mode})
                if kind == "ready":
                    self._ready = True
                elif kind == "fault":
                    self._ready = False
                    self._closed_reason = str(reason or "gate process failed")
        process = self._process
        if (process is not None and not process.is_alive() and not self._stop.is_set()):
            with self._lock:
                if self._closed_reason is None:
                    self._ready = False
                    self._closed_reason = f"gate process exited with code {process.exitcode}"

    def reset(self, generation: int | None = None) -> None:
        """Start a fresh logical epoch without tearing down a live engine.

        The worker may still be loading deployment assets when the user arms.
        Queue order plus per-sample generations makes reset safe even when an
        old sample is already in flight: its result keeps the old generation,
        and the engine resets before consuming a new-generation sample.
        """
        while True:
            try:
                self.queue.get_nowait()
                if not self.process_isolated:
                    self.queue.task_done()
            except (Empty, OSError, ValueError):
                break
        while True:
            try:
                self._results.get_nowait()
            except (Empty, OSError, ValueError):
                break
        self._refresh_process_state()
        with self._lock:
            self._closed_reason = None
            self._skipped_loading = 0
        if ((self._thread and self._thread.is_alive())
                or (self._process is not None and self._process.is_alive())):
            return
        self.start()

    def submit(self, source_grid_ns: int, values: Sequence[float], generation: int,
               valid: bool = True) -> bool:
        if self.mode == "none":
            return True
        self._refresh_process_state()
        with self._lock:
            if self._closed_reason is not None:
                return False
            if not self._ready:
                self._skipped_loading += 1
                return True
        sample = GateSample(int(source_grid_ns), tuple(float(v) for v in values),
                            int(generation), bool(valid))
        try:
            self.queue.put_nowait(sample)
            return True
        except Full:
            with self._lock:
                self._dropped += 1
                self._closed_reason = "gate input queue overflow"
            return False

    def drain(self) -> list[GateResult]:
        self._refresh_process_state()
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except (Empty, OSError, ValueError):
                return results

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._process is not None:
            process = self._process
            if process.is_alive():
                process.join(max(0.0, timeout))
            if process.is_alive():
                process.terminate()
                process.join(1.0)
            with self._lock:
                self._ready = False
            self._close_queue(self.queue)
            self._close_queue(self._results)
            self._close_queue(self._status_queue)
            return
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout)

    @staticmethod
    def _close_queue(q) -> None:
        if q is None:
            return
        try:
            q.cancel_join_thread()
        except (AttributeError, OSError):
            pass
        try:
            q.close()
        except (AttributeError, OSError, ValueError):
            pass

    def _publish(self, result: GateResult) -> None:
        try:
            self._results.put_nowait(result)
        except Full:
            # Dropping a completed decision is fail-closed at dispatch via TTL.
            with self._lock:
                self._dropped += 1
                self._closed_reason = "gate decision queue overflow"

    def _run(self) -> None:
        if self.mode == "none":
            return
        try:
            from exo_collection.apps.model_runtime.gates import load_gate
            self._engine = load_gate(self.mode, deployment_root=self.deployment_root)
            self.provenance = dict(getattr(self._engine, "provenance", {"mode": self.mode}))
            self._engine.reset()
            self._engine_generation = None
            with self._lock:
                self._ready = True
            while not self._stop.is_set():
                try:
                    sample = self.queue.get(timeout=.02)
                except Empty:
                    continue
                try:
                    if self._engine_generation != sample.generation:
                        self._engine.reset()
                        self._engine_generation = sample.generation
                    started = int(self.clock_ns())
                    decision = self._engine.step(sample.source_grid_ns, sample.values,
                                                 valid=sample.valid)
                    finished = int(self.clock_ns())
                    if decision is not None:
                        self._publish(GateResult(decision, finished, sample.generation,
                                                 max(0.0, (finished-started)/1e6)))
                finally:
                    self.queue.task_done()
        except Exception as exc:
            with self._lock:
                self._ready = False
                self._closed_reason = f"gate worker failure: {exc}"
        finally:
            with self._lock:
                self._ready = False
