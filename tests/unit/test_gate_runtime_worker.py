import sys
import time
import types

from exo_collection.apps.model_runtime.gate_worker import GateWorker


def test_worker_consumes_all_grid_samples_and_publishes_every_fifth(monkeypatch):
    class Decision:
        def __init__(self, source, seq):
            self.source_grid_ns = source
            self.sequence = seq
            self.ready = self.allow = True
            self.mode = "ao"
            self.score = .1
            self.threshold = .5
            self.reason = "below_threshold"

    class Engine:
        def __init__(self):
            self.seen = []

        def reset(self):
            self.seen.clear()

        def step(self, source, values, valid=True):
            self.seen.append((source, tuple(values), valid))
            if len(self.seen) % 5:
                return None
            return Decision(source, len(self.seen)//5)

    package_name = "exo_collection.apps.model_runtime.gates"
    package = types.ModuleType(package_name)
    engine = Engine()
    package.load_gate = lambda mode, deployment_root=None: engine
    monkeypatch.setitem(sys.modules, package_name, package)

    worker = GateWorker("ao", queue_size=16, process_isolated=False)
    worker.start()
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not worker.ready:
            time.sleep(.005)
        assert worker.ready
        for i in range(10):
            assert worker.submit(i * 10_000_000, [i] * 12, generation=4)
        deadline = time.monotonic() + 1
        results = []
        while time.monotonic() < deadline and len(results) < 2:
            results.extend(worker.drain())
            if len(results) < 2:
                time.sleep(.005)
        assert len(engine.seen) == 10
        assert [r.decision.source_grid_ns for r in results] == [40_000_000, 90_000_000]
        assert all(r.generation == 4 for r in results)
    finally:
        worker.close()


def test_queue_overflow_latches_closed_without_blocking_producer():
    worker = GateWorker("ao", queue_size=1, process_isolated=False)
    worker._ready = True  # Exercise bounded enqueue behavior after asset loading.
    assert worker.submit(0, range(12), 1)
    assert not worker.submit(10_000_000, range(12), 1)
    assert worker.closed_reason == "gate input queue overflow"
    assert worker.dropped == 1
    worker.close()


def test_reset_during_cold_engine_load_keeps_worker_and_uses_new_generation(monkeypatch):
    entered, release = __import__("threading").Event(), __import__("threading").Event()

    class Decision:
        def __init__(self, source):
            self.source_grid_ns = source
            self.sequence = 1
            self.ready = self.allow = True
            self.mode = "ao"
            self.score = .1
            self.threshold = .5
            self.reason = "below_threshold"

    class Engine:
        provenance = {"mode": "ao", "asset_hash": "frozen"}

        def reset(self):
            pass

        def step(self, source, values, valid=True):
            return Decision(source)

    package_name = "exo_collection.apps.model_runtime.gates"
    package = types.ModuleType(package_name)

    def slow_load(mode, deployment_root=None):
        entered.set()
        release.wait(1)
        return Engine()

    package.load_gate = slow_load
    monkeypatch.setitem(sys.modules, package_name, package)
    worker = GateWorker("ao", process_isolated=False)
    worker.start()
    try:
        assert entered.wait(1)
        assert worker.state == "loading" and not worker.ready
        assert worker.submit(0, [0.0] * 12, generation=1)
        assert worker.queue_depth == 0 and worker.skipped_loading == 1
        worker.reset(generation=2)
        assert worker._thread.is_alive()
        release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not worker.ready:
            time.sleep(.005)
        assert worker.ready
        assert worker.submit(10, [0.0] * 12, generation=2)
        deadline = time.monotonic() + 1
        results = []
        while time.monotonic() < deadline and not results:
            results.extend(worker.drain())
            if not results:
                time.sleep(.005)
        assert len(results) == 1
        assert results[0].generation == 2
        assert worker.provenance["asset_hash"] == "frozen"
    finally:
        release.set()
        worker.close()


def test_unexpected_gate_process_exit_is_latched_as_fault():
    class DeadProcess:
        exitcode = 17

        @staticmethod
        def is_alive():
            return False

    worker = GateWorker("ao")
    worker._process = DeadProcess()
    try:
        assert worker.closed_reason == "gate process exited with code 17"
        assert worker.state == "fault"
    finally:
        worker._stop.set()
        worker._close_queue(worker.queue)
        worker._close_queue(worker._results)
        worker._close_queue(worker._status_queue)
