"""One bounded daemon loader; Qt objects are prepared only by its GUI receiver."""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .backends import ModelRuntime, build_model_runtime
from .spec import ModelSpec, load_model_spec


@dataclass
class LoadResult:
    generation: int
    path: Path
    spec: ModelSpec | None = None
    runtime: ModelRuntime | None = None
    sha256: str | None = None
    error: Exception | None = None


class ModelLoader(QObject):
    completed = Signal(object)

    def __init__(self, diagnostics):
        super().__init__()
        self._diagnostics = diagnostics
        self._condition = threading.Condition()
        self._generation = 0
        self._pending = None
        self._closed = False
        # No QThread teardown wait on the GUI or unsafe terminate of torch.
        self._thread = threading.Thread(target=self._work, name="model-loader", daemon=True)
        self._thread.start()

    def submit(self, path: Path) -> int:
        with self._condition:
            self._generation += 1
            self._pending = self._generation, Path(path)
            self._condition.notify()
            return self._generation

    def cancel(self):
        with self._condition:
            self._generation += 1
            self._pending = None

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._pending = None
            self._condition.notify()

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                generation, path = self._pending
                self._pending = None
            result = LoadResult(generation, path)
            started = time.perf_counter()
            stage = "spec"
            try:
                result.spec = load_model_spec(path)
                self._diagnostics.record("model_load_stage", generation=generation, stage=stage,
                                         elapsed_s=time.perf_counter() - started, path=str(path))
                if result.spec.model.backend in {"torch", "torchscript"}:
                    stage = "torch_import"
                    stage_started = time.perf_counter()
                    __import__("torch")
                    self._diagnostics.record("model_load_stage", generation=generation, stage=stage,
                        duration_s=time.perf_counter() - stage_started, elapsed_s=time.perf_counter() - started)
                stage = "runtime"
                stage_started = time.perf_counter()
                result.runtime = build_model_runtime(result.spec, base_dir=path.parent)
                self._diagnostics.record("model_load_stage", generation=generation, stage=stage,
                    duration_s=time.perf_counter() - stage_started, elapsed_s=time.perf_counter() - started)
                stage = "weights_hash"
                weights = result.spec.model.weights_path
                if weights:
                    weights_path = Path(weights)
                    if not weights_path.is_absolute():
                        weights_path = path.parent / weights_path
                    digest = hashlib.sha256()
                    with weights_path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    result.sha256 = digest.hexdigest()
                self._diagnostics.record("model_load_stage", generation=generation, stage=stage,
                                         elapsed_s=time.perf_counter() - started)
            except Exception as exc:
                result.error = exc
                self._diagnostics.record("model_load_failed", generation=generation, stage=stage,
                                         elapsed_s=time.perf_counter() - started, exception=exc)
            with self._condition:
                stale = self._closed or generation != self._generation
            if stale:
                self._diagnostics.record("model_load_discarded", generation=generation)
                if result.runtime is not None:
                    try:
                        result.runtime.unload()
                    except Exception as exc:
                        self._diagnostics.record("model_unload_failed", generation=generation, exception=exc)
                continue
            self.completed.emit(result)
