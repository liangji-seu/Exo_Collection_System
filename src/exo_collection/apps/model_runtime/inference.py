"""Real-time inference loop: subscriptions → features → model → torque.

``InferenceController`` is a ``QThread`` that owns the fast control loop.  It
drains the ``SubscriptionHub``, assembles a ``[C, T]`` feature window each
control tick, runs the loaded ``ModelRuntime``, maps the output onto bilateral
motor torque, and hands the command to a ``TorqueOutput``.

Any exception inside the loop is treated as a fault: torque is zeroed (when
``safety.zero_on_error``) and a ``fault`` signal is emitted.  Torque output is
gated by the ``TorqueOutput.enabled`` flag (``enable_required``), which the
Mock backend already respects — the loop may still predict and report so the
operator can watch model output before arming the motors.
"""

from __future__ import annotations

import threading
from time import perf_counter_ns

import numpy as np
from PySide6.QtCore import QThread, Signal

from exo_collection.apps.model_runtime.backends import ModelRuntime, ModelRuntimeError
from exo_collection.apps.model_runtime.features import (
    FeatureExtractionError,
    FeatureExtractor,
)
from exo_collection.apps.model_runtime.spec import ModelSpec, OutputSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import TorqueOutput


def map_output_to_torque(output: np.ndarray, output_spec: OutputSpec) -> tuple[float, float]:
    """Map a flat model output onto (left_nm, right_nm).

    - one scalar → the ``single_output_maps_to`` motor (default right hip), other 0;
    - two scalars → (left, right);
    - anything else → error.
    """
    if output_spec.kind != "torque_left_right":
        raise ModelRuntimeError(f"unsupported output kind: {output_spec.kind!r}")
    out = np.asarray(output, dtype=np.float32).reshape(-1)
    if out.size == 2:
        return float(out[0]), float(out[1])
    if out.size == 1:
        target = output_spec.single_output_maps_to or "right_torque"
        if target == "left_torque":
            return float(out[0]), 0.0
        if target == "right_torque":
            return 0.0, float(out[0])
        raise ModelRuntimeError(f"unknown single_output_maps_to: {target!r}")
    raise ModelRuntimeError(f"unsupported model output size: {out.size}")


class InferenceController(QThread):
    """Owns the prediction loop and emits results back to the GUI."""

    prediction = Signal(object)   # (host_monotonic_ns, left_nm, right_nm, raw_output)
    torque_commanded = Signal(object)  # TorqueCommand (from the output backend)
    fault = Signal(str)
    state_changed = Signal(str)

    def __init__(
        self,
        hub: SubscriptionHub,
        spec: ModelSpec,
        runtime: ModelRuntime,
        torque_output: TorqueOutput,
        extractor: FeatureExtractor | None = None,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._hub = hub
        self._spec = spec
        self._runtime = runtime
        self._torque = torque_output
        self._extractor = extractor or _default_extractor()
        self._stop_event = threading.Event()
        self._enabled = False
        self._last_predict_ns: int | None = None
        self._last_error: str | None = None

    # -- state -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_predict_ns(self) -> int | None:
        return self._last_predict_ns

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._torque.set_enabled(self._enabled)

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_and_wait(self, timeout_ms: int = 3000) -> bool:
        self.request_stop()
        return self.wait(timeout_ms)

    # -- loop --------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - explicit safety handling
        self.state_changed.emit("running")
        period_s = 1.0 / max(1.0, self._spec.features.control_rate_hz)
        try:
            while not self._stop_event.is_set():
                self._hub.drain()
                self._tick()
                self._stop_event.wait(period_s)
        except Exception as exc:  # any fault → zero torque + report
            self._last_error = str(exc)
            if self._spec.safety.zero_on_error:
                self._torque.zero()
            self.state_changed.emit("fault")
            self.fault.emit(str(exc))
            return
        self.state_changed.emit("stopped")

    def _tick(self) -> None:
        features = self._extractor.extract(self._hub, self._spec.features)
        if features.size == 0:
            return
        output = self._runtime.predict(features)
        left_nm, right_nm = map_output_to_torque(output, self._spec.output)
        now_ns = perf_counter_ns()
        self._last_predict_ns = now_ns
        self.prediction.emit((now_ns, left_nm, right_nm, output))
        self._torque.send(left_nm, right_nm)
        command = getattr(self._torque, "last", None)
        if command is not None:
            self.torque_commanded.emit(command)


def _default_extractor() -> FeatureExtractor:
    from exo_collection.apps.model_runtime.features import DefaultWindowFeatureExtractor

    return DefaultWindowFeatureExtractor()


__all__ = [
    "InferenceController",
    "map_output_to_torque",
]
