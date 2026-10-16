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

import logging

import threading
from time import perf_counter_ns, sleep

import numpy as np
from PySide6.QtCore import QThread, Signal

from exo_collection.apps.model_runtime.backends import ModelRuntime, ModelRuntimeError
from exo_collection.apps.model_runtime.features import (
    FeatureExtractionError,
    FeatureExtractor,
)
from exo_collection.apps.model_runtime.spec import ModelSpec, OutputSpec
from exo_collection.apps.model_runtime.model_adapter import ModelInputAdapter, ModelOutputAdapter, map_output_values
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import TorqueOutput


def map_output_to_torque(output: np.ndarray, output_spec: OutputSpec) -> tuple[float, float]:
    """Compatibility wrapper mapping (left, right) in the declared model unit.

    - one scalar → the ``single_output_maps_to`` motor (default right hip), other 0;
    - two scalars → (left, right);
    - anything else → error.
    """
    return map_output_values(output, output_spec)


class InferenceController(QThread):
    """Owns the prediction loop and emits results back to the GUI."""

    prediction = Signal(object)   # (host_monotonic_ns, left_nm, right_nm, raw_output)
    torque_commanded = Signal(object)  # TorqueCommand (from the output backend)
    fault = Signal(str)
    state_changed = Signal(str)
    pipeline_sample = Signal(object)  # time_ns, estimate_nm_per_kg, estimate_nm, desired_right_nm
    diagnostic_event = Signal(object)

    def __init__(
        self,
        hub: SubscriptionHub,
        spec: ModelSpec,
        runtime: ModelRuntime,
        torque_output: TorqueOutput,
        extractor: FeatureExtractor | None = None,
        *,
        clock_ns=None,
        parent=None,
        diagnostic_callback=None,
        recording_callback=None,
        allow_input_wait_recovery: bool = False,
    ) -> None:
        super().__init__(parent)
        self._hub = hub
        self._spec = spec
        self._runtime = runtime
        self._torque = torque_output
        self._clock_ns = clock_ns or perf_counter_ns
        self._extractor = extractor or _default_extractor(spec, clock_ns=self._clock_ns)
        self._stop_event = threading.Event()
        self._enabled = False
        self._last_predict_ns: int | None = None
        self._last_error: str | None = None
        self._state_lock = threading.RLock()
        self._generation = 0
        self._predicted_once = False
        self._recording_callback = recording_callback
        self._diagnostic_callback = diagnostic_callback
        self._timing = {}
        self._timing_max = {}
        self._last_fault_details = None
        self._tick_started_ns = None
        self._last_loop_ns = None
        self._allow_input_wait_recovery = bool(allow_input_wait_recovery)
        self._input_wait_reason: str | None = None
        self._input_wait_report_ns: int | None = None
        from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor

        self._imu_extractor = (
            self._extractor if isinstance(self._extractor, ImuTcnFeatureExtractor) else None
        )
        self._input_adapter = ModelInputAdapter(hub, spec, self._extractor, clock_ns=clock_ns)
        self._output_adapter = ModelOutputAdapter(spec)
        self._assistance = self._output_adapter

    # -- state -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        with self._state_lock:
            return self._enabled

    @property
    def ready(self) -> bool:
        return bool(self.preflight().get("ready", False))

    @property
    def last_predict_ns(self) -> int | None:
        return self._last_predict_ns

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def waiting_for_inputs(self) -> bool:
        return self._input_wait_reason is not None

    @property
    def input_wait_reason(self) -> str | None:
        return self._input_wait_reason

    def set_input_wait_recovery(self, allowed: bool) -> None:
        """Explicit display-only policy; output attachment revokes it."""
        with self._state_lock:
            self._allow_input_wait_recovery = bool(allowed)

    def set_enabled(self, enabled: bool) -> None:
        requested = bool(enabled)
        with self._state_lock:
            if requested and self._stop_event.is_set():
                raise ValueError("cannot enable a stopped inference controller")
            if requested and self._enabled:
                return
            if requested:
                self._drain_inputs()
                report = self._preflight_unlocked()
                assistance_report = self.assistance_preflight()
                if not report.get("ready", False) or not assistance_report.get("ready", False):
                    self._enabled = False
                    self._generation += 1
                    self._zero_and_disarm_unlocked()
                    message = "; ".join(list(report.get("errors", [])) + list(assistance_report.get("errors", []))) or "模型输入尚未就绪"
                    if not report.get("ready", False):
                        self._last_error = message
                    raise ValueError(message)
            self._generation += 1
            self._enabled = requested
            if requested:
                # Explicit arming is never a display-only route, including
                # tests which deliberately arm a Mock sink.
                self._allow_input_wait_recovery = False
            if not requested:
                self._zero_and_disarm_unlocked()
                self._assistance.reset()
                return
            else:
                # Do not release assistance accumulated before the explicit
                # arming edge (including a pending delay sample).
                self._assistance.reset()
            try:
                self._torque.set_enabled(self._enabled)
            except Exception:
                self._enabled = False
                self._assistance.reset()
                self._torque.set_enabled(False)
                raise

    def set_assistance(self, config) -> None:
        """Replace middle-layer parameters and force a clean rewarm."""
        from exo_collection.apps.model_runtime.spec import AssistanceSpec

        parsed = config if isinstance(config, AssistanceSpec) else AssistanceSpec.model_validate(config)
        with self._state_lock:
            self._spec.assistance = parsed
            self._assistance.configure(parsed)
            self._generation += 1
            self._enabled = False
            self._zero_and_disarm_unlocked()
            self._reset_processing_history_unlocked()

    def set_torque_output(self, output: TorqueOutput) -> None:
        """Replace the motor sink while keeping inference and processing alive.

        Output attachment is an application/window concern, so this method
        never closes either sink.  The generation edge makes a prediction
        already in ``predict`` ineligible for publication to the replacement
        output, and clearing assistance history prevents delayed samples
        from crossing the attachment boundary.  Input-grid progress is kept
        so attachment does not cause duplicate prediction of a consumed grid.
        Reattaching the current sink is an idempotent no-op.
        """
        if output is None:
            raise TypeError("output must implement TorqueOutput")
        with self._state_lock:
            old_output = self._torque
            if output is old_output:
                return
            self._generation += 1
            self._enabled = False
            self._allow_input_wait_recovery = False
            self._input_wait_reason = None
            self._assistance.reset()
            self._zero_and_disarm_unlocked()
            # A newly attached sink may be permissive by default (for
            # example, MockTorqueOutput(enable_required=False)); attachment
            # must still start disarmed until the window explicitly arms it.
            output.set_enabled(False)
            self._torque = output

    def preflight(self) -> dict[str, object]:
        # Serialize the non-consuming window probe with the tick extractor;
        # the probe temporarily rewinds extractor bookkeeping.
        with self._state_lock:
            return self._preflight_unlocked()

    def _preflight_unlocked(self) -> dict[str, object]:
        """Return a GUI-safe report; this method never arms the motors."""
        self._input_adapter.extractor = self._extractor
        return self._input_adapter.preflight()

    def assistance_preflight(self) -> dict[str, object]:
        """Middle-layer readiness, separate from observing model predictions."""
        with self._state_lock:
            return self._output_adapter.preflight()

    def request_stop(self) -> None:
        # Invalidate the in-flight tick while holding the same lock used by
        # its commit section.  ``predict`` runs outside this lock, so a stop
        # request can disarm immediately even when the model is blocked.
        self._stop_event.set()
        with self._state_lock:
            self._generation += 1
            self._enabled = False
            self._zero_and_disarm_unlocked()
            self._reset_processing_history_unlocked()

    def stop_and_wait(self, timeout_ms: int = 3000) -> bool:
        self.request_stop()
        if not self.isRunning():
            return True
        return self.wait(timeout_ms)

    # -- loop --------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - explicit safety handling
        self.state_changed.emit("running")
        period_ns = int(1e9 / self._spec.features.control_rate_hz)
        next_tick_ns = perf_counter_ns()
        last_report_ns = next_tick_ns
        ticks_since_report = predictions_since_report = 0
        try:
            while not self._stop_event.is_set():
                now_ns = perf_counter_ns()
                if self._last_loop_ns is not None:
                    self._measure("loop_gap_ms", (now_ns - self._last_loop_ns) / 1e6)
                self._last_loop_ns = now_ns
                if now_ns >= next_tick_ns:
                    self._tick_started_ns = now_ns
                    previous_prediction = self._last_predict_ns
                    self._tick()
                    ticks_since_report += 1
                    predictions_since_report += int(self._last_predict_ns != previous_prediction)
                    self._measure("tick_ms", (perf_counter_ns() - now_ns) / 1e6)
                    next_tick_ns = max(next_tick_ns + period_ns, perf_counter_ns())
                else:
                    self._drain_inputs()
                now_ns = perf_counter_ns()
                if now_ns - last_report_ns >= 5_000_000_000:
                    report_interval_s = (now_ns - last_report_ns) / 1e9
                    self._record_diagnostic("inference_timing", timing=dict(self._timing),
                        maxima=dict(self._timing_max), streams=self._hub.stream_diagnostics(),
                        configured_rate_hz=self._spec.features.control_rate_hz,
                        tick_rate_hz=ticks_since_report / report_interval_s,
                        prediction_rate_hz=predictions_since_report / report_interval_s,
                        report_interval_s=report_interval_s)
                    self._timing_max.clear()
                    ticks_since_report = predictions_since_report = 0
                    last_report_ns = now_ns
                # Windows Event.wait may round a 1 ms wait to ~15.6 ms,
                # preventing a declared 100 Hz model from actually running at
                # that rate. Python 3.11 sleep uses a high-resolution timer.
                # Short bounded sleeps keep input pumping and Stop responsive;
                # every tick still rechecks the source grid and safety gates.
                remaining_s = (next_tick_ns - now_ns) / 1e9
                if remaining_s > 0:
                    sleep(min(.001, remaining_s))
        except Exception as exc:  # any fault → zero torque + report
            logging.getLogger(__name__).exception("Inference loop failed; stopping output")
            self._last_error = self._last_error or str(exc)
            self._fault_stop(str(exc))
            self.state_changed.emit("fault")
            self.fault.emit(str(exc))
            return
        finally:
            self._fault_stop(None)
        self.state_changed.emit("stopped")

    def _measure(self, name, value):
        self._timing[name] = value
        self._timing_max[name] = max(value, self._timing_max.get(name, 0))

    def _record_diagnostic(self, kind, **fields):
        if self._diagnostic_callback is not None:
            try:
                self._diagnostic_callback(kind, fields)
            except Exception:
                logging.getLogger(__name__).debug("Runtime diagnostics failed", exc_info=True)
        else:
            self.diagnostic_event.emit({"kind": kind, "controller_id": id(self), **fields})

    @property
    def last_fault_details(self):
        return self._last_fault_details

    def _drain_inputs(self, *, catch_up=False):
        started = perf_counter_ns()
        required = tuple(dict.fromkeys(entry.modality for entry in self._spec.features.inputs))
        count = self._hub.drain(
            max_per_stream=256 if catch_up else 64,
            max_duration_per_stream_s=.012 if catch_up else .002,
            priority_modalities=required, only_modalities=required if catch_up else None,
        )
        self._measure("drain_ms", (perf_counter_ns() - started) / 1e6)
        self._timing["drain_items"] = count
        if catch_up:
            self._measure("catch_up_ms", (perf_counter_ns() - started) / 1e6)
            self._timing["catch_up_items"] = count
        return count

    def _input_unavailable(self, reason: str, *, recoverable: bool) -> None:
        with self._state_lock:
            if not (recoverable and self._allow_input_wait_recovery and not self._enabled
                    and not getattr(self._torque, "enabled", False)):
                self._fault_stop(reason)
                return
            entering = self._input_wait_reason is None
            self._input_wait_reason = reason
            if entering:
                self._generation += 1
                self._enabled = False
                self._zero_and_disarm_unlocked()
                # Keep the consumed grid. Clearing it would allow one old
                # window to be replayed after a short queue visibility gap.
                self._assistance.reset()
            now = int(self._clock_ns())
            if entering or self._input_wait_report_ns is None or now - self._input_wait_report_ns >= 1_000_000_000:
                self._input_wait_report_ns = now
                self._record_diagnostic("input_wait", reason=reason,
                    selected_sources=dict(self._imu_extractor.last_diagnostics) if self._imu_extractor else {},
                    streams=self._hub.stream_diagnostics())

    def _tick(self) -> None:
        # An expired prediction is never published. At most one fresh source
        # grid may replace it in this tick; no recursion or automatic re-arm.
        retry = self._tick_attempt()
        if retry is not None:
            self._tick_attempt(retry=retry)

    def _tick_attempt(self, *, retry=None):
        lock_started = perf_counter_ns()
        with self._state_lock:
            self._measure("state_lock_wait_ms", (perf_counter_ns() - lock_started) / 1e6)
            if self._stop_event.is_set():
                return
            if retry is not None and retry[0] != self._generation:
                return
            if retry is not None and getattr(self._torque, "fault", None):
                self._fault_stop("motor fault before prediction retry")
                return
            # Refresh AFTER acquiring the state lock. A GUI validation may
            # have held it since the previous pump; do not test that old buffer.
            self._drain_inputs()
            generation = self._generation
            extract_started = perf_counter_ns()
            self._input_adapter.extractor = self._extractor
            features = self._input_adapter.extract()
            stale = (self._imu_extractor is not None
                     and self._imu_extractor.last_reason == "selected IMU stream is stale")
            if retry is None and features.size == 0 and (stale or self._input_adapter.last_input_expired):
                # A slow first IPC get can consume the ordinary 2 ms budget
                # while many fresh packets remain queued (incident: 22).
                # Give every required stream one bounded catch-up pass before
                # judging freshness; qsize is deliberately not authoritative.
                self._drain_inputs(catch_up=True)
                features = self._input_adapter.extract()
            self._measure("extract_ms", (perf_counter_ns() - extract_started) / 1e6)
            source_result = (
                self._imu_extractor.last_result
                if self._imu_extractor is not None
                else None
            )
        if features.size == 0:
            if retry is not None:
                self._record_diagnostic("prediction_retry_failed", reason=self._input_adapter.last_reason
                    or getattr(self._extractor, "last_reason", None) or "no fresh source grid",
                    previous_grid_end_ns=retry[1], selected_sources=dict(self._imu_extractor.last_diagnostics))
                self._input_unavailable("source window expired before torque send; no fresh retry window",
                                        recoverable=True)
                return
            if self._imu_extractor is not None and self._imu_extractor.last_reason not in {
                None, "warmup", "no new source grid"
            }:
                self._input_unavailable(self._imu_extractor.last_reason,
                    recoverable=self._imu_extractor.waiting_for_valid_window)
            elif self._imu_extractor is None and self._input_adapter.last_input_expired:
                self._input_unavailable(self._input_adapter.last_reason, recoverable=True)
            return
        if retry is not None:
            with self._state_lock:
                if generation != self._generation or self._stop_event.is_set():
                    return
                if source_result is None or source_result.grid_end_ns <= retry[1]:
                    self._input_unavailable("source window expired before torque send; retry grid did not advance",
                                            recoverable=True)
                    return
                self._record_diagnostic("prediction_retry", previous_grid_end_ns=retry[1],
                    grid_end_ns=source_result.grid_end_ns,
                    selected_sources=dict(self._imu_extractor.last_diagnostics))
                if generation != self._generation or self._stop_event.is_set():
                    return
        try:
            predict_started = perf_counter_ns()
            output = self._runtime.predict(features)
            self._measure("predict_ms", (perf_counter_ns() - predict_started) / 1e6)
            if not np.isfinite(output).all():
                raise ModelRuntimeError("model returned a non-finite output")
            left_value, right_value = self._output_adapter.map_output(output)
        except Exception as exc:
            logging.getLogger(__name__).exception("Model prediction failed; stopping output")
            self._last_error = self._last_error or str(exc)
            self._fault_stop(str(exc))
            self._stop_event.set()
            self.fault.emit(str(exc))
            return
        grid_end_ns = getattr(self._extractor, "last_grid_end_ns", None)
        now_ns = (source_result.grid_end_ns if source_result is not None else
                  perf_counter_ns() if grid_end_ns is None else int(grid_end_ns))
        with self._state_lock:
            if generation != self._generation or self._stop_event.is_set():
                return
            if retry is not None and getattr(self._torque, "fault", None):
                self._fault_stop("motor fault during prediction retry")
                return
            expired_reason = self._input_adapter.result_expired_reason()
            if expired_reason:
                self._input_unavailable(expired_reason, recoverable=True)
                return
            if self._imu_extractor is not None and source_result is not None:
                now_wall_ns = int(self._clock_ns())
                latest_source_ns = min(
                    int(source_result.right_source_times_ns[-1]),
                    int(source_result.pelvis_source_times_ns[-1]),
                )
                # A floor-quantized source grid can legitimately trail the
                # wall clock by one control period.  The extractor already
                # enforced the strict 50 ms age at the grid endpoint; this
                # second check only guards a result delayed by an additional
                # whole period while the model was running.
                send_age_limit = int(
                    self._spec.imu_tcn.max_sample_age_s * 1_000_000_000
                    + 1_000_000_000 / self._spec.features.control_rate_hz
                )
                if now_wall_ns - latest_source_ns > send_age_limit:
                    eligible = retry is None and not getattr(self._torque, "fault", None)
                    self._record_diagnostic("prediction_discarded", reason="source window expired before torque send",
                        retry_requested=eligible, attempt=0 if retry is None else 1,
                        checked_at_ns=now_wall_ns, grid_end_ns=now_ns,
                        grid_age_ms=(now_wall_ns - now_ns) / 1e6,
                        right_source_end_ns=int(source_result.right_source_times_ns[-1]),
                        pelvis_source_end_ns=int(source_result.pelvis_source_times_ns[-1]),
                        source_age_ms=(now_wall_ns - latest_source_ns) / 1e6,
                        send_age_limit_ms=send_age_limit / 1e6,
                        current_source_limit_ms=self._spec.imu_tcn.max_sample_age_s * 1000,
                        selected_sources=dict(self._imu_extractor.last_diagnostics))
                    if eligible and generation == self._generation and not self._stop_event.is_set():
                        self._drain_inputs(catch_up=True)
                        return generation, now_ns
                    self._input_unavailable("source window expired before torque send", recoverable=True)
                    return
                if retry is not None:
                    self._record_diagnostic("prediction_retry_accepted", grid_end_ns=now_ns,
                        checked_at_ns=now_wall_ns, source_age_ms=(now_wall_ns - latest_source_ns) / 1e6,
                        send_age_limit_ms=send_age_limit / 1e6,
                        current_source_limit_ms=self._spec.imu_tcn.max_sample_age_s * 1000)
                    if generation != self._generation or self._stop_event.is_set():
                        return
            if self._input_wait_reason is not None:
                self._input_wait_reason = None
                self._input_wait_report_ns = None
                self._record_diagnostic("input_resumed", grid_end_ns=now_ns)
            sample = self._output_adapter.process(now_ns, left_value, right_value)
            if self._recording_callback is not None:
                self._recording_callback({"host_monotonic_ns": perf_counter_ns(),
                    "source_grid_time_ns": now_ns, "raw_output": np.asarray(output).copy(),
                    "model_features": features.copy(), "model_features_dtype": str(features.dtype),
                    "model_features_shape": list(features.shape),
                    "raw_output_unit": self._spec.output.unit, **sample})
            self.pipeline_sample.emit(sample)
            if generation != self._generation or self._stop_event.is_set():
                return
            self._predicted_once = True
            self._last_predict_ns = now_ns
            # Preserve the historical prediction tuple as the converted model
            # estimate; pipeline_sample carries the assisted command.
            self.prediction.emit((now_ns, sample["estimate_left_nm"], sample["estimate_nm"], output))
            # Direct Qt slots may re-enter stop/output-switch methods while
            # this RLock is held.  Revalidate after notifying observers.
            if generation != self._generation or self._stop_event.is_set():
                return
            if sample["assistance_ready"]:
                self._torque.send(float(sample["desired_left_nm"]), float(sample["desired_right_nm"]))
            else:
                self._enabled = False
                self._zero_and_disarm_unlocked()
            command = getattr(self._torque, "last", None)
            if command is not None:
                self.torque_commanded.emit(command)

    def _fault_stop(self, reason: str | None) -> None:
        with self._state_lock:
            self._generation += 1
            self._enabled = False
            self._input_wait_reason = None
            first_fault = bool(reason) and self._last_fault_details is None
            if reason:
                self._stop_event.set()
            if first_fault:
                self._last_error = reason
                timing = dict(self._timing)
                at_ns = perf_counter_ns()
                if self._tick_started_ns is not None:
                    timing["fault_tick_elapsed_ms"] = (at_ns - self._tick_started_ns) / 1e6
                self._last_fault_details = {
                    "reason": reason, "fault_at_ns": at_ns,
                    "timing": timing, "maxima": dict(self._timing_max),
                    "selected_sources": dict(self._imu_extractor.last_diagnostics) if self._imu_extractor else {},
                }
            # Physical safety precedes any queue inspection or log I/O.
            try:
                self._zero_and_disarm_unlocked()
            finally:
                self._reset_processing_history_unlocked()
                if first_fault:
                    self._last_fault_details["streams"] = self._hub.stream_diagnostics()
                    self._record_diagnostic("inference_input_fault", **self._last_fault_details)

    def _zero_and_disarm_unlocked(self) -> None:
        """Always attempt disable, even if a backend's zero write fails."""
        try:
            self._torque.zero()
        finally:
            self._torque.set_enabled(False)

    def _reset_processing_history_unlocked(self) -> None:
        """Reset state that could produce a delayed result after an edge."""
        self._assistance.reset()
        reset = getattr(self._extractor, "reset", None)
        if callable(reset):
            reset()


def _default_extractor(spec: ModelSpec, *, clock_ns=None) -> FeatureExtractor:
    if spec.imu_tcn.enabled:
        from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor

        return ImuTcnFeatureExtractor(spec.imu_tcn, clock_ns=clock_ns)
    from exo_collection.apps.model_runtime.features import DefaultWindowFeatureExtractor

    return DefaultWindowFeatureExtractor()


__all__ = [
    "InferenceController",
    "map_output_to_torque",
]
