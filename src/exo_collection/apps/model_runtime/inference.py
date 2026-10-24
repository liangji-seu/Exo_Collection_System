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
from dataclasses import asdict, replace

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
from exo_collection.apps.model_runtime.gate_compatibility import gate_compatibility_error
from exo_collection.apps.model_runtime.model_adapter import ModelInputAdapter, ModelOutputAdapter, map_output_values
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import TorqueOutput

_GRID_RETRY_POLL_NS = 1_000_000


def _quick_grid_retry_deadline(now_ns: int, reason: str | None,
                               next_regular_ns: int) -> int | None:
    retry_ns = int(now_ns) + _GRID_RETRY_POLL_NS
    if reason != "no new source grid" or retry_ns >= int(next_regular_ns):
        return None
    return retry_ns


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
        gate_mode: str = "none",
        gate_recording_callback=None,
        gate_process_isolated: bool = True,
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
        self._gate_recording_callback = gate_recording_callback
        self._last_gate_recorded_state = None
        self._last_gate_recorded_provenance: dict | None = None
        self._gate_fault_latched = False
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
        from exo_collection.apps.model_runtime.gate_dispatch import GateDispatch
        from exo_collection.apps.model_runtime.gate_worker import GateWorker
        self._gate_dispatch = GateDispatch(gate_mode)
        self._gate_worker = GateWorker(gate_mode, clock_ns=self._clock_ns,
                                       process_isolated=gate_process_isolated)
        self._gate_worker.start()
        self._gate_cursor_ns: int | None = None
        self._run_active = False
        self._gate_epoch: int | None = None
        self._gate_last_output_epoch: int | None = None
        self._gate_epoch_start_ns: int | None = None
        self._gate_was_open = False
        self._gate_zeroed_for_expiry = False
        self._last_gate_status = None
        self._last_gate_sample: dict[str, object] | None = None
        self._gate_watchdog_stop = threading.Event()
        self._gate_watchdog = None
        if gate_mode != "none":
            self._gate_watchdog = threading.Thread(target=self._gate_watchdog_loop,
                name="gate-ttl-watchdog", daemon=True)
            self._gate_watchdog.start()

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
    def gate_status(self):
        return self._last_gate_status

    @property
    def gate_sample(self):
        return None if self._last_gate_sample is None else dict(self._last_gate_sample)

    @property
    def gate_provenance(self) -> dict:
        return dict(self._gate_worker.provenance)

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
                self._reset_gate_unlocked()
                return
            else:
                self._reset_gate_unlocked()
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
        report = self._input_adapter.preflight()
        compatibility_error = gate_compatibility_error(
            self._spec, self._gate_dispatch.mode,
            has_imu_tcn_extractor=self._imu_extractor is not None,
        )
        if compatibility_error:
            report = dict(report)
            report["errors"] = list(report.get("errors", [])) + [compatibility_error]
            report["ready"] = False
        return report

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
            self._gate_worker.close()
            self._gate_watchdog_stop.set()

    def stop_and_wait(self, timeout_ms: int = 3000) -> bool:
        self.request_stop()
        if not self.isRunning():
            return True
        return self.wait(timeout_ms)

    # -- loop --------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - explicit safety handling
        self._run_active = True
        self.state_changed.emit("running")
        period_ns = int(1e9 / self._spec.features.control_rate_hz)
        next_tick_ns = perf_counter_ns()
        retry_tick_ns = None
        last_report_ns = next_tick_ns
        ticks_since_report = predictions_since_report = 0
        no_grid_retries = no_grid_recovered = 0
        try:
            while not self._stop_event.is_set():
                now_ns = perf_counter_ns()
                if self._last_loop_ns is not None:
                    self._measure("loop_gap_ms", (now_ns - self._last_loop_ns) / 1e6)
                self._last_loop_ns = now_ns
                scheduled_tick = now_ns >= next_tick_ns
                retry_tick = retry_tick_ns is not None and now_ns >= retry_tick_ns
                if scheduled_tick or retry_tick:
                    self._tick_started_ns = now_ns
                    previous_prediction = self._last_predict_ns
                    self._tick()
                    predictions_since_report += int(self._last_predict_ns != previous_prediction)
                    if scheduled_tick:
                        ticks_since_report += 1
                        next_tick_ns = max(next_tick_ns + period_ns, perf_counter_ns())
                    elif retry_tick:
                        no_grid_retries += 1
                        no_grid_recovered += int(
                            self._last_predict_ns != previous_prediction)
                    reason = None if self._imu_extractor is None else self._imu_extractor.last_reason
                    retry_tick_ns = _quick_grid_retry_deadline(
                        perf_counter_ns(), reason, next_tick_ns)
                    self._measure("tick_ms", (perf_counter_ns() - now_ns) / 1e6)
                else:
                    self._drain_inputs()
                self._expire_gate_if_needed()
                now_ns = perf_counter_ns()
                if now_ns - last_report_ns >= 5_000_000_000:
                    report_interval_s = (now_ns - last_report_ns) / 1e9
                    self._record_diagnostic("inference_timing", timing=dict(self._timing),
                        maxima=dict(self._timing_max), streams=self._hub.stream_diagnostics(),
                        configured_rate_hz=self._spec.features.control_rate_hz,
                        tick_rate_hz=ticks_since_report / report_interval_s,
                        prediction_rate_hz=predictions_since_report / report_interval_s,
                        no_new_grid_retries=no_grid_retries,
                        no_new_grid_recovered=no_grid_recovered,
                        report_interval_s=report_interval_s)
                    self._timing_max.clear()
                    ticks_since_report = predictions_since_report = 0
                    no_grid_retries = no_grid_recovered = 0
                    last_report_ns = now_ns
                # Windows Event.wait may round a 1 ms wait to ~15.6 ms,
                # preventing a declared 100 Hz model from actually running at
                # that rate. Python 3.11 sleep uses a high-resolution timer.
                # Short bounded sleeps keep input pumping and Stop responsive;
                # every tick still rechecks the source grid and safety gates.
                next_deadline_ns = next_tick_ns
                if retry_tick_ns is not None:
                    next_deadline_ns = min(next_deadline_ns, retry_tick_ns)
                remaining_s = (next_deadline_ns - now_ns) / 1e9
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
            self._run_active = False
            self._fault_stop(None)
            self._gate_worker.close(timeout=.1)
            self._gate_watchdog_stop.set()
        if self._last_fault_details is None:
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
            if (source_result is not None and source_result.grid_times_ns is not None
                    and source_result.raw_values is not None and self._gate_worker.mode != "none"):
                # Gate history starts at the explicit arm edge. While disarmed,
                # advance the cursor without replaying old hub history.
                for stamp, values in zip(source_result.grid_times_ns, source_result.raw_values):
                    stamp = int(stamp)
                    if self._gate_cursor_ns is not None and stamp <= self._gate_cursor_ns:
                        continue
                    self._gate_cursor_ns = stamp
                    if (self._enabled or self._run_active) and not self._gate_worker.submit(
                            stamp, values, generation, valid=bool(np.isfinite(values).all())):
                        self._record_diagnostic("gate_queue_overflow", source_grid_ns=stamp,
                                                generation=generation,
                                                dropped=self._gate_worker.dropped)
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
            gate_pre_left = sample["desired_left_nm"]
            gate_pre_right = sample["desired_right_nm"]
            gate_epoch = None
            if self._gate_dispatch.mode != "none":
                if not self._collect_gate_results():
                    return
                now_wall = int(self._clock_ns())
                delayed_source_ns = sample.get("delayed_source_time_ns")
                candidate_source_ns = (int(delayed_source_ns) if delayed_source_ns is not None
                                       else max(0, now_ns-int(self._spec.assistance.delay_s*1e9)))
                allowed, gate_epoch, gate_status = self._gate_dispatch.decide(
                    source_grid_ns=now_ns, now_ns=now_wall,
                    worker_closed=self._gate_worker.closed_reason,
                    candidate_source_ns=candidate_source_ns)
                if not allowed and self._gate_worker.state == "loading" and gate_status.state == "warming":
                    gate_status = replace(gate_status, state="loading",
                                          reason="loading verified gate assets")
                reopened = allowed and self._gate_last_output_epoch != gate_epoch
                if reopened:
                    self._gate_epoch_start_ns = int(gate_status.source_grid_ns)
                epoch_candidate_ok = (self._gate_epoch_start_ns is None
                    or (delayed_source_ns is not None
                        and int(delayed_source_ns) >= self._gate_epoch_start_ns))
                if not allowed:
                    sample["desired_left_nm"] = 0.0
                    sample["desired_right_nm"] = 0.0
                    sample["gate_allowed"] = False
                    sample["gate_status"] = gate_status
                    self._gate_was_open = False
                    self._gate_epoch = None
                    self._gate_zeroed_for_expiry = True
                else:
                    sample["gate_allowed"] = True
                    if not epoch_candidate_ok:
                        sample["desired_left_nm"] = 0.0
                        sample["desired_right_nm"] = 0.0
                        sample["gate_allowed"] = False
                        gate_status = replace(gate_status, state="delay_warming",
                                              reason="waiting for post-reopen delayed assistance sample")
                    sample["gate_status"] = gate_status
                    self._gate_was_open = True
                    self._gate_epoch = gate_epoch
                    self._gate_zeroed_for_expiry = False
                    self._gate_last_output_epoch = gate_epoch
                self._record_gate_transition(gate_status, now_wall)
                # Record the command before and after gate separately. The
                # original desired values were captured before any closure.
                sample["gate_generation"] = generation
                sample["gate_epoch"] = gate_epoch
                sample["gate_pre_left_nm"] = gate_pre_left
                sample["gate_pre_right_nm"] = gate_pre_right
                sample["gate_post_left_nm"] = sample["desired_left_nm"]
                sample["gate_post_right_nm"] = sample["desired_right_nm"]
            else:
                sample["gate_allowed"] = True
                sample["gate_status"] = None
                sample["gate_pre_left_nm"] = gate_pre_left
                sample["gate_pre_right_nm"] = gate_pre_right
                sample["gate_post_left_nm"] = gate_pre_left
                sample["gate_post_right_nm"] = gate_pre_right
            self._last_gate_status = sample.get("gate_status")
            self._last_gate_sample = dict(sample)
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
            command_gate_context = {
                "mode": self._gate_dispatch.mode,
                "sequence": None if self._last_gate_status is None else self._last_gate_status.sequence,
                "source_grid_ns": None if self._last_gate_status is None else self._last_gate_status.source_grid_ns,
                "generation": generation,
                "epoch": sample.get("gate_epoch"),
                "pre_left_nm": sample.get("gate_pre_left_nm"),
                "pre_right_nm": sample.get("gate_pre_right_nm"),
                "post_left_nm": sample.get("gate_post_left_nm"),
                "post_right_nm": sample.get("gate_post_right_nm"),
                "state": None if self._last_gate_status is None else self._last_gate_status.state,
                "reason": ("delay_epoch_warmup" if self._last_gate_status is not None
                           and self._last_gate_status.state == "delay_warming" else
                           None if self._last_gate_status is None else self._last_gate_status.reason),
                "output_allowed": bool(sample.get("gate_allowed", True)
                                        and sample.get("assistance_ready", True)),
            }
            if self._gate_dispatch.mode != "none":
                # Warming and rejection are ordinary gate states. Keep the
                # explicitly started session alive and hold zero output.
                if sample["assistance_ready"]:
                    self._torque.send_with_gate_context(
                        float(sample["desired_left_nm"]),
                        float(sample["desired_right_nm"]), command_gate_context)
                else:
                    self._torque.send_with_gate_context(0.0, 0.0, command_gate_context)
            elif sample["assistance_ready"]:
                command_gate_context["state"] = "disabled"
                self._torque.send_with_gate_context(float(sample["desired_left_nm"]),
                                                     float(sample["desired_right_nm"]),
                                                     command_gate_context)
            else:
                self._enabled = False
                self._zero_and_disarm_unlocked()
            command = getattr(self._torque, "last", None)
            if command is not None:
                self.torque_commanded.emit(command)

    def _reset_gate_unlocked(self) -> None:
        if self._gate_dispatch.mode == "none":
            return
        self._gate_dispatch.reset(self._generation)
        self._gate_worker.reset(generation=self._generation)
        self._gate_cursor_ns = (self._imu_extractor.last_grid_end_ns
                                if self._imu_extractor is not None else None)
        self._gate_epoch = None
        self._gate_last_output_epoch = None
        self._gate_epoch_start_ns = None
        self._gate_was_open = False
        self._gate_zeroed_for_expiry = False
        self._last_gate_recorded_state = None

    def _collect_gate_results(self) -> bool:
        results = self._gate_worker.drain()
        self._gate_dispatch.publish(results)
        worker_fault = self._gate_worker.closed_reason
        if worker_fault:
            self._latch_gate_fault(f"assistance gate fault: {worker_fault}")
            return False
        provenance = dict(self._gate_worker.provenance)
        if (self._gate_dispatch.mode != "none" and provenance
                and provenance != self._last_gate_recorded_provenance):
            self._last_gate_recorded_provenance = provenance
            if self._gate_recording_callback is not None:
                try:
                    self._gate_recording_callback("gate_provenance", provenance)
                except Exception as exc:
                    self._record_diagnostic("gate_recording_failed", reason=str(exc))
        if self._gate_recording_callback is None:
            return True
        for result in results:
            decision = result.decision
            payload = {**asdict(decision), "computed_at_ns": int(result.computed_at_ns),
                       "compute_ms": float(result.compute_ms),
                       "generation": int(result.generation),
                       "queue_depth": self._gate_worker.queue_depth,
                       "queue_dropped": self._gate_worker.dropped}
            try:
                self._gate_recording_callback("gate_decision", payload)
            except Exception as exc:
                self._record_diagnostic("gate_recording_failed", reason=str(exc))
        return True

    def _record_gate_transition(self, status, at_ns: int) -> None:
        if status.state == self._last_gate_recorded_state:
            return
        self._last_gate_recorded_state = status.state
        if self._gate_recording_callback is not None:
            try:
                self._gate_recording_callback("gate_transition",
                    {**asdict(status), "host_monotonic_ns": int(at_ns),
                     "worker_state": self._gate_worker.state,
                     "loading_samples_skipped": self._gate_worker.skipped_loading,
                     "queue_depth": self._gate_worker.queue_depth,
                     "queue_dropped": self._gate_worker.dropped})
            except Exception as exc:
                self._record_diagnostic("gate_recording_failed", reason=str(exc))

    def _latch_gate_fault(self, reason: str) -> None:
        if self._gate_fault_latched:
            return
        self._gate_fault_latched = True
        try:
            self._fault_stop(reason)
        except Exception as exc:
            self._record_diagnostic("gate_fault_cleanup_failed", reason=str(exc))
        finally:
            self.state_changed.emit("fault")
            self.fault.emit(reason)

    def _expire_gate_if_needed(self) -> None:
        if self._gate_dispatch.mode == "none" or not self._enabled:
            return
        with self._state_lock:
            if not self._collect_gate_results():
                return
            if self._gate_dispatch.expire(int(self._clock_ns())):
                self._gate_was_open = False
                self._gate_epoch = None
                self._gate_zeroed_for_expiry = True
                status = self._gate_dispatch.status
                self._torque.send_with_gate_context(0.0, 0.0, {
                    "mode": self._gate_dispatch.mode,
                    "sequence": status.sequence, "source_grid_ns": status.source_grid_ns,
                    "generation": self._generation, "epoch": None,
                    "pre_left_nm": None, "pre_right_nm": None,
                    "post_left_nm": 0.0, "post_right_nm": 0.0,
                    "state": "stale"})
                self._record_diagnostic("gate_transition", state="stale",
                                        reason=self._gate_dispatch.status.reason,
                                        generation=self._generation)
                self._record_gate_transition(self._gate_dispatch.status,
                                             int(self._clock_ns()))

    def _gate_watchdog_loop(self) -> None:
        # This independent 5 ms monitor remains live while model.predict or a
        # device callback stalls. It only closes the current output epoch.
        while not self._gate_watchdog_stop.wait(.005):
            with self._state_lock:
                if not self._enabled or self._gate_dispatch.mode == "none":
                    continue
                if not self._collect_gate_results():
                    continue
                if self._gate_dispatch.expire(int(self._clock_ns())):
                    self._gate_was_open = False
                    self._gate_epoch = None
                    self._gate_zeroed_for_expiry = True
                    try:
                        self._torque.send_with_gate_context(0.0, 0.0, {
                            "mode": self._gate_dispatch.mode,
                            "sequence": None if self._last_gate_status is None else self._last_gate_status.sequence,
                            "source_grid_ns": None if self._last_gate_status is None else self._last_gate_status.source_grid_ns,
                            "generation": self._generation, "epoch": None,
                            "pre_left_nm": None, "pre_right_nm": None,
                            "post_left_nm": 0.0, "post_right_nm": 0.0,
                            "state": "stale"})
                    except Exception as exc:
                        self._record_diagnostic("gate_ttl_zero_failed", reason=str(exc),
                                                generation=self._generation)
                        self._latch_gate_fault(f"gate TTL zero failed: {exc}")
                    self._record_diagnostic("gate_transition", state="stale",
                        reason=self._gate_dispatch.status.reason,
                        generation=self._generation)
                    self._record_gate_transition(self._gate_dispatch.status,
                                                 int(self._clock_ns()))

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
            self._torque.set_gate_context({
                "mode": self._gate_dispatch.mode,
                "sequence": None if self._last_gate_status is None else self._last_gate_status.sequence,
                "source_grid_ns": None if self._last_gate_status is None else self._last_gate_status.source_grid_ns,
                "generation": self._generation, "epoch": None,
                "pre_left_nm": None, "pre_right_nm": None,
                "post_left_nm": 0.0, "post_right_nm": 0.0,
                "output_allowed": False, "state": "closed", "reason": "session_closed"})
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
