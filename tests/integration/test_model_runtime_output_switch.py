"""Runtime output attachment and stop-edge tests.

These tests deliberately keep inference running with a disarmed mock sink: the
model and middle layer are allowed to produce observations while the window
attaches, detaches, or replaces a motor output.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import Qt

from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


def _spec(*, unit: str = "Nm") -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": "1.0.0",
            "model": {"backend": "demo", "demo": {"amplitude": 1.0}},
            "features": {
                "control_rate_hz": 100.0,
                "window_s": 0.03,
                "inputs": [{"modality": "imu", "channels": None}],
            },
            "output": {
                "kind": "torque_left_right",
                "single_output_maps_to": "right_torque",
                "unit": unit,
            },
            "assistance": {
                "mass_kg": 2.0,
                "gain": 1.0,
                "delay_s": 0.0,
            },
        }
    )


def _controller(*args, **kwargs):
    return InferenceController(*args, **kwargs, clock_ns=lambda: 1_020_000_000)


def _hub() -> SubscriptionHub:
    hub = SubscriptionHub()
    from queue import Queue

    queue: Queue = Queue()
    hub.register_stream("imu", queue, channels=("x",), sample_shape=(1,), rate_hz=100.0)
    queue.put(
        SampleBatch(
            device_id="imu",
            modality="imu",
            clock_domain="clock",
            first_sample_index=0,
            sample_count=3,
            sequence_number=0,
            sample_rate_hz=100.0,
            host_monotonic_ns=1_000_000_000,
            data=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        )
    )
    hub.drain()
    return hub


class _BlockingRuntime:
    output_size = 1
    backend = "fake"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.entered.set()
        assert self.release.wait(2.0)
        return np.asarray([1.0], dtype=np.float32)


class _TrackingMock(MockTorqueOutput):
    def __init__(self) -> None:
        super().__init__(5.0, enable_required=False)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


def test_output_switch_discards_inflight_result_and_does_not_close_sinks() -> None:
    old_output = _TrackingMock()
    new_output = _TrackingMock()
    runtime = _BlockingRuntime()
    controller = _controller(_hub(), _spec(), runtime, old_output)
    predictions: list[object] = []
    controller.prediction.connect(predictions.append)

    worker = threading.Thread(target=controller._tick)
    worker.start()
    assert runtime.entered.wait(1.0)

    controller.set_torque_output(new_output)
    assert old_output.enabled is False
    assert old_output.last is not None
    assert (old_output.last.left_nm, old_output.last.right_nm) == (0.0, 0.0)
    assert new_output.enabled is False
    assert old_output.closed is False
    assert new_output.closed is False

    runtime.release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert predictions == []
    assert all(command.left_nm == 0.0 and command.right_nm == 0.0 for command in new_output.history)


def test_disarmed_replacement_still_emits_prediction_and_middle_layer_sample() -> None:
    class FakeImuExtractor:
        last_result = None
        last_reason = None

        def __init__(self) -> None:
            self.reset_count = 0

        def extract(self, hub, spec) -> np.ndarray:
            return np.ones((1, 3), dtype=np.float32)

        def reset(self) -> None:
            self.reset_count += 1

    first = _TrackingMock()
    second = _TrackingMock()
    controller = _controller(
        _hub(), _spec(unit="Nm/kg"), build_model_runtime(_spec(unit="Nm/kg")), first
    )
    fake_extractor = FakeImuExtractor()
    controller._extractor = fake_extractor
    controller._imu_extractor = fake_extractor
    predictions: list[object] = []
    pipeline_samples: list[object] = []
    controller.prediction.connect(predictions.append)
    controller.pipeline_sample.connect(pipeline_samples.append)

    controller._tick()
    controller.set_torque_output(second)
    controller._tick()

    assert len(predictions) == 2
    assert len(pipeline_samples) == 2
    assert fake_extractor.reset_count == 0
    assert second.enabled is False
    assert second.last is not None
    assert (second.last.left_nm, second.last.right_nm) == (0.0, 0.0)


def test_request_stop_invalidates_blocked_tick_and_unstarted_stop_is_successful() -> None:
    unstarted_output = _TrackingMock()
    unstarted = _controller(_hub(), _spec(), build_model_runtime(_spec()), unstarted_output)
    assert unstarted.stop_and_wait(timeout_ms=1) is True
    assert unstarted_output.enabled is False

    output = _TrackingMock()
    runtime = _BlockingRuntime()
    controller = _controller(_hub(), _spec(), runtime, output)
    controller.set_enabled(True)
    predictions: list[object] = []
    controller.prediction.connect(predictions.append)
    controller.start()
    assert runtime.entered.wait(1.0)

    controller.request_stop()
    assert output.enabled is False
    assert output.last is not None
    assert (output.last.left_nm, output.last.right_nm) == (0.0, 0.0)
    runtime.release.set()
    assert controller.stop_and_wait(timeout_ms=2_000) is True
    assert not controller.isRunning()
    assert predictions == []
    assert all(command.left_nm == 0.0 and command.right_nm == 0.0 for command in output.history)


class _CountingOutput(_TrackingMock):
    def __init__(self) -> None:
        super().__init__()
        self.send_calls: list[tuple[float, float]] = []
        self.fail_zero = False

    def send(self, left_nm: float, right_nm: float) -> None:
        self.send_calls.append((left_nm, right_nm))
        super().send(left_nm, right_nm)

    def zero(self) -> None:
        if self.fail_zero:
            raise OSError("zero write failed")
        super().zero()


class _ConstantRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray([1.0], dtype=np.float32)


@pytest.mark.parametrize("signal_name", ["prediction", "pipeline_sample"])
@pytest.mark.parametrize("edge", ["switch", "stop"])
def test_direct_observer_cannot_submit_current_result_after_output_edge(signal_name, edge) -> None:
    first, second = _CountingOutput(), _CountingOutput()
    controller = _controller(_hub(), _spec(), _ConstantRuntime(), first)

    class FakeImuExtractor:
        last_result = None
        last_reason = None

        def extract(self, hub, spec):
            return np.ones((1, 3), dtype=np.float32)

        def preflight(self, hub, spec):
            return {"ready": True, "errors": []}

    if signal_name == "pipeline_sample":
        controller._extractor = controller._imu_extractor = FakeImuExtractor()
    controller.set_enabled(True)

    def change_output(_sample):
        if edge == "switch":
            controller.set_torque_output(second)
            controller.set_enabled(True)
        else:
            controller.request_stop()

    getattr(controller, signal_name).connect(change_output, Qt.ConnectionType.DirectConnection)
    controller._tick()
    assert first.send_calls == []
    assert second.send_calls == []
    assert first.enabled is False


@pytest.mark.parametrize("edge", ["switch", "stop", "disable"])
def test_zero_failure_still_disarms_and_invalidates_inflight_tick(edge) -> None:
    output = _CountingOutput()
    replacement = _CountingOutput()
    runtime = _BlockingRuntime()
    controller = _controller(_hub(), _spec(), runtime, output)
    controller.set_enabled(True)
    worker = threading.Thread(target=controller._tick)
    worker.start()
    assert runtime.entered.wait(1.0)
    output.fail_zero = True
    try:
        with pytest.raises(OSError, match="zero write failed"):
            if edge == "switch":
                controller.set_torque_output(replacement)
            elif edge == "stop":
                controller.request_stop()
            else:
                controller.set_enabled(False)
        assert controller.enabled is False
        assert output.enabled is False
    finally:
        runtime.release.set()
        worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert output.send_calls == []
    assert replacement.send_calls == []


def test_stopped_controller_rejects_enable_and_does_not_predict_again() -> None:
    runtime, output = _ConstantRuntime(), _CountingOutput()
    controller = _controller(_hub(), _spec(), runtime, output)
    controller.set_enabled(True)
    controller.request_stop()
    with pytest.raises(ValueError, match="stopped inference controller"):
        controller.set_enabled(True)
    controller._tick()
    assert runtime.calls == 0
    assert output.send_calls == []
    assert output.enabled is False


def test_reattaching_same_output_is_idempotent_during_prediction() -> None:
    runtime, output = _ConstantRuntime(), _CountingOutput()
    controller = _controller(_hub(), _spec(), runtime, output)
    controller.set_enabled(True)
    controller.prediction.connect(
        lambda _sample: controller.set_torque_output(output), Qt.ConnectionType.DirectConnection
    )
    controller._tick()
    assert controller.enabled is True
    assert output.enabled is True
    assert output.send_calls == [(0.0, 1.0)]
