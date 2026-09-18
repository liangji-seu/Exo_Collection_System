"""Unit tests for output→torque mapping and the inference loop."""

from __future__ import annotations

import time
from queue import Queue

import numpy as np
import pytest

from exo_collection.apps.model_runtime.backends import ModelRuntimeError, build_model_runtime
from exo_collection.apps.model_runtime.inference import InferenceController, map_output_to_torque
from exo_collection.apps.model_runtime.spec import ModelSpec, OutputSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


def _output(kind="torque_left_right", single_output_maps_to=None) -> OutputSpec:
    return OutputSpec(kind=kind, single_output_maps_to=single_output_maps_to, torque_limit_nm=5.0)


def _spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": "1.0.0",
            "model": {"backend": "demo", "demo": {"amplitude": 1.0}},
            "features": {
                "control_rate_hz": 100.0,
                "window_s": 0.03,
                "inputs": [{"modality": "imu", "channels": None}],
            },
            "output": {"kind": "torque_left_right", "single_output_maps_to": "right_torque"},
        }
    )


def _batch(data: np.ndarray, t_ns: int) -> SampleBatch:
    return SampleBatch(
        device_id="i",
        modality="imu",
        clock_domain="c",
        first_sample_index=0,
        sample_count=int(data.shape[0]),
        sequence_number=0,
        sample_rate_hz=100.0,
        host_monotonic_ns=t_ns,
        data=data.astype(np.float32),
    )


# -- mapping ---------------------------------------------------------------


def test_map_single_to_right() -> None:
    assert map_output_to_torque(np.array([2.5]), _output(single_output_maps_to="right_torque")) == (0.0, 2.5)


def test_map_single_to_left() -> None:
    assert map_output_to_torque(np.array([2.5]), _output(single_output_maps_to="left_torque")) == (2.5, 0.0)


def test_map_single_defaults_to_right() -> None:
    assert map_output_to_torque(np.array([3.0]), _output()) == (0.0, 3.0)


def test_map_pair() -> None:
    assert map_output_to_torque(np.array([1.0, -2.0]), _output()) == (1.0, -2.0)


def test_map_unsupported_size_raises() -> None:
    with pytest.raises(ModelRuntimeError):
        map_output_to_torque(np.array([1.0, 2.0, 3.0]), _output())


def test_map_unsupported_kind_raises() -> None:
    from types import SimpleNamespace

    bogus = SimpleNamespace(kind="not_a_kind", single_output_maps_to=None)
    with pytest.raises(ModelRuntimeError):
        map_output_to_torque(np.array([1.0]), bogus)


# -- controller loop -------------------------------------------------------


def test_controller_tick_predicts_and_sends() -> None:
    hub = SubscriptionHub()
    queue: Queue = Queue()
    hub.register_stream("imu", queue, channels=("x",), sample_shape=(1,), rate_hz=100.0)
    queue.put(_batch(np.array([[0.0], [1.0], [2.0]], dtype=np.float32), 0))
    hub.drain()

    spec = _spec()
    output = MockTorqueOutput(5.0, enable_required=False)
    controller = InferenceController(hub, spec, build_model_runtime(spec), output)
    controller.set_enabled(True)
    controller._tick()

    assert output.history
    assert output.last.enabled is True
    assert controller.last_predict_ns is not None
    assert controller.last_error is None


def test_controller_loop_starts_and_stops() -> None:
    hub = SubscriptionHub()
    queue: Queue = Queue()
    hub.register_stream("imu", queue, channels=("x",), sample_shape=(1,), rate_hz=100.0)
    queue.put(_batch(np.array([[0.0], [1.0], [2.0]], dtype=np.float32), 0))

    spec = _spec()
    output = MockTorqueOutput(5.0, enable_required=False)
    controller = InferenceController(hub, spec, build_model_runtime(spec), output)
    controller.set_enabled(True)
    controller.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(output.history) < 2:
        time.sleep(0.01)
    controller.stop_and_wait(timeout_ms=2000)
    assert not controller.isRunning()
    assert len(output.history) >= 1


def test_controller_fault_zeroes_torque() -> None:
    hub = SubscriptionHub()
    queue: Queue = Queue()
    hub.register_stream("imu", queue, channels=("x",), sample_shape=(1,), rate_hz=100.0)
    queue.put(_batch(np.array([[0.0], [1.0], [2.0]], dtype=np.float32), 0))
    spec = _spec()

    class ExplodingRuntime:
        def predict(self, features):
            raise RuntimeError("boom")

    output = MockTorqueOutput(5.0, enable_required=False)
    output.send(1.0, 1.0)
    controller = InferenceController(hub, spec, ExplodingRuntime(), output)
    controller.set_enabled(True)
    controller.start()
    controller.wait(2000)
    assert not controller.isRunning()
    assert output.last.left_nm == 0.0
    assert output.last.right_nm == 0.0
    assert controller.last_error == "boom"
