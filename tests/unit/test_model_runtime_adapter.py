"""Contract boundaries across model shapes, units and bilateral mappings."""
from queue import Queue

import numpy as np
import pytest

from exo_collection.apps.model_runtime.features import DefaultWindowFeatureExtractor
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.model_adapter import ModelInputAdapter, ModelOutputAdapter
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


def spec(*, unit="Nm", rate=10, window=0.3, mass=0, channels=None, target="right_torque"):
    return ModelSpec.model_validate({
        "model": {"backend": "demo", "input_layout": "btc"},
        "features": {"control_rate_hz": rate, "window_s": window,
                     "inputs": [{"modality": "imu", "channels": channels or ["y", "x"]}]},
        "output": {"unit": unit, "single_output_maps_to": target},
        "assistance": {"mass_kg": mass, "gain": 0.5, "right_sign": -1, "lowpass_hz": 0},
    })


def input_adapter(hub, model, extractor):
    return ModelInputAdapter(hub, model, extractor, clock_ns=lambda: max((hub.latest(m)[0] for m in hub.modalities if hub.latest(m) is not None), default=0))


def hub_and_queue():
    hub, queue = SubscriptionHub(), Queue()
    hub.register_stream("imu", queue, channels=("x", "y"), sample_shape=(2,), rate_hz=10)
    return hub, queue


def append(hub, queue, data, t=1_000_000_000, rate=10):
    data = np.asarray(data, dtype=np.float32)
    queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host",
        first_sample_index=0, sample_count=len(data), sequence_number=0,
        sample_rate_hz=rate, host_monotonic_ns=t, data=data))
    hub.drain()


@pytest.mark.parametrize("rate,count", [(10, 3), (50, 7), (2, 2)])
def test_declared_window_rate_channel_order_and_nonconsuming_probe(rate, count):
    hub, queue = hub_and_queue()
    model = spec(rate=rate, window=count/rate)
    data = [[i, 100+i] for i in range(count)]
    append(hub, queue, data, rate=rate)
    extractor = DefaultWindowFeatureExtractor()
    adapter = input_adapter(hub, model, extractor)
    report = adapter.preflight()
    assert report["ready"], report
    assert report["contract"]["input_layout"] == "BTC"
    assert report["contract"]["window_samples"] == count
    assert extractor.last_grid_end_ns is None
    tensor = adapter.extract()
    assert tensor.shape == (2, count)
    np.testing.assert_array_equal(tensor[0], np.arange(count)+100)
    np.testing.assert_array_equal(tensor[1], np.arange(count))
    assert adapter.preflight()["ready"]
    assert adapter.extract().size == 0


def test_missing_modality_short_window_bad_channel_and_normalization_are_not_ready():
    model = spec()
    empty = input_adapter(SubscriptionHub(), model, DefaultWindowFeatureExtractor())
    assert not empty.preflight()["ready"]
    hub, queue = hub_and_queue()
    append(hub, queue, [[1, 2]])
    adapter = input_adapter(hub, model, DefaultWindowFeatureExtractor())
    assert not adapter.preflight()["ready"]
    assert adapter.extract().size == 0  # No first-sample padding into earlier grid.
    append(hub, queue, [[1, 2], [3, 4], [5, 6]])
    bad_channel = spec(channels=["absent"])
    assert not ModelInputAdapter(hub, bad_channel, DefaultWindowFeatureExtractor()).preflight()["ready"]
    model.features.normalization.mean = [0, 0]
    model.features.normalization.std = [1, 0]
    assert not adapter.preflight()["ready"]
    model.features.normalization.std = [1, 1]
    append(hub, queue, [[1, 2], [np.nan, 4], [5, 6]], t=2_000_000_000)
    assert not adapter.preflight()["ready"]


def test_async_modalities_use_causal_slowest_watermark():
    hub, queue = hub_and_queue()
    other = Queue()
    hub.register_stream("slow", other, channels=("z",), sample_shape=(1,), rate_hz=5)
    model = spec()
    model.safety.watchdog_s = 1.0
    from exo_collection.apps.model_runtime.spec import FeatureInput
    model.features.inputs.append(FeatureInput(modality="slow", channels=["z"]))
    append(hub, queue, [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]])
    append(hub, other, [[100], [200]], rate=5)
    adapter = input_adapter(hub, model, DefaultWindowFeatureExtractor())
    tensor = adapter.extract()
    assert tensor[2].tolist() == [100, 100, 200]
    append(hub, queue, [[5, 15]], t=1_500_000_000)
    assert adapter.extract().size == 0


@pytest.mark.parametrize("unit,mass,scale", [("Nm", 0, 1), ("Nm", 60, 1), ("Nm/kg", 60, 60)])
def test_all_outputs_use_one_unit_conversion_and_bilateral_middle_layer(unit, mass, scale):
    model = spec(unit=unit, mass=mass)
    adapter = ModelOutputAdapter(model)
    sample = adapter.process(1, 2, 3)
    assert sample["assistance_ready"]
    assert sample["estimate_left_nm"] == 2*scale
    assert sample["estimate_nm"] == 3*scale
    assert sample["desired_left_nm"] == scale
    assert sample["desired_right_nm"] == -1.5*scale
    assert sample["estimate_nm_per_kg"] == (3 if unit == "Nm/kg" else None)


def test_normalized_prediction_survives_missing_mass_but_cannot_arm_or_command_nm():
    hub, queue = hub_and_queue()
    append(hub, queue, [[1, 2], [3, 4], [5, 6]])
    model = spec(unit="Nm/kg")
    class Runtime:
        def predict(self, features):
            return np.array([2.0], dtype=np.float32)
    sink = MockTorqueOutput(5, enable_required=False)
    controller = InferenceController(hub, model, Runtime(), sink, clock_ns=lambda: 1_200_000_000)
    predictions, samples = [], []
    controller.prediction.connect(predictions.append)
    controller.pipeline_sample.connect(samples.append)
    assert controller.preflight()["ready"]
    assert not controller.assistance_preflight()["ready"]
    with pytest.raises(ValueError, match="体重"):
        controller.set_enabled(True)
    assert controller.last_error is None
    controller._tick()
    assert predictions[0][1:3] == (None, None)
    assert samples[0]["estimate_nm_per_kg"] == 2
    assert samples[0]["raw_estimate_right"] == 2
    assert samples[0]["estimate_nm"] is None
    assert samples[0]["desired_right_nm"] is None
    assert not sink.enabled
    assert sink.last.left_nm == sink.last.right_nm == 0
    append(hub, queue, [[7, 8]], t=1_300_000_000)
    controller._tick()
    assert len(predictions) == 2
    assert controller.last_error is None


@pytest.mark.parametrize("target,expected", [("left_torque", (2, 0)), ("right_torque", (0, 2))])
def test_single_output_mapping_preserves_declared_side(target, expected):
    hub, _ = hub_and_queue()
    adapter = ModelOutputAdapter(spec(target=target))
    assert adapter.map_output(np.array([2])) == expected


def test_pipeline_bilateral_delay_and_reset_keep_independent_histories():
    model = spec(unit="Nm", mass=0)
    model.assistance.delay_s = 0.1
    adapter = ModelOutputAdapter(model)
    first = adapter.process(0, 2, 4)
    assert first["desired_left_nm"] == first["desired_right_nm"] == 0
    second = adapter.process(100_000_000, 0, 0)
    assert (second["desired_left_nm"], second["desired_right_nm"]) == (1, -2)
    adapter.reset()
    assert adapter.process(200_000_000, 0, 0)["desired_right_nm"] == 0


def test_fast_stream_leading_slow_stream_retains_cached_grid_history():
    hub, fast = hub_and_queue()
    slow = Queue()
    hub.register_stream("slow", slow, channels=("z",), sample_shape=(1,), rate_hz=10)
    model = spec(rate=20, window=0.15)
    model.safety.watchdog_s = 0.8
    from exo_collection.apps.model_runtime.spec import FeatureInput
    model.features.inputs.append(FeatureInput(modality="slow"))
    # Fast stream leads by 600 ms, which is allowed by the declared watchdog.
    # Its original cached 1.1 s sample must still be used for the slow grid.
    append(hub, fast, [[i, 100+i] for i in range(34)], rate=40)
    append(hub, slow, [[1000], [1001], [1002]], rate=10)
    adapter = ModelInputAdapter(hub, model, DefaultWindowFeatureExtractor(), clock_ns=lambda: 1_825_000_000)
    assert adapter.preflight()["ready"], adapter.preflight()
    tensor = adapter.extract()
    assert tensor.shape == (3, 3)
    assert tensor[1].tolist() == [4, 6, 8]
    assert tensor[2].tolist() == [1001, 1001, 1002]


def test_generic_watchdog_applies_to_preflight_prepare_and_delayed_result():
    hub, queue = hub_and_queue()
    append(hub, queue, [[1, 2], [3, 4], [5, 6]])
    model = spec()
    now = [1_200_000_000]
    adapter = ModelInputAdapter(hub, model, DefaultWindowFeatureExtractor(), clock_ns=lambda: now[0])
    assert adapter.preflight()["ready"]
    assert adapter.extract().size
    now[0] += 300_000_000
    assert not adapter.preflight()["ready"]
    assert adapter.extract().size == 0
    assert "watchdog" in adapter.result_expired_reason()

    # New input arriving during slow inference cannot rescue an old prepared
    # result: commit validates the source snapshot that produced that result.
    now[0] = 1_200_000_000
    class Runtime:
        def predict(self, features):
            now[0] = 1_600_000_000
            append(hub, queue, [[7, 8]], t=now[0])
            return np.array([2], dtype=np.float32)
    sink = MockTorqueOutput(5, enable_required=False)
    controller = InferenceController(hub, model, Runtime(), sink, clock_ns=lambda: now[0])
    samples = []
    controller.prediction.connect(samples.append)
    controller.set_enabled(True)
    controller._tick()
    assert not samples
    assert not sink.enabled
    assert sink.last.right_nm == 0
    assert "watchdog" in controller.last_error


def imu_model():
    from pathlib import Path
    from exo_collection.apps.model_runtime.spec import load_model_spec
    return load_model_spec(Path(__file__).resolve().parents[2] /
        "runtime_data/models/imu_right_pelvis_torque_v1_20261010/model_spec.json")


@pytest.mark.parametrize("right_sign", [1, -1])
def test_deployed_right_assistance_is_processed_once_then_copied_without_left_prediction(right_sign):
    from exo_collection.apps.model_runtime.assistance import AssistancePipeline
    model = imu_model()
    model.assistance.mass_kg = 60
    model.assistance.gain = .25
    model.assistance.lowpass_hz = 3
    model.assistance.delay_s = .1
    model.assistance.right_sign = right_sign
    adapter = ModelOutputAdapter(model)
    reference = AssistancePipeline(model.assistance)
    assert adapter.left is None
    for index, raw in enumerate([1, -2, 3, -4, 2]):
        time_ns = index * 50_000_000
        expected = reference.process(time_ns, raw, unit="Nm/kg")
        sample = adapter.process(time_ns, 999, raw)
        assert sample["desired_left_nm"] == sample["desired_right_nm"] == expected.desired_right_nm
        assert sample["estimate_nm"] == raw * 60
        assert sample["raw_estimate_left"] == sample["estimate_left_nm"] == 0
        assert sample["motor_distribution"] == "right_to_both_same_sign"
        assert "仅估计右髋" in sample["contract_description"]
        assert "同号同值" in sample["contract_description"]
    assert list(adapter.right._history) == list(reference._history)


def test_shared_assistance_without_mass_keeps_both_desired_values_unavailable():
    model = imu_model()
    adapter = ModelOutputAdapter(model)
    sample = adapter.process(1, 0, .2)
    assert not sample["assistance_ready"]
    assert sample["desired_left_nm"] is sample["desired_right_nm"] is None
    assert sample["estimate_left_nm"] == 0 and sample["raw_estimate_left"] == 0
    assert sample["raw_estimate_right"] == .2


def test_explicit_generic_shared_mode_preserves_native_estimates_and_default_stays_independent():
    model = spec(unit="Nm/kg", mass=60)
    model.assistance.motor_distribution = "right_to_both_same_sign"
    adapter = ModelOutputAdapter(model)
    sample = adapter.process(1, 2, 3)
    assert sample["raw_estimate_left"] == 2
    assert sample["estimate_left_nm"] == 120  # Real left prediction, not a copy of right.
    assert sample["estimate_nm"] == 180
    assert sample["desired_left_nm"] == sample["desired_right_nm"] == -90
    adapter.configure(model.assistance.model_copy(update={"motor_distribution": "per_side"}))
    sample = adapter.process(2, 2, 3)
    assert sample["desired_left_nm"] == 60 and sample["desired_right_nm"] == -90
    assert adapter.left is not None


def test_contract_log_separates_right_estimate_from_bilateral_motor_distribution():
    from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor
    model = imu_model()
    adapter = ModelInputAdapter(SubscriptionHub(), model, ImuTcnFeatureExtractor(model.imu_tcn))
    contract = adapter.contract()
    assert contract["output_mapping"].startswith("single->right_torque")
    assert contract["motor_distribution"] == "right_to_both_same_sign"
    assert "不产生左侧预测" in contract["contract_description"]


def test_mutated_left_only_conflict_fails_closed_at_output_boundary():
    from exo_collection.apps.model_runtime.backends import ModelRuntimeError
    model = spec(target="left_torque")
    adapter = ModelOutputAdapter(model)
    model.assistance.motor_distribution = "right_to_both_same_sign"
    with pytest.raises(ModelRuntimeError, match="left_torque"):
        adapter.process(1, 2, 0)
