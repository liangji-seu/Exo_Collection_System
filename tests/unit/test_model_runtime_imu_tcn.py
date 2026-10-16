from __future__ import annotations

from queue import Queue

import numpy as np

from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor
from exo_collection.apps.model_runtime.spec import FeatureSpec, ImuTcnSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.domain.events import SampleBatch


def _setup(
    clock_ns=2_000_000_000,
    *,
    control_rate_hz=20.0,
    window_s=2.0,
    sample_count=201,
    sample_rate_hz=100.0,
    descriptor_rate_hz=None,
    include_queue=False,
):
    hub = SubscriptionHub()
    q: Queue = Queue()
    channels = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z", "roll", "pitch", "yaw")
    hub.register_stream(
        "imu", q, channels=channels, sample_shape=(2, 12), rate_hz=sample_rate_hz,
        descriptor={"channels": channels, "units": ("m/s2",) * 3 + ("rad/s",) * 3 + ("",) * 6,
                    "sample_shape": (2, 12), "metadata": {"device_ids": ["right", "pelvis"]}},
    )
    descriptor = hub.descriptor("imu")
    if descriptor_rate_hz is not None:
        descriptor["nominal_rate_hz"] = descriptor_rate_hz
        hub._streams["imu"].descriptor = descriptor
    for index in range(sample_count):
        t_ns = int(index * (1_000_000_000 / sample_rate_hz))
        for slot in (0, 1):
            data = np.full((1, 2, 12), np.nan, dtype=np.float32)
            data[0, slot, :6] = index + slot
            q.put(SampleBatch(device_id="imu", modality="imu", clock_domain="imu",
                              first_sample_index=index, sample_count=1, sequence_number=index,
                              sample_rate_hz=sample_rate_hz, host_monotonic_ns=t_ns, data=data))
    hub.drain()
    spec = FeatureSpec.model_validate({"control_rate_hz": control_rate_hz, "window_s": window_s,
                                       "inputs": [{"modality": "imu"}]})
    cfg = ImuTcnSpec(enabled=True, right_device_id="right", pelvis_device_id="pelvis")
    clock = clock_ns if callable(clock_ns) else (lambda: clock_ns)
    result = (hub, spec, ImuTcnFeatureExtractor(cfg, clock_ns=clock))
    return (*result, q) if include_queue else result


def test_causal_tcn_window_uses_two_device_rows_and_only_once_per_grid():
    hub, spec, extractor = _setup()
    features = extractor.extract(hub, spec)
    assert features.shape == (12, 40)
    assert features[:, -1].tolist() == [200.0] * 6 + [201.0] * 6
    assert extractor.extract(hub, spec).size == 0
    assert extractor.last_reason == "no new source grid"


def test_stale_device_is_rejected_without_reusing_old_latest_as_now():
    hub, spec, extractor = _setup(clock_ns=2_100_000_000)
    assert extractor.extract(hub, spec).size == 0
    assert "stale" in (extractor.last_reason or "")


def test_new_tcn_window_is_100hz_and_94_frames():
    hub, spec, extractor = _setup(control_rate_hz=100.0, window_s=0.94)
    features = extractor.extract(hub, spec)
    assert features.shape == (12, 94)
    assert extractor.last_result is not None
    assert extractor.last_result.grid_end_ns == 2_000_000_000
    assert extractor.last_result.right_source_times_ns[0] == 1_070_000_000
    assert features[:, -1].tolist() == [200.0] * 6 + [201.0] * 6


def test_100hz_tcn_advances_once_per_new_10ms_grid():
    clock = [2_000_000_000]
    hub, spec, extractor, queue = _setup(
        clock_ns=lambda: clock[0], control_rate_hz=100.0, window_s=0.94,
        include_queue=True,
    )
    assert extractor.extract(hub, spec).shape == (12, 94)
    assert extractor.extract(hub, spec).size == 0
    for slot in (0, 1):
        data = np.full((1, 2, 12), np.nan, dtype=np.float32)
        data[0, slot, :6] = 201.0 + slot
        queue.put(SampleBatch(
            device_id="imu", modality="imu", clock_domain="imu",
            first_sample_index=201, sample_count=1, sequence_number=201,
            sample_rate_hz=100.0, host_monotonic_ns=2_010_000_000, data=data,
        ))
    hub.drain()
    clock[0] = 2_010_000_000
    features = extractor.extract(hub, spec)
    assert features.shape == (12, 94)
    assert features[:, -1].tolist() == [201.0] * 6 + [202.0] * 6


def test_94_frame_window_warms_up_until_history_is_complete():
    hub, spec, extractor = _setup(
        clock_ns=920_000_000, control_rate_hz=100.0, window_s=0.94,
        sample_count=93,
    )
    assert extractor.extract(hub, spec).size == 0
    assert extractor.waiting_for_valid_window
    assert "warmup" in (extractor.last_reason or "")


def test_declared_raw_rate_below_model_rate_is_rejected():
    hub, spec, extractor = _setup(
        control_rate_hz=100.0, window_s=0.94, descriptor_rate_hz=20.0,
    )
    status = extractor.preflight(hub, spec)
    assert not status["ready"]
    assert any("below model control rate" in error for error in status["errors"])
    assert extractor.extract(hub, spec).size == 0
    assert "below model control rate" in (extractor.last_reason or "")


def test_real_descriptor_rate_wins_over_model_input_rate_override():
    hub, _spec, extractor = _setup(
        control_rate_hz=100.0, window_s=0.94, sample_rate_hz=20.0,
        descriptor_rate_hz=20.0,
    )
    spec = FeatureSpec.model_validate({
        "control_rate_hz": 100.0,
        "window_s": 0.94,
        "inputs": [{"modality": "imu", "rate_hz": 100.0}],
    })
    status = extractor.preflight(hub, spec)
    assert not status["ready"]
    assert any("20 Hz" in error for error in status["errors"])
    assert extractor.extract(hub, spec).size == 0
    assert "20 Hz" in (extractor.last_reason or "")


def test_source_rate_above_model_rate_is_allowed():
    hub, spec, extractor = _setup(
        control_rate_hz=100.0, window_s=0.94, sample_count=241,
        sample_rate_hz=120.0, descriptor_rate_hz=120.0,
    )
    status = extractor.preflight(hub, spec)
    assert status["ready"], status
    assert extractor.extract(hub, spec).shape == (12, 94)


def test_standalone_imu_extractor_rejects_non_integer_window():
    hub, _spec, extractor = _setup(control_rate_hz=100.0, window_s=0.94)
    spec = FeatureSpec.model_validate({
        "control_rate_hz": 100.0,
        "window_s": 0.935,
        "inputs": [{"modality": "imu"}],
    })
    status = extractor.preflight(hub, spec)
    assert not status["ready"]
    assert any("integer positive" in error for error in status["errors"])
    assert extractor.extract(hub, spec).size == 0
