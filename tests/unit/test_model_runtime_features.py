"""Unit tests for the default feature extractor and hold-resample helper."""

from __future__ import annotations

from queue import Queue

import numpy as np
import pytest

from exo_collection.apps.model_runtime.features import (
    DefaultWindowFeatureExtractor,
    FeatureExtractionError,
    hold_resample,
)
from exo_collection.apps.model_runtime.spec import FeatureSpec, ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.domain.events import SampleBatch


def _batch(modality: str, device_id: str, data: np.ndarray, t_ns: int) -> SampleBatch:
    return SampleBatch(
        device_id=device_id,
        modality=modality,
        clock_domain=f"{modality}_clock",
        first_sample_index=0,
        sample_count=int(data.shape[0]),
        sequence_number=0,
        sample_rate_hz=100.0,
        host_monotonic_ns=t_ns,
        data=data.astype(np.float32),
    )


def _two_modality_hub() -> tuple[SubscriptionHub, Queue, Queue]:
    hub = SubscriptionHub()
    imu_q: Queue = Queue()
    enc_q: Queue = Queue()
    hub.register_stream("imu", imu_q, channels=("x", "y"), sample_shape=(2,), rate_hz=100.0)
    hub.register_stream("enc", enc_q, channels=("a", "b"), sample_shape=(2,), rate_hz=100.0)
    return hub, imu_q, enc_q


def _spec(window_s: float = 0.03, inputs=None) -> FeatureSpec:
    inputs = inputs or [
        {"modality": "imu", "channels": None},
        {"modality": "enc", "channels": None},
    ]
    return ModelSpec.model_validate(
        {
            "schema_version": "1.0.0",
            "model": {"backend": "demo"},
            "features": {"control_rate_hz": 100.0, "window_s": window_s, "inputs": inputs},
            "output": {"kind": "torque_left_right"},
        }
    ).features


# -- hold_resample ---------------------------------------------------------


def test_hold_resample_basic() -> None:
    times = np.array([0, 10, 20], dtype=np.int64)
    values = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    grid = np.array([0, 5, 15, 25], dtype=np.int64)
    out = hold_resample(times, values, grid)
    assert out[:, 0].tolist() == [0.0, 0.0, 10.0, 20.0]


def test_hold_resample_empty() -> None:
    out = hold_resample(np.array([], dtype=np.int64), np.zeros((0, 2), dtype=np.float32), np.array([1, 2, 3]))
    assert out.shape == (3, 2)
    assert np.all(out == 0)


# -- extractor -------------------------------------------------------------


def test_extractor_builds_channel_time_tensor() -> None:
    hub, imu_q, enc_q = _two_modality_hub()
    imu_q.put(_batch("imu", "i", np.array([[0, 10], [1, 11], [2, 12]], dtype=np.float32), 0))
    enc_q.put(_batch("enc", "e", np.array([[100, 110], [101, 111], [102, 112]], dtype=np.float32), 0))
    hub.drain()

    tensor = DefaultWindowFeatureExtractor().extract(hub, _spec())
    assert tensor.shape == (4, 3)
    assert tensor[0].tolist() == [0.0, 1.0, 2.0]     # imu x
    assert tensor[1].tolist() == [10.0, 11.0, 12.0]  # imu y
    assert tensor[2].tolist() == [100.0, 101.0, 102.0]  # enc a
    assert tensor[3].tolist() == [110.0, 111.0, 112.0]  # enc b


def test_extractor_channel_selection() -> None:
    hub, imu_q, enc_q = _two_modality_hub()
    imu_q.put(_batch("imu", "i", np.array([[0, 10], [1, 11], [2, 12]], dtype=np.float32), 0))
    enc_q.put(_batch("enc", "e", np.array([[100, 110], [101, 111], [102, 112]], dtype=np.float32), 0))
    hub.drain()

    spec = _spec(
        inputs=[
            {"modality": "imu", "channels": ["y"]},
            {"modality": "enc", "channels": ["b"]},
        ]
    )
    tensor = DefaultWindowFeatureExtractor().extract(hub, spec)
    assert tensor.shape == (2, 3)
    assert tensor[0].tolist() == [10.0, 11.0, 12.0]
    assert tensor[1].tolist() == [110.0, 111.0, 112.0]


def test_extractor_normalization() -> None:
    hub, imu_q, enc_q = _two_modality_hub()
    imu_q.put(_batch("imu", "i", np.array([[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]], dtype=np.float32), 0))
    enc_q.put(_batch("enc", "e", np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32), 0))
    hub.drain()

    spec = ModelSpec.model_validate(
        {
            "schema_version": "1.0.0",
            "model": {"backend": "demo"},
            "features": {
                "control_rate_hz": 100.0,
                "window_s": 0.03,
                "inputs": [
                    {"modality": "imu", "channels": None},
                    {"modality": "enc", "channels": None},
                ],
                "normalization": {"mean": [2.0, 20.0, 0.0, 0.0], "std": [2.0, 20.0, 1.0, 1.0]},
            },
            "output": {"kind": "torque_left_right"},
        }
    ).features
    tensor = DefaultWindowFeatureExtractor().extract(hub, spec)
    assert tensor[0].tolist() == [0.0, 1.0, 2.0]
    assert tensor[1].tolist() == [0.0, 1.0, 2.0]


def test_extractor_insufficient_data_returns_empty() -> None:
    hub, imu_q, _enc_q = _two_modality_hub()
    # imu has data, enc is empty → empty tensor.
    imu_q.put(_batch("imu", "i", np.array([[1.0, 2.0]], dtype=np.float32), 0))
    hub.drain()
    tensor = DefaultWindowFeatureExtractor().extract(hub, _spec())
    assert tensor.shape == (0, 0)


def test_extractor_missing_channel_raises() -> None:
    hub, imu_q, enc_q = _two_modality_hub()
    imu_q.put(_batch("imu", "i", np.array([[1.0, 2.0]], dtype=np.float32), 0))
    enc_q.put(_batch("enc", "e", np.array([[3.0, 4.0]], dtype=np.float32), 0))
    hub.drain()
    spec = _spec(inputs=[{"modality": "imu", "channels": ["does_not_exist"]}])
    with pytest.raises(FeatureExtractionError):
        DefaultWindowFeatureExtractor().extract(hub, spec)


def test_expected_channel_count() -> None:
    hub, _imu_q, _enc_q = _two_modality_hub()
    extractor = DefaultWindowFeatureExtractor()
    assert extractor.expected_channel_count(hub, _spec()) == 4
