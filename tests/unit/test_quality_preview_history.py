from __future__ import annotations

import numpy as np

from exo_collection.domain.events import FrameBatch, SampleBatch
from exo_collection.reporting.preview_png import BoundedPreviewHistory


def _ultrasound_event(sequence: int) -> FrameBatch:
    data = np.arange(2 * 4 * 1000, dtype=np.uint16).reshape(2, 4, 1000)
    return FrameBatch(
        device_id="us",
        modality="ultrasound",
        clock_domain="us-clock",
        host_monotonic_ns=1_000_000_000 + sequence,
        first_frame_index=sequence * 2,
        frame_count=2,
        sequence_number=sequence,
        frame_rate_hz=20.0,
        data=data,
    )


def _sample_event(modality: str, sequence: int) -> SampleBatch:
    if modality == "imu":
        data = np.zeros((20, 2, 12), dtype=np.float32)
        data[:, 0, 9] = np.linspace(-10.0, 10.0, 20)
        data[:, 0, 10] = np.linspace(5.0, -5.0, 20)
    else:
        data = np.zeros((20, 6), dtype=np.float32)
        data[:, 0] = np.linspace(-0.5, 0.5, 20)
        data[:, 3] = np.linspace(0.5, -0.5, 20)
    return SampleBatch(
        device_id=modality,
        modality=modality,
        clock_domain=f"{modality}-clock",
        first_sample_index=sequence * 20,
        sample_count=20,
        sequence_number=sequence,
        sample_rate_hz=200.0,
        data=data,
    )


def test_preview_history_is_bounded_and_reports_only_soft_metrics() -> None:
    history = BoundedPreviewHistory(
        max_ultrasound_frames=3,
        ultrasound_depth_samples=64,
        max_signal_points=25,
    )
    for sequence in range(4):
        history.capture(_ultrasound_event(sequence))
        history.capture(_sample_event("imu", sequence))
        history.capture(_sample_event("encoder", sequence))

    _, ultrasound = history.ultrasound_snapshot()
    assert ultrasound.shape == (3, 4, 64)
    metrics = history.ultrasound_soft_metrics(
        formal_t0_host_monotonic_ns=1_000_000_002
    )
    assert metrics["frames_seen"] == 8
    assert metrics["frames_retained"] == 3
    assert metrics["hard_thresholds_applied"] is False
    assert metrics["metric_type"] == "soft_uncalibrated_preview_metrics"
    assert metrics["includes_pretrigger"] is False
    assert metrics["channel_count"] == 4
    assert 0.0 <= metrics["zero_fraction"] < 1.0
    assert len(metrics["channels"]) == 4

    signal_metrics = history.signal_soft_metrics()
    assert signal_metrics["hard_thresholds_applied"] is False
    assert signal_metrics["imu"]["point_count"] == 25
    assert signal_metrics["encoder"]["point_count"] == 25
