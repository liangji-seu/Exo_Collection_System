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


def _raw_ultrasound_packet(channel: int, ordinal: int, value: int) -> FrameBatch:
    return FrameBatch(
        device_id="raw-us",
        modality="ultrasound",
        clock_domain="host_monotonic",
        host_monotonic_ns=2_000_000_000 + ordinal * 10 + channel,
        first_frame_index=ordinal * 4 + channel,
        frame_count=1,
        sequence_number=ordinal * 4 + channel,
        frame_rate_hz=20.0,
        data=np.full((1, 1000), value, dtype=np.uint8),
        channel=channel,
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


def test_raw_ultrasound_history_aligns_only_complete_channel_ordinals() -> None:
    history = BoundedPreviewHistory(
        max_ultrasound_frames=3,
        ultrasound_depth_samples=50,
    )
    for ordinal in range(4):
        for channel in range(4):
            history.capture(
                _raw_ultrasound_packet(channel, ordinal, 10 * channel + ordinal)
            )

    timestamps, frames = history.ultrasound_snapshot()
    assert timestamps.tolist() == [2_000_000_013, 2_000_000_023, 2_000_000_033]
    assert frames.shape == (3, 4, 50)
    for retained_ordinal, original_ordinal in enumerate((1, 2, 3)):
        for channel in range(4):
            assert np.all(
                frames[retained_ordinal, channel] == 10 * channel + original_ordinal
            )

    metrics = history.ultrasound_soft_metrics(
        formal_t0_host_monotonic_ns=2_000_000_015
    )
    assert metrics["frames_seen"] == 16
    assert metrics["frames_retained"] == 3
    assert metrics["channel_count"] == 4
    assert metrics["packets_retained_per_channel"] == [3, 3, 3, 3]
    assert metrics["includes_pretrigger"] is True
    assert metrics["device_synchronized_frames"] is False
    assert metrics["alignment_semantics"] == (
        "independent_channel_arrival_ordinal_for_qc_preview_only"
    )
    assert metrics["timestamp_semantics"] == (
        "maximum_host_arrival_timestamp_per_qc_ordinal_row"
    )


def test_raw_ultrasound_history_never_zero_fills_missing_channel() -> None:
    history = BoundedPreviewHistory(ultrasound_depth_samples=40)
    for channel in range(3):
        history.capture(_raw_ultrasound_packet(channel, 0, channel + 1))

    timestamps, frames = history.ultrasound_snapshot()
    assert timestamps.shape == (0,)
    assert frames.shape == (0, 4, 40)
    metrics = history.ultrasound_soft_metrics()
    assert metrics["frames_seen"] == 3
    assert metrics["frames_retained"] == 0
    assert metrics["channel_count"] == 0
    assert metrics["packets_retained_per_channel"] == [1, 1, 1, 0]
    assert metrics["device_synchronized_frames"] is False


def test_traditional_ultrasound_frames_keep_device_alignment_semantics() -> None:
    history = BoundedPreviewHistory(ultrasound_depth_samples=32)
    history.capture(_ultrasound_event(0))

    _, frames = history.ultrasound_snapshot()
    assert frames.shape == (2, 4, 32)
    metrics = history.ultrasound_soft_metrics()
    assert metrics["device_synchronized_frames"] is True
    assert metrics["alignment_semantics"] == (
        "device_synchronized_multichannel_frame"
    )
    assert "packets_retained_per_channel" not in metrics
