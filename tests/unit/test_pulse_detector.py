from __future__ import annotations

import numpy as np
import pytest

from exo_collection.timing import PulseDetector, PulseDetectorConfig


def test_hysteresis_rejects_short_spike_and_reports_valid_pulse() -> None:
    detector = PulseDetector(
        PulseDetectorConfig(
            high_threshold=2.0,
            low_threshold=1.0,
            min_pulse_width_ns=2_000_000,
            debounce_ns=1_000_000,
        )
    )
    # The first one-sample spike is rejected. The second pulse is valid.
    values = np.asarray([0, 2.4, 0, 0, 2.2, 1.5, 2.5, 2.3, 0.5, 0.4, 0.0])
    events = detector.process(
        values,
        first_sample_index=100,
        host_monotonic_ns=10_000,
        sample_rate_hz=1000,
    )
    assert [event.edge_type.value for event in events] == ["rising", "falling"]
    assert events[0].sample_index == 104
    assert events[1].sample_index == 108
    assert events[1].pulse_width_ns == 4_000_000
    assert events[0].pulse_id == events[1].pulse_id
    assert events[1].amplitude == pytest.approx(2.5)


def test_detector_state_crosses_chunk_boundaries() -> None:
    detector = PulseDetector(
        PulseDetectorConfig(
            high_threshold=2.0,
            low_threshold=1.0,
            min_pulse_width_ns=2_000_000,
            debounce_ns=1_000_000,
        ),
        source_device="daq_1",
    )
    first = detector.process(
        [0.0, 2.2, 2.3],
        first_sample_index=0,
        host_monotonic_ns=1_000,
        sample_rate_hz=1000,
    )
    second = detector.process(
        [2.4, 1.4, 0.2, 0.1],
        first_sample_index=3,
        host_monotonic_ns=3_001_000,
        sample_rate_hz=1000,
    )
    events = first + second
    assert [event.edge_type.value for event in events] == ["rising", "falling"]
    assert events[0].sample_index == 1
    assert events[1].sample_index == 5
    assert events[1].pulse_width_ns == 4_000_000


def test_brief_low_excursion_is_debounced_without_splitting_pulse() -> None:
    detector = PulseDetector(
        PulseDetectorConfig(
            high_threshold=2.0,
            low_threshold=1.0,
            min_pulse_width_ns=0,
            debounce_ns=2_000_000,
        )
    )
    events = detector.process(
        [0, 3, 3, 0.2, 3, 3, 0.2, 0.1, 0.0],
        first_sample_index=0,
        host_monotonic_ns=0,
        sample_rate_hz=1000,
    )
    assert [event.edge_type.value for event in events] == ["rising", "falling"]
    assert events[0].sample_index == 1
    assert events[1].sample_index == 6
    assert detector.pulse_count == 1


def test_pulse_equal_to_minimum_width_is_accepted_at_falling_edge() -> None:
    detector = PulseDetector(
        PulseDetectorConfig(
            high_threshold=2.0,
            low_threshold=1.0,
            min_pulse_width_ns=2_000_000,
            debounce_ns=0,
        )
    )
    events = detector.process(
        [3.0, 3.0, 0.0],
        first_sample_index=0,
        host_monotonic_ns=0,
        sample_rate_hz=1000,
    )
    assert [event.edge_type.value for event in events] == ["rising", "falling"]
    assert events[1].pulse_width_ns == 2_000_000


def test_detector_validates_thresholds_and_monotonic_chunks() -> None:
    with pytest.raises(ValueError, match="less than"):
        PulseDetectorConfig(high_threshold=1.0, low_threshold=1.0)
    detector = PulseDetector()
    detector.process([0.0, 0.0], first_sample_index=2, host_monotonic_ns=10, sample_rate_hz=10)
    with pytest.raises(ValueError, match="indices"):
        detector.process([0.0], first_sample_index=2, host_monotonic_ns=200_000_011, sample_rate_hz=10)
    with pytest.raises(ValueError, match="timestamp array"):
        PulseDetector().process(
            [0.0, 0.0],
            first_sample_index=0,
            host_monotonic_ns=np.asarray([1]),
        )
