from __future__ import annotations

import numpy as np
import pytest

from exo_collection.timing.alignment import align_shared_pulses
from exo_collection.timing.clock_model import DeviceClockMapper, fit_affine_clock


def test_multiple_pulses_fit_offset_and_drift() -> None:
    source = np.array([0, 1_000_000, 2_000_000, 4_000_000], dtype=np.int64)
    target = 1.000125 * source + 9_000_000_000
    model = fit_affine_clock(source, target)
    assert model.scale_a == pytest.approx(1.000125)
    assert model.offset_b_ns == pytest.approx(9_000_000_000)
    assert model.residuals.max_absolute_ns < 1e-3


def test_one_pulse_estimates_only_offset() -> None:
    model = fit_affine_clock([125], [1_000_125])
    assert model.scale_a == 1.0
    assert model.offset_b_ns == 1_000_000
    assert model.algorithm_version.startswith("single-anchor")


def test_shared_pulse_pairing_ignores_unmatched_events() -> None:
    model, pulse_ids = align_shared_pulses(
        [("p0", 0), ("p1", 1000), ("external-only", 2000)],
        [("p0", 10_000), ("p1", 11_001), ("host-only", 12_000)],
    )
    assert pulse_ids == ("p0", "p1")
    assert model.map_one(1000) == pytest.approx(11_001)


def test_clock_fit_rejects_non_monotonic_anchors() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_affine_clock([0, 2, 1], [100, 102, 103])


# ──────────────────────────────────────────────────────────────
#  DeviceClockMapper (IMU host-timestamp reconstruction)
# ──────────────────────────────────────────────────────────────


def test_device_clock_mapper_first_sample_returns_arrival_exactly() -> None:
    mapper = DeviceClockMapper(period_ns=5_000_000)  # 200 Hz
    assert mapper.map(raw_counter=7, arrival_ns=10) == 10


def test_device_clock_mapper_uniform_spacing_after_warmup_despite_bursty_arrival() -> None:
    period = 5_000_000
    mapper = DeviceClockMapper(period_ns=period, warmup=16)
    # Bursty arrivals with the correct mean rate: bursts of 3 samples arrive
    # every 3 periods (15 ms), so within-burst spacing is ~0, not 5 ms.
    count = 40
    arrivals = [(index // 3) * 3 * period for index in range(count)]
    mapped = [mapper.map(counter, arrival) for counter, arrival in enumerate(arrivals)]
    assert mapped[0] == arrivals[0]
    # Once the offset is frozen, spacing is exactly one nominal period.
    assert np.all(np.diff(mapped[16:]) == period)


def test_device_clock_mapper_frozen_offset_equals_mean_latency() -> None:
    period = 5_000_000
    latency = 2_000_000  # constant 2 ms transport latency
    mapper = DeviceClockMapper(period_ns=period, warmup=4)
    mapped = [mapper.map(index, index * period + latency) for index in range(10)]
    assert mapped[-1] == 9 * period + latency


def test_device_clock_mapper_unwraps_uint16_counter() -> None:
    period = 5_000_000
    mapper = DeviceClockMapper(period_ns=period)
    # 65534, 65535, then wrap to 0, 1 — must stay monotonic and uniform.
    base = 65_534 * period
    mapped = [
        mapper.map(65534, base),
        mapper.map(65535, base + period),
        mapper.map(0, base + 2 * period),
        mapper.map(1, base + 3 * period),
    ]
    assert np.all(np.diff(mapped) == period)

