from __future__ import annotations

import numpy as np
import pytest

from exo_collection.timing.alignment import align_shared_pulses
from exo_collection.timing.clock_model import fit_affine_clock


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

