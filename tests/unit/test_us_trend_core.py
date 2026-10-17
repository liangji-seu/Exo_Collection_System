import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from us_trend_core import CausalTrend, DEFAULT_CONFIG, estimate_trend_pair


def textured_profile():
    z = np.arange(957, dtype=float)
    return np.sin(z / 11.0) + 0.42 * np.sin(z / 3.7) + 0.22 * np.cos(z / 23.0)


def shifted(base, displacement):
    z = np.arange(base.size, dtype=float)
    return np.interp(z + displacement, z, base)


def test_fixed_parameters_and_constant_translation():
    base = textured_profile()
    result = estimate_trend_pair(shifted(base, 6.0), base)
    assert DEFAULT_CONFIG["window_sizes"] == (49, 99, 197)
    assert DEFAULT_CONFIG["window_weights"] == (0.2, 0.5, 0.3)
    assert DEFAULT_CONFIG["shift_min"] == -24
    assert DEFAULT_CONFIG["shift_max"] == 24
    assert np.all(result["support"][128:829])
    assert not result["support"][:128].any()
    assert abs(np.nanmedian(result["backward"][128:829]) - 6.0) < 0.3
    assert abs(np.nanmedian(result["direction"][128:829]) + 6.0) < 0.3
    assert result["screen_pass"][128:829].mean() > 0.9


def test_local_compression_has_negative_relative_spacing():
    base = textured_profile()
    z = np.arange(957, dtype=float)
    # current(z) = past(z + v(z)); positive dv/dz means compression.
    displacement = 0.02 * (z - 480.0)
    result = estimate_trend_pair(shifted(base, displacement), base)
    valid = result["screen_pass"] & (z >= 220) & (z <= 780)
    assert valid.sum() > 300
    compress_regions = [r for r in result["regions"] if r["deformation"] == "compress"]
    assert compress_regions
    assert float(np.median([r["median_spacing"] for r in compress_regions])) < 0


def test_local_expansion_has_positive_relative_spacing():
    base = textured_profile()
    z = np.arange(957, dtype=float)
    displacement = 4.0 - 0.03 * (z - 500.0)
    result = estimate_trend_pair(shifted(base, displacement), base)
    valid = result["screen_pass"] & (z >= 220) & (z <= 780)
    assert valid.sum() > 250
    assert float(np.nanmedian(result["relative_spacing"][valid])) > 0.005
    assert any(region["deformation"] == "expand" for region in result["regions"])


def test_shallow_and_deep_opposite_translation_regions_split():
    base = textured_profile()
    z = np.arange(957, dtype=float)
    displacement = np.where(z < 480.0, 6.0, -6.0)
    result = estimate_trend_pair(shifted(base, displacement), base)
    labels = {region["label"] for region in result["regions"]}
    assert "left" in labels and "right" in labels


def test_gain_only_is_screened_as_zero_motion():
    base = textured_profile()
    result = estimate_trend_pair(2.7 * base + 8.0, base)
    assert result["screen_pass"][128:829].mean() > 0.9
    assert abs(np.nanmedian(result["direction"][128:829])) < 0.3
    assert abs(np.nanmedian(result["relative_spacing"][128:829])) < 0.01


def test_flat_profile_is_unknown():
    result = estimate_trend_pair(np.ones(957), np.ones(957))
    assert not result["screen_pass"].any()
    assert any(region["label"] == "unknown" for region in result["regions"])


def test_causal_gap_reset_and_prefix_invariance():
    base = textured_profile()
    causal = CausalTrend(tau=0.15)
    first = causal.update((shifted(base, 2.0), base), 0.0)
    second = causal.update((shifted(base, 6.0), base), 0.05)
    assert first["reset"]
    assert not second["reset"]
    prefix = second["backward"][:300].copy()
    third = causal.update((shifted(base, -6.0), base), 0.10)
    # Each depth location is filtered independently; future updates cannot
    # alter the already returned prefix object.
    assert np.array_equal(prefix, second["backward"][:300])
    assert not third["reset"]
    reset = causal.update((shifted(base, 6.0), base), 0.20)
    assert reset["reset"]
    assert np.all(reset["screen_pass"] <= reset["support"])


def test_depth_mapping_remains_strictly_monotonic():
    base = textured_profile()
    result = estimate_trend_pair(shifted(base, 6.0), base)
    mapping = np.arange(957, dtype=float) + result["backward"]
    assert np.all(np.diff(mapping[128:829]) > 0)

