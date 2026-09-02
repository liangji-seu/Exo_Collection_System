"""Tests for the gaitway-3D field self-check verdict computation."""

from __future__ import annotations

import numpy as np

from exo_collection.adapters.force_plate.gaitway_test import (
    GaitwayTestReport,
    compute_checks,
)


def _walking_traces() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic gait: alternating single stance with a brief double-stance."""
    fz_left: list[float] = []
    fz_right: list[float] = []
    fz_total: list[float] = []
    for step in range(10):
        # right single stance
        for _ in range(10):
            fz_right.append(600.0)
            fz_left.append(20.0)
            fz_total.append(620.0)
        # double stance
        for _ in range(4):
            fz_right.append(300.0)
            fz_left.append(300.0)
            fz_total.append(600.0)
        # left single stance
        for _ in range(10):
            fz_right.append(20.0)
            fz_left.append(600.0)
            fz_total.append(620.0)
        for _ in range(4):
            fz_right.append(300.0)
            fz_left.append(300.0)
            fz_total.append(600.0)
    return (
        np.asarray(fz_total, dtype=np.float64),
        np.asarray(fz_left, dtype=np.float64),
        np.asarray(fz_right, dtype=np.float64),
    )


def test_checks_pass_for_healthy_walking_trace() -> None:
    fz_total, fz_left, fz_right = _walking_traces()
    checks = compute_checks(
        fz_total=fz_total,
        fz_left=fz_left,
        fz_right=fz_right,
        type_i_received=True,
        type_ii_received=True,
    )

    assert checks["type_i_received"]["ok"]
    assert checks["type_ii_received"]["ok"]
    assert checks["fz_left_plus_right_matches_total"]["ok"]
    assert checks["right_single_stance_detected"]["ok"]
    assert checks["left_single_stance_detected"]["ok"]
    assert checks["double_stance_detected"]["ok"]
    # magnitude agreement is tight for the synthetic trace
    assert checks["fz_left_plus_right_matches_total"]["relative_error"] < 0.05


def test_missing_type_ii_marks_type_ii_and_skips_stance_checks() -> None:
    fz_total, _, _ = _walking_traces()
    checks = compute_checks(
        fz_total=fz_total,
        fz_left=np.asarray([], dtype=np.float64),
        fz_right=np.asarray([], dtype=np.float64),
        type_i_received=True,
        type_ii_received=False,
    )

    assert checks["type_i_received"]["ok"]
    assert not checks["type_ii_received"]["ok"]
    assert "no Type II" in checks["type_ii_received"]["detail"]
    assert not checks["fz_left_plus_right_matches_total"]["ok"]
    for name in (
        "right_single_stance_detected",
        "left_single_stance_detected",
        "double_stance_detected",
    ):
        assert not checks[name]["ok"]
        assert "skipped" in checks[name]["detail"]


def test_magnitude_mismatch_is_reported_when_left_right_exceed_total() -> None:
    fz_total, fz_left, fz_right = _walking_traces()
    # Force a large disagreement by doubling the decomposed sides.
    checks = compute_checks(
        fz_total=fz_total,
        fz_left=fz_left * 2.0,
        fz_right=fz_right * 2.0,
        type_i_received=True,
        type_ii_received=True,
    )

    assert not checks["fz_left_plus_right_matches_total"]["ok"]
    assert checks["fz_left_plus_right_matches_total"]["relative_error"] > 0.5


def test_single_stance_detection_distinguishes_sides() -> None:
    # Right-only loading: right single detected, left single and double absent.
    fz_total = np.full(100, 600.0, dtype=np.float64)
    fz_right = np.full(100, 600.0, dtype=np.float64)
    fz_left = np.full(100, 10.0, dtype=np.float64)
    checks = compute_checks(
        fz_total=fz_total,
        fz_left=fz_left,
        fz_right=fz_right,
        type_i_received=True,
        type_ii_received=True,
    )

    assert checks["right_single_stance_detected"]["ok"]
    assert not checks["left_single_stance_detected"]["ok"]
    assert not checks["double_stance_detected"]["ok"]


def test_report_round_trips_to_json_ready_dict() -> None:
    report = GaitwayTestReport(
        ok=True,
        host="127.0.0.1",
        port=49500,
        sample_rate_hz=1000,
        duration_requested_s=15.0,
        duration_actual_s=14.9,
        started_at_utc="2026-09-02T00:00:00Z",
        software_version="0.1.0",
        grf_source_type="gaitway_single_platform_decomposed_left_right",
        sent_command="startDS 1000 0 0 0 2 2",
        server_response_summary="collected 14.9s + stopDS ACK",
        gaitway_settings_version=5,
        settings_packet_hex="0000",
        type_i_received=True,
        type_ii_received=True,
        type_i_packet_count=10,
        type_ii_packet_count=10,
        type_i_sample_count=1000,
        type_ii_sample_count=100,
        fz_total_min=10.0,
        fz_total_max=700.0,
        fz_left_min=10.0,
        fz_left_max=600.0,
        fz_right_min=10.0,
        fz_right_max=600.0,
        checks={"type_i_received": {"ok": True, "detail": "x"}},
        errors=[],
    )
    data = report.to_dict()
    assert data["ok"] is True
    assert data["grf_source_type"] == "gaitway_single_platform_decomposed_left_right"
    assert data["fz_total_max"] == 700.0
