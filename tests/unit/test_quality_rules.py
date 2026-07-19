from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from exo_collection.domain.models import QualityGrade
from exo_collection.quality import (
    ClockMappingEvidence,
    DiskSpaceEvidence,
    InsufficientDiskSpaceError,
    SignalEvidence,
    SyncEdgeEvidence,
    TrialQualityEvidence,
    UltrasoundEvidence,
    check_disk_space,
    evaluate_trial_quality,
    load_quality_rules,
    load_storage_policy,
    scan_hdf5_signal_evidence,
)
from exo_collection.quality.config import QualityRulesDocument
from exo_collection.quality.engine import RuleStatus


def clean_evidence(**updates) -> TrialQualityEvidence:
    payload = {
        "formal_duration_s": 1.0,
        "formal_item_counts": {
            "ultrasound": 20,
            "imu": 200,
            "encoder": 100,
            "sync_pulse": 1000,
        },
        "sequence_gap_counts": {
            "ultrasound": 0,
            "imu": 0,
            "encoder": 0,
            "sync_pulse": 0,
        },
        "dropped_batch_counts": {
            "ultrasound": 0,
            "imu": 0,
            "encoder": 0,
            "sync_pulse": 0,
        },
        "sync_edges": (
            SyncEdgeEvidence(
                pulse_id="p1",
                edge_type="rising",
                host_monotonic_ns=100,
            ),
            SyncEdgeEvidence(
                pulse_id="p1",
                edge_type="falling",
                host_monotonic_ns=20_000_100,
                pulse_width_ns=20_000_000,
            ),
        ),
        "first_trigger_host_monotonic_ns": 100,
        "clock_mappings": tuple(
            ClockMappingEvidence(
                modality=modality,
                anchor_count=10,
                rms_residual_ns=100.0,
            )
            for modality in ("ultrasound", "imu", "encoder", "sync_pulse")
        ),
        "ultrasound": UltrasoundEvidence(
            formal_frame_count=20,
            zero_fraction=0.01,
            saturation_fraction=0.0,
        ),
        "signals": {
            "imu": SignalEvidence(
                formal_sample_count=200,
                nonfinite_value_count=0,
                minimum=[-1.0, -2.0],
                maximum=[1.0, 2.0],
                maximum_absolute_jump=0.1,
            ),
            "encoder": SignalEvidence(
                formal_sample_count=100,
                nonfinite_value_count=0,
                minimum=[-0.5, -0.25],
                maximum=[0.5, 0.25],
                maximum_absolute_jump=0.05,
            ),
        },
        "disk_space": DiskSpaceEvidence(
            path="C:/data",
            free_bytes=10 * 1024**3,
            required_free_bytes=2 * 1024**3,
        ),
    }
    payload.update(updates)
    return TrialQualityEvidence.model_validate(payload)


def test_default_quality_rules_are_strict_versioned_and_uncalibrated() -> None:
    rules = load_quality_rules()
    assert rules.algorithm_version == "exo-quality-rules-1.0.0"
    assert rules.required_modalities == (
        "ultrasound",
        "imu",
        "encoder",
        "sync_pulse",
    )
    assert rules.ultrasound.saturation_fraction_warning is None
    assert rules.imu.maximum_absolute_jump is None

    payload = rules.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        QualityRulesDocument.model_validate(payload)
    payload = rules.model_dump(mode="json")
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError, match="literal_error"):
        QualityRulesDocument.model_validate(payload)


def test_invalid_ranges_and_unreferenced_calibration_are_rejected() -> None:
    payload = load_quality_rules().model_dump(mode="json")
    payload["sync"]["pulse_width_ns"] = {"minimum": 20, "maximum": 10}
    with pytest.raises(ValidationError, match="maximum must be"):
        QualityRulesDocument.model_validate(payload)

    payload = load_quality_rules().model_dump(mode="json")
    payload["encoder"]["maximum_absolute_jump"] = 2.0
    with pytest.raises(ValidationError, match="calibration_reference"):
        QualityRulesDocument.model_validate(payload)

    payload = load_quality_rules().model_dump(mode="json")
    payload["ultrasound"]["saturation_fraction_warning"] = 0.1
    with pytest.raises(ValidationError, match="calibration_reference"):
        QualityRulesDocument.model_validate(payload)


def test_clean_structural_evidence_can_be_a_with_calibration_rules_unassessed() -> None:
    evaluation = evaluate_trial_quality(clean_evidence(), load_quality_rules())
    assert evaluation.grade is QualityGrade.A
    assert not evaluation.issues
    assert evaluation.unassessed_count > 0
    unassessed_codes = {
        result.code
        for result in evaluation.results
        if result.status is RuleStatus.UNASSESSED
    }
    assert {
        "SYNC_PULSE_INTERVAL",
        "CLOCK_MAPPING_RESIDUAL",
        "ULTRASOUND_SATURATION",
        "SIGNAL_CALIBRATED_RANGE",
        "SIGNAL_ABSOLUTE_JUMP",
    } <= unassessed_codes
    required = [result for result in evaluation.results if result.required_for_grade_a]
    assert required
    assert all(result.status is RuleStatus.PASS for result in required)


def test_required_modality_absence_is_invalid_and_gap_is_c() -> None:
    counts = dict(clean_evidence().formal_item_counts)
    counts["imu"] = 0
    missing = evaluate_trial_quality(
        clean_evidence(formal_item_counts=counts), load_quality_rules()
    )
    assert missing.grade is QualityGrade.INVALID
    assert any(
        issue.code == "REQUIRED_MODALITY_FORMAL_DATA" and issue.modality == "imu"
        for issue in missing.issues
    )

    gaps = dict(clean_evidence().sequence_gap_counts)
    gaps["encoder"] = 1
    discontinuous = evaluate_trial_quality(
        clean_evidence(sequence_gap_counts=gaps), load_quality_rules()
    )
    assert discontinuous.grade is QualityGrade.C
    assert any(issue.code == "SEQUENCE_CONTINUITY" for issue in discontinuous.issues)


def test_optional_sync_does_not_hide_modality_loss() -> None:
    optional_sync = evaluate_trial_quality(
        clean_evidence(
            sync_edges=(),
            first_trigger_host_monotonic_ns=None,
            synchronization_required=False,
            clock_mappings=(),
        ),
        load_quality_rules(),
    )

    assert optional_sync.grade is QualityGrade.A
    assert not optional_sync.issues
    optional_codes = {
        result.code
        for result in optional_sync.results
        if result.status is RuleStatus.UNASSESSED
    }
    assert {
        "SYNC_RISING_EDGE_COUNT",
        "FIRST_SYNC_TRIGGER",
        "SYNC_COMPLETE_PULSE_COUNT",
        "SYNC_PULSE_WIDTH",
        "SYNC_PULSE_INTERVAL",
        "CLOCK_MAPPING_ANCHORS",
        "CLOCK_MAPPING_RESIDUAL",
    } <= optional_codes

    optional_rising_only = evaluate_trial_quality(
        clean_evidence(
            sync_edges=(
                SyncEdgeEvidence(
                    pulse_id="optional-pulse",
                    edge_type="rising",
                    host_monotonic_ns=100,
                ),
            ),
            first_trigger_host_monotonic_ns=100,
            synchronization_required=False,
            clock_mappings=(),
        ),
        load_quality_rules(),
    )
    assert optional_rising_only.grade is QualityGrade.A
    assert not optional_rising_only.issues

    counts = dict(clean_evidence().formal_item_counts)
    counts["imu"] = 0
    missing_modality = evaluate_trial_quality(
        clean_evidence(
            formal_item_counts=counts,
            sync_edges=(),
            first_trigger_host_monotonic_ns=None,
            synchronization_required=False,
            clock_mappings=(),
        ),
        load_quality_rules(),
    )

    assert missing_modality.grade is QualityGrade.INVALID
    assert any(
        issue.code == "REQUIRED_MODALITY_FORMAL_DATA" and issue.modality == "imu"
        for issue in missing_modality.issues
    )


def test_nonfinite_constant_and_all_zero_rules_are_not_silent_a() -> None:
    signals = dict(clean_evidence().signals)
    signals["imu"] = SignalEvidence(
        formal_sample_count=200,
        nonfinite_value_count=1,
        minimum=[0.0],
        maximum=[1.0],
        maximum_absolute_jump=1.0,
    )
    nonfinite = evaluate_trial_quality(
        clean_evidence(signals=signals), load_quality_rules()
    )
    assert nonfinite.grade is QualityGrade.C
    assert any(issue.code == "SIGNAL_NONFINITE_VALUES" for issue in nonfinite.issues)

    signals = dict(clean_evidence().signals)
    signals["encoder"] = SignalEvidence(
        formal_sample_count=100,
        nonfinite_value_count=0,
        minimum=[3.0, 4.0],
        maximum=[3.0, 4.0],
        maximum_absolute_jump=0.0,
    )
    constant = evaluate_trial_quality(
        clean_evidence(signals=signals), load_quality_rules()
    )
    assert constant.grade is QualityGrade.B
    assert any(issue.code == "SIGNAL_CONSTANT" for issue in constant.issues)

    all_zero = evaluate_trial_quality(
        clean_evidence(
            ultrasound=UltrasoundEvidence(
                formal_frame_count=20,
                zero_fraction=1.0,
                saturation_fraction=0.0,
            )
        ),
        load_quality_rules(),
    )
    assert all_zero.grade is QualityGrade.B
    assert any(issue.code == "ULTRASOUND_ALL_ZERO" for issue in all_zero.issues)


def test_configured_sync_and_signal_thresholds_are_evaluated() -> None:
    payload = load_quality_rules().model_dump(mode="json")
    payload["sync"]["pulse_width_ns"] = {
        "minimum": 10_000_000,
        "maximum": 15_000_000,
    }
    payload["sync"]["pulse_interval_ns"] = {
        "minimum": 900_000_000,
        "maximum": 1_100_000_000,
    }
    payload["sync"]["maximum_mapping_rms_residual_ns"] = 50.0
    payload["imu"].update(
        {
            "calibrated_minimum": -0.5,
            "calibrated_maximum": 0.5,
            "maximum_absolute_jump": 0.05,
            "calibration_reference": "test-fixture-2026-01",
            "calibrated_violation_severity": "WARNING",
        }
    )
    rules = QualityRulesDocument.model_validate(payload)
    evaluation = evaluate_trial_quality(clean_evidence(), rules)
    assert evaluation.grade is QualityGrade.C
    assert any(issue.code == "SYNC_PULSE_WIDTH" for issue in evaluation.issues)
    assert any(issue.code == "CLOCK_MAPPING_RESIDUAL" for issue in evaluation.issues)
    assert any(
        issue.code in {"SIGNAL_CALIBRATED_RANGE", "SIGNAL_ABSOLUTE_JUMP"}
        and issue.severity.value == "WARNING"
        for issue in evaluation.issues
    )

    two_pulses = clean_evidence(
        sync_edges=(
            *clean_evidence().sync_edges,
            SyncEdgeEvidence(
                pulse_id="p2",
                edge_type="rising",
                host_monotonic_ns=2_000_000_100,
            ),
            SyncEdgeEvidence(
                pulse_id="p2",
                edge_type="falling",
                host_monotonic_ns=2_020_000_100,
                pulse_width_ns=20_000_000,
            ),
        )
    )
    interval_evaluation = evaluate_trial_quality(two_pulses, rules)
    interval_result = next(
        result
        for result in interval_evaluation.results
        if result.code == "SYNC_PULSE_INTERVAL"
    )
    assert interval_result.status is RuleStatus.FAIL
    assert interval_result.observed_value == [2_000_000_000.0]


def test_disk_space_preflight_uses_strict_storage_policy(monkeypatch, tmp_path) -> None:
    policy = load_storage_policy()
    monkeypatch.setattr(
        "exo_collection.quality.engine.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10, used=9, free=1),
    )
    with pytest.raises(InsufficientDiskSpaceError) as captured:
        check_disk_space(tmp_path, policy)
    assert captured.value.evidence.free_bytes == 1
    assert captured.value.evidence.required_free_bytes == 2 * 1024**3

    payload = policy.model_dump(mode="json")
    payload["unknown"] = True
    path = tmp_path / "invalid-storage.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_storage_policy(path)


def test_hdf5_signal_scan_uses_only_formal_window_and_detects_nonfinite(tmp_path) -> None:
    path = tmp_path / "signal.h5"
    with h5py.File(path, "w") as file:
        samples = file.create_group("samples")
        samples.create_dataset(
            "host_monotonic_ns",
            data=np.asarray([10, 20, 30, 40, 50], dtype=np.uint64),
        )
        samples.create_dataset(
            "data",
            data=np.asarray(
                [
                    [100.0, 100.0],
                    [1.0, 2.0],
                    [2.0, np.nan],
                    [4.0, 8.0],
                    [200.0, 200.0],
                ],
                dtype=np.float32,
            ),
            chunks=(2, 2),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        discontinuity_dtype = np.dtype(
            [
                ("sample_index", np.uint64),
                ("host_monotonic_ns", np.uint64),
                ("kind", string_dtype),
                ("details_json", string_dtype),
            ]
        )
        events = file.create_group("events")
        events.create_dataset(
            "discontinuities",
            data=np.asarray(
                [
                    (2, 30, "sample_index_gap", "{}"),
                    (4, 50, "sample_index_gap", "{}"),
                ],
                dtype=discontinuity_dtype,
            ),
        )

    evidence = scan_hdf5_signal_evidence(
        path,
        formal_start_ns=20,
        formal_stop_ns=40,
    )
    assert evidence.formal_sample_count == 3
    assert evidence.sequence_gap_count == 1
    assert evidence.nonfinite_value_count == 1
    assert evidence.minimum == [1.0, 2.0]
    assert evidence.maximum == [4.0, 8.0]
    # Jumps crossing a non-finite sample are not invented; the non-finite rule
    # already fails independently and only adjacent finite pairs are measured.
    assert evidence.maximum_absolute_jump == 2.0
