"""Explainable, configuration-driven quality evaluation for one Trial."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import shutil
from typing import Any, Literal

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from exo_collection.domain.models import QualityGrade
from exo_collection.storage.manifest import (
    QualityIssue,
    QualityIssueSeverity,
)

from .config import QualityRulesDocument, SignalQualityRules, StoragePolicyDocument


class QualityEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class RuleStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNASSESSED = "UNASSESSED"


class RuleResult(QualityEvidenceModel):
    code: str = Field(min_length=1)
    status: RuleStatus
    scope: Literal["storage", "structural", "sync", "clock", "ultrasound", "signal"]
    message: str = Field(min_length=1)
    modality: str | None = None
    metric: str | None = None
    observed_value: Any = None
    threshold: Any = None
    required_for_grade_a: bool = False


class DiskSpaceEvidence(QualityEvidenceModel):
    path: str
    free_bytes: int = Field(ge=0)
    required_free_bytes: int = Field(gt=0)

    @property
    def passed(self) -> bool:
        return self.free_bytes >= self.required_free_bytes


class SyncEdgeEvidence(QualityEvidenceModel):
    pulse_id: str
    edge_type: Literal["rising", "falling"]
    host_monotonic_ns: int = Field(ge=0)
    pulse_width_ns: int | None = Field(default=None, ge=0)


class ClockMappingEvidence(QualityEvidenceModel):
    modality: str
    anchor_count: int = Field(ge=0)
    rms_residual_ns: float | None = Field(default=None, ge=0)


class UltrasoundEvidence(QualityEvidenceModel):
    formal_frame_count: int = Field(ge=0)
    zero_fraction: float | None = Field(default=None, ge=0, le=1)
    saturation_fraction: float | None = Field(default=None, ge=0, le=1)
    evidence_scope: str = "bounded_spatially_downsampled_acquisition_history"


class SignalEvidence(QualityEvidenceModel):
    formal_sample_count: int = Field(ge=0)
    sequence_gap_count: int = Field(default=0, ge=0)
    nonfinite_value_count: int = Field(ge=0)
    minimum: list[float | None] = Field(default_factory=list)
    maximum: list[float | None] = Field(default_factory=list)
    maximum_absolute_jump: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_channel_extrema(self) -> SignalEvidence:
        if len(self.minimum) != len(self.maximum):
            raise ValueError("minimum and maximum channel vectors must have equal length")
        return self

    @property
    def maximum_channel_span(self) -> float | None:
        spans = [
            high - low
            for low, high in zip(self.minimum, self.maximum, strict=True)
            if low is not None and high is not None
        ]
        return max(spans) if spans else None


class TrialQualityEvidence(QualityEvidenceModel):
    formal_duration_s: float = Field(ge=0)
    formal_item_counts: dict[str, int]
    sequence_gap_counts: dict[str, int]
    dropped_batch_counts: dict[str, int]
    sync_edges: tuple[SyncEdgeEvidence, ...]
    first_trigger_host_monotonic_ns: int | None = Field(default=None, ge=0)
    clock_mappings: tuple[ClockMappingEvidence, ...]
    ultrasound: UltrasoundEvidence
    signals: dict[str, SignalEvidence]
    disk_space: DiskSpaceEvidence

    @model_validator(mode="after")
    def validate_nonnegative_counts(self) -> TrialQualityEvidence:
        for field_name in (
            "formal_item_counts",
            "sequence_gap_counts",
            "dropped_batch_counts",
        ):
            values = getattr(self, field_name)
            if any(value < 0 for value in values.values()):
                raise ValueError(f"{field_name} values must be non-negative")
        return self


class QualityEvaluation(QualityEvidenceModel):
    algorithm_version: str
    grade: QualityGrade
    results: tuple[RuleResult, ...]
    issues: tuple[QualityIssue, ...]

    @property
    def unassessed_count(self) -> int:
        return sum(result.status is RuleStatus.UNASSESSED for result in self.results)


class InsufficientDiskSpaceError(RuntimeError):
    def __init__(self, evidence: DiskSpaceEvidence) -> None:
        self.evidence = evidence
        super().__init__(
            "insufficient free disk space: "
            f"{evidence.free_bytes} bytes available, "
            f"{evidence.required_free_bytes} bytes required at {evidence.path}"
        )


def check_disk_space(path: str | Path, policy: StoragePolicyDocument) -> DiskSpaceEvidence:
    target = Path(path).expanduser().resolve()
    usage = shutil.disk_usage(target)
    evidence = DiskSpaceEvidence(
        path=str(target),
        free_bytes=int(usage.free),
        required_free_bytes=int(policy.minimum_free_space_gib * 1024**3),
    )
    if not evidence.passed:
        raise InsufficientDiskSpaceError(evidence)
    return evidence


def scan_hdf5_signal_evidence(
    path: str | Path,
    *,
    formal_start_ns: int,
    formal_stop_ns: int,
) -> SignalEvidence:
    """Scan one medium-rate HDF5 signal in bounded chunks after Writer close."""

    sample_count = 0
    sequence_gap_count = 0
    nonfinite_count = 0
    minima: np.ndarray | None = None
    maxima: np.ndarray | None = None
    maximum_jump: float | None = None
    previous: np.ndarray | None = None
    with h5py.File(Path(path), "r") as file:
        data = file["samples/data"]
        timestamps = file["samples/host_monotonic_ns"]
        chunk_rows = int(data.chunks[0] if data.chunks else 4096)
        for start in range(0, int(data.shape[0]), chunk_rows):
            stop = min(int(data.shape[0]), start + chunk_rows)
            chunk_times = np.asarray(timestamps[start:stop], dtype=np.uint64)
            keep = (chunk_times >= formal_start_ns) & (chunk_times <= formal_stop_ns)
            if not np.any(keep):
                continue
            values = np.asarray(data[start:stop])[keep].reshape(int(np.sum(keep)), -1)
            sample_count += int(values.shape[0])
            finite = np.isfinite(values)
            nonfinite_count += int(values.size - np.count_nonzero(finite))
            safe = np.where(finite, values, np.nan).astype(np.float64, copy=False)
            with np.errstate(all="ignore"):
                chunk_min = np.nanmin(safe, axis=0)
                chunk_max = np.nanmax(safe, axis=0)
            minima = chunk_min if minima is None else np.fmin(minima, chunk_min)
            maxima = chunk_max if maxima is None else np.fmax(maxima, chunk_max)

            if previous is not None:
                joined = np.vstack((previous, safe))
            else:
                joined = safe
            if joined.shape[0] > 1:
                with np.errstate(all="ignore"):
                    candidate = float(np.nanmax(np.abs(np.diff(joined, axis=0))))
                if np.isfinite(candidate):
                    maximum_jump = (
                        candidate if maximum_jump is None else max(maximum_jump, candidate)
                    )
            previous = safe[-1:]

        discontinuities = file.get("events/discontinuities")
        if discontinuities is not None:
            for start in range(0, int(discontinuities.shape[0]), 1024):
                records = discontinuities[start : start + 1024]
                for record in records:
                    host_ns = int(record["host_monotonic_ns"])
                    kind = record["kind"]
                    if isinstance(kind, bytes):
                        kind = kind.decode("utf-8")
                    if (
                        formal_start_ns <= host_ns <= formal_stop_ns
                        and str(kind) == "sample_index_gap"
                    ):
                        sequence_gap_count += 1

    def serialise(values: np.ndarray | None) -> list[float | None]:
        if values is None:
            return []
        return [float(value) if np.isfinite(value) else None for value in values]

    return SignalEvidence(
        formal_sample_count=sample_count,
        sequence_gap_count=sequence_gap_count,
        nonfinite_value_count=nonfinite_count,
        minimum=serialise(minima),
        maximum=serialise(maxima),
        maximum_absolute_jump=maximum_jump,
    )


def _result(
    code: str,
    status: RuleStatus,
    scope: Literal["storage", "structural", "sync", "clock", "ultrasound", "signal"],
    message: str,
    *,
    modality: str | None = None,
    metric: str | None = None,
    observed_value: Any = None,
    threshold: Any = None,
    required_for_grade_a: bool = False,
) -> RuleResult:
    return RuleResult(
        code=code,
        status=status,
        scope=scope,
        message=message,
        modality=modality,
        metric=metric,
        observed_value=observed_value,
        threshold=threshold,
        required_for_grade_a=required_for_grade_a,
    )


def _range_result(
    *,
    code: str,
    scope: Literal["sync", "clock"],
    values: list[float],
    minimum: float | int | None,
    maximum: float | int | None,
    metric: str,
    modality: str,
) -> RuleResult:
    configured = minimum is not None or maximum is not None
    threshold = {"minimum": minimum, "maximum": maximum}
    if not configured:
        return _result(
            code,
            RuleStatus.UNASSESSED,
            scope,
            f"{metric} thresholds are not configured; no calibrated claim was made",
            modality=modality,
            metric=metric,
            observed_value=values,
            threshold=threshold,
        )
    if not values:
        return _result(
            code,
            RuleStatus.UNASSESSED,
            scope,
            f"insufficient evidence to evaluate configured {metric} thresholds",
            modality=modality,
            metric=metric,
            observed_value=[],
            threshold=threshold,
        )
    failures = [
        value
        for value in values
        if (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ]
    return _result(
        code,
        RuleStatus.FAIL if failures else RuleStatus.PASS,
        scope,
        f"{metric} {'violated' if failures else 'passed'} configured bounds",
        modality=modality,
        metric=metric,
        observed_value=values,
        threshold=threshold,
        required_for_grade_a=True,
    )


def _signal_results(
    modality: Literal["imu", "encoder"],
    evidence: SignalEvidence,
    rules: SignalQualityRules,
) -> list[RuleResult]:
    results: list[RuleResult] = []
    results.append(
        _result(
            "SIGNAL_NONFINITE_VALUES",
            RuleStatus.FAIL if evidence.nonfinite_value_count else RuleStatus.PASS,
            "signal",
            f"{modality} contains "
            f"{evidence.nonfinite_value_count} non-finite values",
            modality=modality,
            metric="nonfinite_value_count",
            observed_value=evidence.nonfinite_value_count,
            threshold=0,
            required_for_grade_a=True,
        )
    )
    span = evidence.maximum_channel_span
    if rules.constant_tolerance is None:
        results.append(
            _result(
                "SIGNAL_CONSTANT",
                RuleStatus.UNASSESSED,
                "signal",
                f"{modality} constant-signal tolerance is not configured",
                modality=modality,
                metric="maximum_channel_span",
                observed_value=span,
            )
        )
    else:
        is_constant = span is not None and span <= rules.constant_tolerance
        results.append(
            _result(
                "SIGNAL_CONSTANT",
                RuleStatus.WARNING if is_constant else RuleStatus.PASS,
                "signal",
                f"{modality} is {'constant' if is_constant else 'not constant'} "
                "under the configured format-level tolerance",
                modality=modality,
                metric="maximum_channel_span",
                observed_value=span,
                threshold=rules.constant_tolerance,
                required_for_grade_a=True,
            )
        )

    calibrated_threshold = {
        "minimum": rules.calibrated_minimum,
        "maximum": rules.calibrated_maximum,
        "calibration_reference": rules.calibration_reference,
    }
    calibrated_configured = (
        rules.calibrated_minimum is not None or rules.calibrated_maximum is not None
    )
    if not calibrated_configured:
        results.append(
            _result(
                "SIGNAL_CALIBRATED_RANGE",
                RuleStatus.UNASSESSED,
                "signal",
                f"{modality} calibrated range is not configured",
                modality=modality,
                metric="value_range",
                observed_value={"minimum": evidence.minimum, "maximum": evidence.maximum},
                threshold=calibrated_threshold,
            )
        )
    else:
        lows = [value for value in evidence.minimum if value is not None]
        highs = [value for value in evidence.maximum if value is not None]
        violated = (
            rules.calibrated_minimum is not None
            and lows
            and min(lows) < rules.calibrated_minimum
        ) or (
            rules.calibrated_maximum is not None
            and highs
            and max(highs) > rules.calibrated_maximum
        )
        status = (
            RuleStatus.FAIL
            if violated and rules.calibrated_violation_severity == "ERROR"
            else RuleStatus.WARNING
            if violated
            else RuleStatus.PASS
        )
        results.append(
            _result(
                "SIGNAL_CALIBRATED_RANGE",
                status,
                "signal",
                f"{modality} {'violated' if violated else 'passed'} calibrated range",
                modality=modality,
                metric="value_range",
                observed_value={"minimum": evidence.minimum, "maximum": evidence.maximum},
                threshold=calibrated_threshold,
                required_for_grade_a=True,
            )
        )

    if rules.maximum_absolute_jump is None:
        results.append(
            _result(
                "SIGNAL_ABSOLUTE_JUMP",
                RuleStatus.UNASSESSED,
                "signal",
                f"{modality} jump threshold is not configured",
                modality=modality,
                metric="maximum_absolute_jump",
                observed_value=evidence.maximum_absolute_jump,
            )
        )
    else:
        violated = (
            evidence.maximum_absolute_jump is not None
            and evidence.maximum_absolute_jump > rules.maximum_absolute_jump
        )
        status = (
            RuleStatus.FAIL
            if violated and rules.calibrated_violation_severity == "ERROR"
            else RuleStatus.WARNING
            if violated
            else RuleStatus.PASS
        )
        results.append(
            _result(
                "SIGNAL_ABSOLUTE_JUMP",
                status,
                "signal",
                f"{modality} {'violated' if violated else 'passed'} calibrated jump bound",
                modality=modality,
                metric="maximum_absolute_jump",
                observed_value=evidence.maximum_absolute_jump,
                threshold={
                    "maximum": rules.maximum_absolute_jump,
                    "calibration_reference": rules.calibration_reference,
                },
                required_for_grade_a=True,
            )
        )
    return results


def evaluate_trial_quality(
    evidence: TrialQualityEvidence,
    rules: QualityRulesDocument,
) -> QualityEvaluation:
    results: list[RuleResult] = []
    results.append(
        _result(
            "DISK_SPACE_PREFLIGHT",
            RuleStatus.PASS if evidence.disk_space.passed else RuleStatus.FAIL,
            "storage",
            "data-root free space passed the configured preflight requirement"
            if evidence.disk_space.passed
            else "data-root free space did not pass the configured preflight requirement",
            metric="free_bytes",
            observed_value=evidence.disk_space.free_bytes,
            threshold=evidence.disk_space.required_free_bytes,
            required_for_grade_a=True,
        )
    )
    duration_passed = evidence.formal_duration_s >= rules.structural.minimum_formal_duration_s
    results.append(
        _result(
            "FORMAL_RECORDING_WINDOW",
            RuleStatus.PASS if duration_passed else RuleStatus.FAIL,
            "structural",
            "formal recording window passed the configured minimum"
            if duration_passed
            else "formal recording window is missing or too short",
            metric="formal_duration_s",
            observed_value=evidence.formal_duration_s,
            threshold=rules.structural.minimum_formal_duration_s,
            required_for_grade_a=True,
        )
    )
    for modality in rules.required_modalities:
        count = evidence.formal_item_counts.get(modality, 0)
        results.append(
            _result(
                "REQUIRED_MODALITY_FORMAL_DATA",
                RuleStatus.PASS if count > 0 else RuleStatus.FAIL,
                "structural",
                f"required modality {modality} has "
                f"{count} persisted items in the formal window",
                modality=modality,
                metric="formal_item_count",
                observed_value=count,
                threshold=1,
                required_for_grade_a=True,
            )
        )
        gaps = evidence.sequence_gap_counts.get(modality, 0)
        results.append(
            _result(
                "SEQUENCE_CONTINUITY",
                RuleStatus.PASS
                if gaps <= rules.structural.maximum_sequence_gaps
                else RuleStatus.FAIL,
                "structural",
                f"{modality} sequence gap count is {gaps}",
                modality=modality,
                metric="sequence_gap_count",
                observed_value=gaps,
                threshold=rules.structural.maximum_sequence_gaps,
                required_for_grade_a=True,
            )
        )
        dropped = evidence.dropped_batch_counts.get(modality, 0)
        results.append(
            _result(
                "DROPPED_BATCHES",
                RuleStatus.PASS
                if dropped <= rules.structural.maximum_dropped_batches
                else RuleStatus.FAIL,
                "structural",
                f"{modality} dropped/overflow batch count is {dropped}",
                modality=modality,
                metric="dropped_batch_count",
                observed_value=dropped,
                threshold=rules.structural.maximum_dropped_batches,
                required_for_grade_a=True,
            )
        )

    rising = [edge for edge in evidence.sync_edges if edge.edge_type == "rising"]
    falling = [edge for edge in evidence.sync_edges if edge.edge_type == "falling"]
    results.append(
        _result(
            "SYNC_RISING_EDGE_COUNT",
            RuleStatus.PASS
            if len(rising) >= rules.sync.minimum_rising_edges
            else RuleStatus.FAIL,
            "sync",
            f"detected {len(rising)} qualified rising synchronization edges",
            modality="sync_pulse",
            metric="rising_edge_count",
            observed_value=len(rising),
            threshold=rules.sync.minimum_rising_edges,
            required_for_grade_a=True,
        )
    )
    results.append(
        _result(
            "FIRST_SYNC_TRIGGER",
            RuleStatus.PASS
            if evidence.first_trigger_host_monotonic_ns is not None
            else RuleStatus.FAIL,
            "sync",
            "the first qualified rising edge established formal t0"
            if evidence.first_trigger_host_monotonic_ns is not None
            else "no qualified rising edge established formal t0",
            modality="sync_pulse",
            metric="first_trigger_host_monotonic_ns",
            observed_value=evidence.first_trigger_host_monotonic_ns,
            threshold="required",
            required_for_grade_a=True,
        )
    )
    results.append(
        _result(
            "SYNC_COMPLETE_PULSE_COUNT",
            RuleStatus.PASS
            if len(falling) >= rules.sync.minimum_complete_pulses
            else RuleStatus.FAIL,
            "sync",
            f"detected {len(falling)} complete synchronization pulses",
            modality="sync_pulse",
            metric="complete_pulse_count",
            observed_value=len(falling),
            threshold=rules.sync.minimum_complete_pulses,
            required_for_grade_a=True,
        )
    )
    widths = [float(edge.pulse_width_ns) for edge in falling if edge.pulse_width_ns is not None]
    results.append(
        _range_result(
            code="SYNC_PULSE_WIDTH",
            scope="sync",
            values=widths,
            minimum=rules.sync.pulse_width_ns.minimum,
            maximum=rules.sync.pulse_width_ns.maximum,
            metric="pulse_width_ns",
            modality="sync_pulse",
        )
    )
    rising_times = sorted(edge.host_monotonic_ns for edge in rising)
    intervals = [
        float(current - previous)
        for previous, current in zip(rising_times, rising_times[1:])
    ]
    results.append(
        _range_result(
            code="SYNC_PULSE_INTERVAL",
            scope="sync",
            values=intervals,
            minimum=rules.sync.pulse_interval_ns.minimum,
            maximum=rules.sync.pulse_interval_ns.maximum,
            metric="pulse_interval_ns",
            modality="sync_pulse",
        )
    )

    mapping_by_modality = {item.modality: item for item in evidence.clock_mappings}
    for modality in rules.required_modalities:
        mapping = mapping_by_modality.get(modality)
        anchors = 0 if mapping is None else mapping.anchor_count
        results.append(
            _result(
                "CLOCK_MAPPING_ANCHORS",
                RuleStatus.PASS
                if anchors >= rules.sync.minimum_mapping_anchors
                else RuleStatus.FAIL,
                "clock",
                f"{modality} clock mapping has {anchors} anchors",
                modality=modality,
                metric="anchor_count",
                observed_value=anchors,
                threshold=rules.sync.minimum_mapping_anchors,
                required_for_grade_a=True,
            )
        )
        if rules.sync.maximum_mapping_rms_residual_ns is None:
            results.append(
                _result(
                    "CLOCK_MAPPING_RESIDUAL",
                    RuleStatus.UNASSESSED,
                    "clock",
                    "clock residual threshold is not calibrated/configured",
                    modality=modality,
                    metric="rms_residual_ns",
                    observed_value=None if mapping is None else mapping.rms_residual_ns,
                )
            )
        else:
            residual = None if mapping is None else mapping.rms_residual_ns
            failed = residual is None or residual > rules.sync.maximum_mapping_rms_residual_ns
            results.append(
                _result(
                    "CLOCK_MAPPING_RESIDUAL",
                    RuleStatus.FAIL if failed else RuleStatus.PASS,
                    "clock",
                    f"{modality} clock mapping residual "
                    f"{'failed' if failed else 'passed'} the configured bound",
                    modality=modality,
                    metric="rms_residual_ns",
                    observed_value=residual,
                    threshold=rules.sync.maximum_mapping_rms_residual_ns,
                    required_for_grade_a=True,
                )
            )

    if rules.ultrasound.all_zero_fraction_warning is None:
        results.append(
            _result(
                "ULTRASOUND_ALL_ZERO",
                RuleStatus.UNASSESSED,
                "ultrasound",
                "ultrasound all-zero fraction threshold is not configured",
                modality="ultrasound",
                metric="zero_fraction",
                observed_value=evidence.ultrasound.zero_fraction,
            )
        )
    else:
        zero_warning = (
            evidence.ultrasound.zero_fraction is not None
            and evidence.ultrasound.zero_fraction
            >= rules.ultrasound.all_zero_fraction_warning
        )
        results.append(
            _result(
                "ULTRASOUND_ALL_ZERO",
                RuleStatus.WARNING if zero_warning else RuleStatus.PASS,
                "ultrasound",
                "ultrasound retained evidence is all zero"
                if zero_warning
                else "ultrasound retained evidence is not all zero",
                modality="ultrasound",
                metric="zero_fraction",
                observed_value=evidence.ultrasound.zero_fraction,
                threshold=rules.ultrasound.all_zero_fraction_warning,
                required_for_grade_a=True,
            )
        )
    if rules.ultrasound.saturation_fraction_warning is None:
        results.append(
            _result(
                "ULTRASOUND_SATURATION",
                RuleStatus.UNASSESSED,
                "ultrasound",
                "ultrasound saturation threshold is not calibrated/configured",
                modality="ultrasound",
                metric="saturation_fraction",
                observed_value=evidence.ultrasound.saturation_fraction,
            )
        )
    else:
        saturation_warning = (
            evidence.ultrasound.saturation_fraction is not None
            and evidence.ultrasound.saturation_fraction
            >= rules.ultrasound.saturation_fraction_warning
        )
        results.append(
            _result(
                "ULTRASOUND_SATURATION",
                RuleStatus.WARNING if saturation_warning else RuleStatus.PASS,
                "ultrasound",
                "ultrasound saturation fraction violated calibrated threshold"
                if saturation_warning
                else "ultrasound saturation fraction passed calibrated threshold",
                modality="ultrasound",
                metric="saturation_fraction",
                observed_value=evidence.ultrasound.saturation_fraction,
                threshold={
                    "maximum": rules.ultrasound.saturation_fraction_warning,
                    "calibration_reference": rules.ultrasound.calibration_reference,
                },
                required_for_grade_a=True,
            )
        )

    for modality in ("imu", "encoder"):
        signal = evidence.signals.get(modality, SignalEvidence(
            formal_sample_count=0,
            nonfinite_value_count=0,
        ))
        results.extend(_signal_results(modality, signal, getattr(rules, modality)))

    issues: list[QualityIssue] = []
    for result in results:
        if result.status not in {RuleStatus.WARNING, RuleStatus.FAIL}:
            continue
        issues.append(
            QualityIssue(
                code=result.code,
                severity=(
                    QualityIssueSeverity.ERROR
                    if result.status is RuleStatus.FAIL
                    else QualityIssueSeverity.WARNING
                ),
                message=result.message,
                modality=result.modality,
                metric=result.metric,
                observed_value=result.observed_value,
                threshold=result.threshold,
            )
        )

    invalid_codes = {
        "DISK_SPACE_PREFLIGHT",
        "FORMAL_RECORDING_WINDOW",
        "REQUIRED_MODALITY_FORMAL_DATA",
        "SYNC_RISING_EDGE_COUNT",
        "FIRST_SYNC_TRIGGER",
    }
    failed = [result for result in results if result.status is RuleStatus.FAIL]
    warnings = [result for result in results if result.status is RuleStatus.WARNING]
    required_results = [result for result in results if result.required_for_grade_a]
    if any(result.code in invalid_codes for result in failed):
        grade = QualityGrade.INVALID
    elif failed:
        grade = QualityGrade.C
    elif warnings:
        grade = QualityGrade.B
    elif required_results and all(result.status is RuleStatus.PASS for result in required_results):
        grade = QualityGrade.A
    else:
        # This branch prevents an empty or accidentally skipped ruleset from
        # yielding A solely because it produced no issues.
        grade = QualityGrade.C
    return QualityEvaluation(
        algorithm_version=rules.algorithm_version,
        grade=grade,
        results=tuple(results),
        issues=tuple(issues),
    )


__all__ = [
    "ClockMappingEvidence",
    "DiskSpaceEvidence",
    "InsufficientDiskSpaceError",
    "QualityEvaluation",
    "RuleResult",
    "RuleStatus",
    "SignalEvidence",
    "SyncEdgeEvidence",
    "TrialQualityEvidence",
    "UltrasoundEvidence",
    "check_disk_space",
    "evaluate_trial_quality",
    "scan_hdf5_signal_evidence",
]
