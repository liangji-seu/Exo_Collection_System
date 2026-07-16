"""Strict, versioned configuration for automatic Trial quality rules."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QualityConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class StructuralQualityRules(QualityConfigModel):
    minimum_formal_duration_s: float = Field(gt=0)
    maximum_sequence_gaps: int = Field(ge=0)
    maximum_dropped_batches: int = Field(ge=0)


class OptionalRange(QualityConfigModel):
    minimum: float | int | None = None
    maximum: float | int | None = None

    @model_validator(mode="after")
    def validate_order(self) -> OptionalRange:
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.maximum < self.minimum
        ):
            raise ValueError("maximum must be greater than or equal to minimum")
        return self

    @property
    def configured(self) -> bool:
        return self.minimum is not None or self.maximum is not None


class SyncQualityRules(QualityConfigModel):
    minimum_rising_edges: int = Field(ge=1)
    minimum_complete_pulses: int = Field(ge=1)
    minimum_mapping_anchors: int = Field(ge=1)
    pulse_width_ns: OptionalRange
    pulse_interval_ns: OptionalRange
    maximum_mapping_rms_residual_ns: float | None = Field(default=None, gt=0)

    @field_validator("pulse_width_ns", "pulse_interval_ns")
    @classmethod
    def require_nonnegative_temporal_bounds(cls, value: OptionalRange) -> OptionalRange:
        if value.minimum is not None and value.minimum < 0:
            raise ValueError("temporal minimum must be non-negative")
        if value.maximum is not None and value.maximum <= 0:
            raise ValueError("temporal maximum must be positive")
        return value


class UltrasoundQualityRules(QualityConfigModel):
    all_zero_fraction_warning: float | None = Field(default=None, ge=0, le=1)
    saturation_fraction_warning: float | None = Field(default=None, gt=0, le=1)
    calibration_reference: NonEmptyStr | None = None

    @model_validator(mode="after")
    def require_saturation_basis(self) -> UltrasoundQualityRules:
        if self.saturation_fraction_warning is not None and self.calibration_reference is None:
            raise ValueError(
                "saturation_fraction_warning requires calibration_reference"
            )
        return self


class SignalQualityRules(QualityConfigModel):
    constant_tolerance: float | None = Field(default=None, ge=0)
    calibrated_minimum: float | None = None
    calibrated_maximum: float | None = None
    maximum_absolute_jump: float | None = Field(default=None, gt=0)
    calibration_reference: NonEmptyStr | None = None
    calibrated_violation_severity: Literal["WARNING", "ERROR"] = "WARNING"

    @model_validator(mode="after")
    def validate_calibrated_thresholds(self) -> SignalQualityRules:
        if (
            self.calibrated_minimum is not None
            and self.calibrated_maximum is not None
            and self.calibrated_maximum < self.calibrated_minimum
        ):
            raise ValueError("calibrated_maximum must not be below calibrated_minimum")
        calibrated = any(
            value is not None
            for value in (
                self.calibrated_minimum,
                self.calibrated_maximum,
                self.maximum_absolute_jump,
            )
        )
        if calibrated and self.calibration_reference is None:
            raise ValueError(
                "calibrated signal thresholds require calibration_reference"
            )
        return self


class QualityRulesDocument(QualityConfigModel):
    schema_version: Literal["1.0.0"]
    algorithm_version: NonEmptyStr
    required_modalities: tuple[NonEmptyStr, ...]
    structural: StructuralQualityRules
    sync: SyncQualityRules
    ultrasound: UltrasoundQualityRules
    imu: SignalQualityRules
    encoder: SignalQualityRules

    @field_validator("required_modalities")
    @classmethod
    def validate_required_modalities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("required_modalities must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("required_modalities must be unique")
        return values


class StoragePolicyDocument(QualityConfigModel):
    schema_version: Literal["1.0.0"]
    data_root: NonEmptyStr
    minimum_free_space_gib: float = Field(gt=0)
    ultrasound_block_frames: int = Field(gt=0)
    hdf5_chunk_samples: int = Field(gt=0)


def _configuration_candidates(*relative_parts: str) -> list[Path]:
    candidates: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root).joinpath("config", *relative_parts))
    candidates.extend(
        [
            Path(sys.executable).resolve().parent.joinpath("config", *relative_parts),
            Path(__file__).resolve().parents[3].joinpath("config", *relative_parts),
            Path.cwd().joinpath("config", *relative_parts),
        ]
    )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _default_path(*relative_parts: str) -> Path:
    candidates = _configuration_candidates(*relative_parts)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    display = "/".join(relative_parts)
    raise FileNotFoundError(f"configuration {display} not found; searched:\n{searched}")


def default_quality_rules_path() -> Path:
    return _default_path("quality_rules", "default.json")


def default_storage_policy_path() -> Path:
    return _default_path("storage.json")


def _load_json(path: Path, *, label: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc


def load_quality_rules(path: str | Path | None = None) -> QualityRulesDocument:
    source = (
        default_quality_rules_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    return QualityRulesDocument.model_validate(_load_json(source, label="quality rules"))


def load_storage_policy(path: str | Path | None = None) -> StoragePolicyDocument:
    source = (
        default_storage_policy_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    return StoragePolicyDocument.model_validate(_load_json(source, label="storage policy"))


__all__ = [
    "OptionalRange",
    "QualityRulesDocument",
    "SignalQualityRules",
    "StoragePolicyDocument",
    "StructuralQualityRules",
    "SyncQualityRules",
    "UltrasoundQualityRules",
    "default_quality_rules_path",
    "default_storage_policy_path",
    "load_quality_rules",
    "load_storage_policy",
]
