"""Append-only External Artifact Importer.

Finalized Trial packages are immutable.  Consequently, an external force-plate,
motion-capture, or generic file is not inserted into ``raw/external`` after the
fact and the published Trial Manifest is never rewritten.  Each import creates
an independently checksummed annex under ``<dataset>/external_annexes``.  The
annex binds itself to the exact base Manifest bytes and Trial UUID, so consumers
can join it without guessing from filenames.

Only generic files and user-selected CSV columns are understood here.  This
module deliberately contains no vendor protocol assumptions.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Annotated, Any, Iterable, Literal, Mapping, Sequence
from uuid import UUID, uuid4

import h5py
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from exo_collection.domain.models import Sha256
from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import read_activity
from exo_collection.storage.layout import path_has_unpublished_component
from exo_collection.storage.manifest import ManifestArtifact, TrialManifest, load_manifest
from exo_collection.timing.alignment import align_shared_pulses


ANNEX_SCHEMA_VERSION = "1.0.0"
ALIGNMENT_QUALITY_VERSION = "external-pulse-residual-quality-1.0.0"
ANNEX_DIRECTORY_NAME = "external_annexes"
_COPY_CHUNK_BYTES = 4 * 1024 * 1024
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(password|passwd|pwd|token|secret|credential|api[_-]?key)"
    r"(\s*[:=]\s*)([^/\\]+)"
)
_URI_USERINFO = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)([^/@:\s]+):([^/@\s]+)@"
)
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,12}$")

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExternalImportError(RuntimeError):
    """A safe import could not be completed.

    ``code`` is stable for UI handling.  Messages intentionally avoid raw
    exception text because OS errors can contain credential-bearing paths.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ExternalModality(StrEnum):
    FORCE_PLATE = "force_plate"
    MOCAP = "mocap"
    OTHER = "other"


class ExternalImportRequest(BaseModel):
    """Pure service request suitable for a future Data Studio dialog."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dataset_root: Path
    trial_manifest_path: Path
    source_path: Path
    modality: ExternalModality
    source_system: NonEmptyText = "manual_external_import"
    other_modality_label: str | None = None
    external_clock_domain: NonEmptyText = "external_clock"
    external_time_unit: NonEmptyText = "s"
    external_time_scale_to_ns: float | None = Field(default=None, gt=0)
    external_pulse_times: list[float] | None = None
    pulse_csv_path: Path | None = None
    pulse_csv_column: str | None = None
    csv_delimiter: str | None = None
    csv_encoding: NonEmptyText = "utf-8-sig"
    annex_uuid: UUID = Field(default_factory=uuid4)

    @field_validator("other_modality_label", "pulse_csv_column")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("external_pulse_times", mode="before")
    @classmethod
    def reject_boolean_pulse_times(cls, value: Any) -> Any:
        if value is not None and any(isinstance(item, bool) for item in value):
            raise ValueError("boolean values are not pulse times")
        return value

    @field_validator("csv_delimiter")
    @classmethod
    def validate_delimiter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 1 or value in {"\r", "\n", "\0"}:
            raise ValueError("csv_delimiter must be one printable character")
        return value

    @model_validator(mode="after")
    def validate_pulse_source(self) -> "ExternalImportRequest":
        has_manual = self.external_pulse_times is not None
        has_csv = self.pulse_csv_column is not None or self.pulse_csv_path is not None
        if has_manual == has_csv:
            raise ValueError(
                "provide exactly one pulse source: external_pulse_times or a CSV column"
            )
        if has_csv and self.pulse_csv_column is None:
            raise ValueError("pulse_csv_column is required for CSV pulse extraction")
        if self.modality is ExternalModality.OTHER and not self.other_modality_label:
            raise ValueError("other_modality_label is required when modality is other")
        if self.modality is not ExternalModality.OTHER and self.other_modality_label:
            raise ValueError("other_modality_label is only valid for modality=other")
        return self


class AnnexModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class SourceAudit(AnnexModel):
    source_path_redacted: NonEmptyText
    source_path_sha256: Sha256
    original_filename_redacted: NonEmptyText
    source_size_bytes: int = Field(ge=0)
    source_mtime_ns: int = Field(ge=0)


class AnnexFile(AnnexModel):
    artifact_uuid: UUID
    role: Literal["external_original", "pulse_evidence"]
    relative_path: NonEmptyText
    media_type: NonEmptyText
    size_bytes: int = Field(ge=0)
    sha256: Sha256
    source_audit: SourceAudit

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        candidate = Path(value.replace("\\", "/"))
        if (
            candidate.is_absolute()
            or not candidate.parts
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or any(_has_active_component(Path(part)) for part in candidate.parts)
        ):
            raise ValueError("annex file path must be safe and relative")
        return candidate.as_posix()


class MappingReference(AnnexModel):
    mapping_uuid: UUID
    relative_path: Literal["alignment/mapping.json"] = "alignment/mapping.json"
    size_bytes: int = Field(ge=0)
    sha256: Sha256
    quality: Literal["GOOD", "ACCEPTABLE", "POOR", "UNAVAILABLE"]
    offset_only: bool
    anchor_count: int = Field(ge=1)


class ExternalAnnexManifest(AnnexModel):
    schema_name: Literal["exo-external-artifact-annex"] = (
        "exo-external-artifact-annex"
    )
    schema_version: Literal[ANNEX_SCHEMA_VERSION] = ANNEX_SCHEMA_VERSION
    annex_uuid: UUID
    trial_uuid: UUID
    base_manifest_uuid: UUID
    base_manifest_schema_version: NonEmptyText
    base_manifest_relative_path: NonEmptyText
    base_manifest_sha256: Sha256
    modality: ExternalModality
    other_modality_label: str | None = None
    source_system: NonEmptyText
    external_clock_domain: NonEmptyText
    imported_at_utc: datetime
    immutable: Literal[True] = True
    files: tuple[AnnexFile, ...]
    mapping: MappingReference
    checksum_manifest: Literal["checksums.sha256"] = "checksums.sha256"

    @field_validator("imported_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("imported_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("base_manifest_relative_path")
    @classmethod
    def validate_base_manifest_path(cls, value: str) -> str:
        candidate = Path(value.replace("\\", "/"))
        if (
            candidate.is_absolute()
            or not candidate.parts
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or _has_active_component(candidate)
        ):
            raise ValueError("base Manifest path must be safe and relative")
        return candidate.as_posix()

    @model_validator(mode="after")
    def validate_file_roles(self) -> "ExternalAnnexManifest":
        if not self.files or self.files[0].role != "external_original":
            raise ValueError("annex needs one external original file")
        ids = [item.artifact_uuid for item in self.files]
        paths = [item.relative_path for item in self.files]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("annex file UUIDs and paths must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ExternalImportResult:
    annex_uuid: UUID
    trial_uuid: UUID
    annex_directory: Path
    annex_manifest_path: Path
    mapping_path: Path
    copied_artifact_path: Path
    base_manifest_sha256: str
    copied_artifact_sha256: str
    quality: str
    offset_only: bool
    anchor_count: int


@dataclass(frozen=True, slots=True)
class _InternalPulse:
    pulse_id: str
    host_monotonic_ns: int


def _has_active_component(path: Path) -> bool:
    return path_has_unpublished_component(path)


def _redact_sensitive_text(value: str) -> str:
    redacted = _URI_USERINFO.sub(r"\1\2:***@", value)
    return _SENSITIVE_ASSIGNMENT.sub(r"\1\2***", redacted)


def _safe_path_display(path: Path) -> str:
    return _redact_sensitive_text(str(path))


def _path_audit_sha256(path: Path) -> str:
    return hashlib.sha256(os.fsencode(str(path))).hexdigest()


def _require_idle(dataset_root: Path) -> None:
    if read_activity(dataset_root) is not None:
        raise ExternalImportError(
            "ACQUISITION_ACTIVE",
            "Collector 正在采集；外部模态导入只允许人工离线执行。",
        )


def _ensure_under(candidate: Path, root: Path, *, code: str, message: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ExternalImportError(code, message) from None


def _is_link_or_reparse(path: Path) -> bool:
    """Detect POSIX links plus Windows junction/reparse-point aliases."""

    information = path.lstat()
    file_attributes = int(getattr(information, "st_file_attributes", 0))
    return stat.S_ISLNK(information.st_mode) or bool(file_attributes & 0x400)


def _resolve_regular_source(path: Path, *, role: str) -> Path:
    if _has_active_component(path):
        raise ExternalImportError(
            "TEMPORARY_PATH_REJECTED",
            f"拒绝把 .recording/.partial 路径作为{role}。",
        )
    expanded = path.expanduser()
    try:
        if expanded.is_symlink():
            raise ExternalImportError("INVALID_SOURCE", f"{role}不能是符号链接。")
        resolved = expanded.resolve(strict=True)
    except OSError:
        raise ExternalImportError("SOURCE_NOT_FOUND", f"{role}不存在或不可访问。") from None
    if _has_active_component(resolved):
        raise ExternalImportError(
            "TEMPORARY_PATH_REJECTED",
            f"拒绝把 .recording/.partial 路径作为{role}。",
        )
    if resolved.is_symlink() or not resolved.is_file():
        raise ExternalImportError("INVALID_SOURCE", f"{role}必须是普通文件。")
    return resolved


def _hash_file_with_idle(path: Path, dataset_root: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_COPY_CHUNK_BYTES):
                _require_idle(dataset_root)
                digest.update(chunk)
    except ExternalImportError:
        raise
    except OSError:
        raise ExternalImportError("READ_FAILED", "读取完整性数据失败。") from None
    return digest.hexdigest()


def _media_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _safe_suffix(path: Path) -> str:
    return path.suffix.casefold() if _SAFE_SUFFIX.fullmatch(path.suffix) else ".bin"


def _source_audit(path: Path, stat: os.stat_result) -> SourceAudit:
    return SourceAudit(
        source_path_redacted=_safe_path_display(path),
        source_path_sha256=_path_audit_sha256(path),
        original_filename_redacted=_redact_sensitive_text(path.name),
        source_size_bytes=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )


def _copy_source(
    source: Path,
    destination: Path,
    *,
    dataset_root: Path,
) -> tuple[str, int, SourceAudit]:
    """Copy exactly once, hashing the bytes that reached the annex."""

    _require_idle(dataset_root)
    try:
        before = source.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        copied = 0
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while chunk := input_stream.read(_COPY_CHUNK_BYTES):
                _require_idle(dataset_root)
                output_stream.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        after = source.stat()
    except ExternalImportError:
        raise
    except OSError:
        raise ExternalImportError("COPY_FAILED", "外部原文件复制失败。") from None
    stable_fields = (
        before.st_size == after.st_size == copied
        and before.st_mtime_ns == after.st_mtime_ns
        and (before.st_ino == 0 or after.st_ino == 0 or before.st_ino == after.st_ino)
    )
    if not stable_fields:
        raise ExternalImportError(
            "SOURCE_CHANGED",
            "导入期间外部源文件发生变化；本次临时导入已取消。",
        )
    return digest.hexdigest(), copied, _source_audit(source, before)


def _load_finalized_trial(
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[Path, Path, TrialManifest, str]:
    if _has_active_component(manifest_path):
        raise ExternalImportError(
            "TEMPORARY_PATH_REJECTED",
            "External Artifact Importer 不读取 .recording/.partial Trial。",
        )
    try:
        resolved = manifest_path.expanduser().resolve(strict=True)
    except OSError:
        raise ExternalImportError("MANIFEST_NOT_FOUND", "Trial Manifest 不存在。") from None
    if resolved.name != "manifest.json" or _has_active_component(resolved):
        raise ExternalImportError("INVALID_MANIFEST_PATH", "请选择已发布的 manifest.json。")
    _ensure_under(
        resolved,
        dataset_root,
        code="MANIFEST_OUTSIDE_DATASET",
        message="Trial Manifest 不在指定数据根目录内。",
    )
    try:
        manifest = load_manifest(resolved)
    except Exception:
        raise ExternalImportError(
            "INVALID_TRIAL_MANIFEST", "Trial Manifest 无法通过 Schema 校验。"
        ) from None
    if manifest.state is not TrialState.FINALIZED:
        raise ExternalImportError(
            "NOT_FINALIZED",
            f"仅允许导入到 FINALIZED Trial；当前状态为 {manifest.state.value}。",
        )
    base_sha256 = _hash_file_with_idle(resolved, dataset_root)
    return resolved, resolved.parent.resolve(), manifest, base_sha256


def _resolve_manifest_artifact(
    trial_root: Path,
    artifact: ManifestArtifact,
) -> Path:
    lexical = trial_root / Path(artifact.relative_path)
    if _has_active_component(lexical):
        raise ExternalImportError(
            "TEMPORARY_SYNC_ARTIFACT", "同步 Artifact 指向临时文件。"
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError:
        raise ExternalImportError("SYNC_ARTIFACT_MISSING", "同步 HDF5 文件缺失。") from None
    _ensure_under(
        resolved,
        trial_root,
        code="SYNC_ARTIFACT_ESCAPE",
        message="同步 Artifact 路径逃逸 Trial 目录。",
    )
    if _has_active_component(resolved) or not resolved.is_file():
        raise ExternalImportError("INVALID_SYNC_ARTIFACT", "同步 Artifact 不是已发布文件。")
    return resolved


def _sync_artifact(manifest: TrialManifest) -> ManifestArtifact:
    candidates = [
        artifact
        for artifact in manifest.artifacts
        if artifact.modality == "sync_pulse"
        and artifact.relative_path.casefold().endswith(".h5")
    ]
    exact = [item for item in candidates if item.relative_path == "raw/sync_pulse.h5"]
    selected = exact or candidates
    if len(selected) != 1:
        raise ExternalImportError(
            "SYNC_ARTIFACT_AMBIGUOUS",
            "Manifest 必须唯一标识 raw/sync_pulse.h5。",
        )
    return selected[0]


def _read_internal_rising_edges(
    sync_path: Path,
    manifest: TrialManifest,
) -> tuple[_InternalPulse, ...]:
    start_ns = manifest.timing.start_host_monotonic_ns
    stop_ns = manifest.timing.stop_host_monotonic_ns
    try:
        with h5py.File(sync_path, "r") as handle:
            if not bool(handle.attrs.get("closed_cleanly", False)):
                raise ExternalImportError(
                    "UNCLEAN_SYNC_ARTIFACT", "同步 HDF5 未标记为正常关闭。"
                )
            if "events/records" not in handle:
                raise ExternalImportError(
                    "SYNC_EVENTS_MISSING", "同步 HDF5 缺少 events/records。"
                )
            raw_records = handle["events/records"][:]
    except ExternalImportError:
        raise
    except (OSError, ValueError, KeyError):
        raise ExternalImportError("INVALID_SYNC_HDF5", "同步 HDF5 无法安全读取。") from None

    pulses: list[_InternalPulse] = []
    for raw in raw_records:
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
            payload = json.loads(text)
        except (UnicodeError, json.JSONDecodeError, TypeError):
            raise ExternalImportError(
                "INVALID_SYNC_EVENT", "同步 HDF5 包含无法解析的事件。"
            ) from None
        if not isinstance(payload, Mapping):
            raise ExternalImportError("INVALID_SYNC_EVENT", "同步事件必须是 JSON 对象。")
        if str(payload.get("event_type", "")).casefold() != "sync_pulse":
            continue
        if str(payload.get("edge_type", "")).casefold() != "rising":
            continue
        event_trial_uuid = payload.get("trial_uuid")
        if event_trial_uuid is not None and str(event_trial_uuid) != str(manifest.trial_uuid):
            raise ExternalImportError(
                "INVALID_SYNC_EVENT", "同步事件引用了其他 Trial UUID。"
            )
        try:
            timestamp = int(payload["host_monotonic_ns"])
            pulse_id = str(payload["pulse_id"]).strip()
        except (KeyError, TypeError, ValueError):
            raise ExternalImportError(
                "INVALID_SYNC_EVENT", "同步上升沿缺少 pulse_id 或时间。"
            ) from None
        if not pulse_id or timestamp < 0:
            raise ExternalImportError("INVALID_SYNC_EVENT", "同步上升沿字段无效。")
        if timestamp < start_ns or (stop_ns is not None and timestamp > stop_ns):
            # Raw pre/post-trigger samples remain immutable in the Trial, but
            # only pulses inside the formal recording window are alignment anchors.
            continue
        pulses.append(_InternalPulse(pulse_id=pulse_id, host_monotonic_ns=timestamp))

    if not pulses:
        raise ExternalImportError(
            "NO_INTERNAL_RISING_EDGE", "Trial 正式时间窗内没有同步上升沿。"
        )
    ids = [item.pulse_id for item in pulses]
    times = [item.host_monotonic_ns for item in pulses]
    if len(ids) != len(set(ids)) or any(b <= a for a, b in zip(times, times[1:])):
        raise ExternalImportError(
            "INVALID_INTERNAL_PULSES", "内部同步脉冲 ID 或时间顺序异常。"
        )
    return tuple(pulses)


_UNIT_SCALE_TO_NS: dict[str, float] = {
    "s": 1_000_000_000.0,
    "sec": 1_000_000_000.0,
    "second": 1_000_000_000.0,
    "seconds": 1_000_000_000.0,
    "ms": 1_000_000.0,
    "millisecond": 1_000_000.0,
    "milliseconds": 1_000_000.0,
    "us": 1_000.0,
    "microsecond": 1_000.0,
    "microseconds": 1_000.0,
    "ns": 1.0,
    "nanosecond": 1.0,
    "nanoseconds": 1.0,
}


def _time_scale_to_ns(request: ExternalImportRequest) -> float:
    if request.external_time_scale_to_ns is not None:
        scale = float(request.external_time_scale_to_ns)
    else:
        scale = _UNIT_SCALE_TO_NS.get(request.external_time_unit.strip().casefold(), math.nan)
    if not math.isfinite(scale) or scale <= 0:
        raise ExternalImportError(
            "UNKNOWN_EXTERNAL_TIME_UNIT",
            "未知外部时间单位需要明确提供 external_time_scale_to_ns。",
        )
    return scale


def _validate_external_pulses(values: Iterable[int | float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ExternalImportError("INVALID_EXTERNAL_PULSE", "外部脉冲时间不能是布尔值。")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ExternalImportError(
                "INVALID_EXTERNAL_PULSE", "外部脉冲时间必须是数值。"
            ) from None
        if not math.isfinite(number):
            raise ExternalImportError(
                "INVALID_EXTERNAL_PULSE", "外部脉冲时间必须是有限数值。"
            )
        result.append(number)
    if not result:
        raise ExternalImportError("NO_EXTERNAL_PULSES", "至少需要一个外部脉冲时间。")
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ExternalImportError(
            "NON_MONOTONIC_EXTERNAL_PULSES", "外部脉冲时间必须严格递增。"
        )
    return tuple(result)


def _read_csv_pulses(
    path: Path,
    *,
    column: str,
    delimiter: str | None,
    encoding: str,
    dataset_root: Path,
) -> tuple[float, ...]:
    _require_idle(dataset_root)
    try:
        with path.open("r", encoding=encoding, newline="") as stream:
            if delimiter is None:
                sample = stream.read(64 * 1024)
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    # A valid single-column CSV has no delimiter to sniff.
                    dialect = csv.excel
                reader = csv.DictReader(stream, dialect=dialect)
            else:
                reader = csv.DictReader(stream, delimiter=delimiter)
            headers = reader.fieldnames or []
            if headers.count(column) != 1:
                raise ExternalImportError(
                    "CSV_COLUMN_MISSING",
                    "CSV 必须包含且只能包含一个指定的脉冲时间列。",
                )
            values: list[float] = []
            for row_number, row in enumerate(reader, start=2):
                if row_number % 1024 == 0:
                    _require_idle(dataset_root)
                raw = row.get(column)
                if raw is None or not raw.strip():
                    continue
                try:
                    value = float(raw.strip())
                except ValueError:
                    raise ExternalImportError(
                        "INVALID_CSV_PULSE",
                        f"CSV 脉冲时间列第 {row_number} 行不是数值。",
                    ) from None
                values.append(value)
    except ExternalImportError:
        raise
    except (OSError, UnicodeError, csv.Error):
        raise ExternalImportError("CSV_READ_FAILED", "通用 CSV 脉冲列解析失败。") from None
    _require_idle(dataset_root)
    return _validate_external_pulses(values)


def _quality(anchor_count: int, rms_ns: float, max_ns: float) -> tuple[str, str]:
    if anchor_count == 1:
        return "UNAVAILABLE", "单脉冲仅估计偏移，无法从残差评价漂移。"
    if anchor_count == 2:
        return "ACCEPTABLE", "双脉冲可拟合偏移和漂移，但没有冗余锚点验证残差。"
    if rms_ns <= 1_000_000 and max_ns <= 2_000_000:
        return "GOOD", "冗余脉冲拟合残差在软阈值内。"
    if rms_ns <= 5_000_000 and max_ns <= 10_000_000:
        return "ACCEPTABLE", "冗余脉冲拟合残差偏高，建议人工复核。"
    return "POOR", "冗余脉冲拟合残差过高，不应静默视为已对齐。"


def _mapping_document(
    *,
    request: ExternalImportRequest,
    mapping_uuid: UUID,
    external_artifact_uuid: UUID,
    external_pulses: Sequence[float],
    internal_pulses: Sequence[_InternalPulse],
    trial_t0_ns: int,
    time_scale_to_ns: float,
) -> tuple[dict[str, Any], str, bool]:
    if len(external_pulses) != len(internal_pulses):
        raise ExternalImportError(
            "PULSE_COUNT_MISMATCH",
            "外部脉冲数量与 Trial 正式时间窗内的内部上升沿数量不一致。",
        )
    external_ns = np.asarray(external_pulses, dtype=np.float64) * time_scale_to_ns
    if not np.all(np.isfinite(external_ns)) or np.any(np.diff(external_ns) <= 0):
        raise ExternalImportError(
            "INVALID_NORMALIZED_PULSES", "外部脉冲换算为纳秒后无效或不再递增。"
        )
    host_pairs = [(item.pulse_id, item.host_monotonic_ns) for item in internal_pulses]
    external_pairs = [
        (internal.pulse_id, float(source_ns))
        for internal, source_ns in zip(internal_pulses, external_ns, strict=True)
    ]
    try:
        model, pulse_ids = align_shared_pulses(external_pairs, host_pairs)
    except ValueError:
        raise ExternalImportError("CLOCK_FIT_FAILED", "外部时钟模型拟合失败。") from None

    predicted = model.map(external_ns)
    targets = np.asarray(
        [item.host_monotonic_ns for item in internal_pulses], dtype=np.float64
    )
    residuals = targets - predicted
    quality, quality_reason = _quality(
        len(internal_pulses),
        model.residuals.rms_ns,
        model.residuals.max_absolute_ns,
    )
    offset_only = len(internal_pulses) == 1
    raw_scale = model.scale_a * time_scale_to_ns
    anchors = [
        {
            "ordinal": index,
            "pulse_id": internal.pulse_id,
            "external_time": float(external_time),
            "external_time_normalized_ns": float(source_ns),
            "host_monotonic_ns": internal.host_monotonic_ns,
            "trial_relative_ns": internal.host_monotonic_ns - trial_t0_ns,
            "fit_residual_ns": float(residual),
        }
        for index, (external_time, source_ns, internal, residual) in enumerate(
            zip(external_pulses, external_ns, internal_pulses, residuals, strict=True),
            start=1,
        )
    ]
    document: dict[str, Any] = {
        "schema_name": "exo-external-clock-mapping",
        "schema_version": "1.0.0",
        "mapping_uuid": str(mapping_uuid),
        "external_artifact_uuid": str(external_artifact_uuid),
        "source_clock_domain": _redact_sensitive_text(request.external_clock_domain),
        "target_clock_domain": "host_monotonic",
        "equation": "t_global_ns = scale_a * t_external + offset_b_ns",
        "external_time_unit": _redact_sensitive_text(request.external_time_unit),
        "external_time_scale_to_ns": time_scale_to_ns,
        "scale_a": raw_scale,
        "offset_b_ns": model.offset_b_ns,
        "scale_estimated": not offset_only,
        "offset_only": offset_only,
        "anchor_count": model.anchor_count,
        "valid_external_start": float(external_pulses[0]),
        "valid_external_end": float(external_pulses[-1]),
        "normalized_fit": {
            "equation": "t_global_ns = scale_a * t_external_normalized_ns + offset_b_ns",
            "scale_a": model.scale_a,
            "offset_b_ns": model.offset_b_ns,
            "algorithm_version": model.algorithm_version,
        },
        "residuals": asdict(model.residuals),
        "quality": {
            "grade": quality,
            "reason": quality_reason,
            "algorithm_version": ALIGNMENT_QUALITY_VERSION,
            "soft_thresholds_ns": {
                "good_rms": 1_000_000,
                "good_max": 2_000_000,
                "acceptable_rms": 5_000_000,
                "acceptable_max": 10_000_000,
                "minimum_redundant_anchor_count": 3,
            },
        },
        "pulse_ids": list(pulse_ids),
        "anchors": anchors,
    }
    return document, quality, offset_only


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise ExternalImportError("ANNEX_WRITE_FAILED", "外部导入包写入失败。") from None


def _write_checksums(
    root: Path,
    digests: Mapping[str, str],
) -> Path:
    output = root / "checksums.sha256"
    lines: list[str] = []
    for relative, digest in sorted(digests.items()):
        candidate = (root / Path(relative)).resolve()
        _ensure_under(
            candidate,
            root.resolve(),
            code="ANNEX_PATH_ESCAPE",
            message="外部导入包校验路径越界。",
        )
        if not candidate.is_file():
            raise ExternalImportError("ANNEX_FILE_MISSING", "外部导入包文件缺失。")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ExternalImportError("ANNEX_VERIFY_FAILED", "外部导入摘要格式无效。")
        lines.append(f"{digest}  {Path(relative).as_posix()}\n")
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise ExternalImportError("ANNEX_WRITE_FAILED", "校验清单写入失败。") from None
    return output


def _verify_annex_checksums(path: Path, *, dataset_root: Path) -> None:
    root = path.parent.resolve()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ExternalImportError("ANNEX_VERIFY_FAILED", "校验清单无法读取。") from None
    if not lines:
        raise ExternalImportError("ANNEX_VERIFY_FAILED", "校验清单为空。")
    for line in lines:
        expected, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ExternalImportError("ANNEX_VERIFY_FAILED", "校验清单格式无效。")
        candidate = (root / Path(relative)).resolve()
        _ensure_under(
            candidate,
            root,
            code="ANNEX_PATH_ESCAPE",
            message="外部导入包校验路径越界。",
        )
        if (
            not candidate.is_file()
            or _hash_file_with_idle(candidate, dataset_root) != expected
        ):
            raise ExternalImportError("ANNEX_VERIFY_FAILED", "外部导入包 SHA-256 校验失败。")


def import_external_artifact(
    request: ExternalImportRequest | Mapping[str, Any],
) -> ExternalImportResult:
    """Create and atomically publish one immutable external-artifact annex.

    No call path writes into the Trial directory.  A failed call removes its
    private build directory; a previously published annex is never overwritten.
    """

    validated = (
        request
        if isinstance(request, ExternalImportRequest)
        else ExternalImportRequest.model_validate(request)
    )
    dataset_root = validated.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise ExternalImportError("DATASET_NOT_FOUND", "数据根目录不存在。")
    _require_idle(dataset_root)
    source = _resolve_regular_source(validated.source_path, role="外部原文件")
    pulse_csv = (
        _resolve_regular_source(validated.pulse_csv_path, role="脉冲 CSV")
        if validated.pulse_csv_path is not None
        else None
    )
    manifest_path, trial_root, manifest, base_manifest_sha = _load_finalized_trial(
        validated.trial_manifest_path, dataset_root
    )
    annex_root = dataset_root / ANNEX_DIRECTORY_NAME
    # External evidence must not alias the immutable Trial package or an older annex.
    for candidate in (source, pulse_csv):
        if candidate is None:
            continue
        try:
            candidate.relative_to(trial_root)
        except ValueError:
            pass
        else:
            raise ExternalImportError(
                "SOURCE_INSIDE_TRIAL", "外部源文件不能来自目标 Trial 数据包内部。"
            )
        try:
            candidate.relative_to(annex_root)
        except ValueError:
            pass
        else:
            raise ExternalImportError(
                "SOURCE_INSIDE_ANNEX", "不能把既有外部导入包递归导入为新原文件。"
            )

    sync_artifact = _sync_artifact(manifest)
    sync_path = _resolve_manifest_artifact(trial_root, sync_artifact)
    if sync_path.stat().st_size != sync_artifact.size_bytes:
        raise ExternalImportError(
            "SYNC_INTEGRITY_FAILED", "同步 HDF5 大小与 Manifest 不一致。"
        )
    sync_sha = _hash_file_with_idle(sync_path, dataset_root)
    if sync_sha != sync_artifact.sha256:
        raise ExternalImportError(
            "SYNC_INTEGRITY_FAILED", "同步 HDF5 SHA-256 与 Manifest 不一致。"
        )
    internal_pulses = _read_internal_rising_edges(sync_path, manifest)
    _require_idle(dataset_root)

    try:
        annex_root.mkdir(parents=False, exist_ok=True)
    except OSError:
        raise ExternalImportError("ANNEX_DIRECTORY_FAILED", "无法创建外部导入目录。") from None
    if _is_link_or_reparse(annex_root):
        raise ExternalImportError("ANNEX_ROOT_ESCAPE", "外部导入目录不能是符号链接。")
    annex_root_resolved = annex_root.resolve()
    _ensure_under(
        annex_root_resolved,
        dataset_root,
        code="ANNEX_ROOT_ESCAPE",
        message="外部导入目录逃逸数据根目录。",
    )
    annex_parent = annex_root_resolved / str(manifest.trial_uuid)
    try:
        annex_parent.mkdir(parents=False, exist_ok=True)
    except OSError:
        raise ExternalImportError("ANNEX_DIRECTORY_FAILED", "无法创建 Trial 外部导入目录。") from None
    if _is_link_or_reparse(annex_parent):
        raise ExternalImportError(
            "ANNEX_PARENT_ESCAPE",
            "Trial 外部导入目录不能是符号链接或 Windows 重解析点。",
        )
    annex_parent_resolved = annex_parent.resolve()
    _ensure_under(
        annex_parent_resolved,
        annex_root_resolved,
        code="ANNEX_PARENT_ESCAPE",
        message="Trial 外部导入目录逃逸外部导入根目录。",
    )
    final_directory = annex_parent_resolved / str(validated.annex_uuid)
    build_directory = annex_parent_resolved / f".{validated.annex_uuid}.building"
    if final_directory.exists() or build_directory.exists():
        raise ExternalImportError("ANNEX_EXISTS", "该外部导入 UUID 已存在，禁止覆盖。")

    artifact_uuid = uuid4()
    mapping_uuid = uuid4()
    pulse_evidence_uuid: UUID | None = None
    build_directory_created = False
    try:
        build_directory.mkdir(parents=False, exist_ok=False)
        build_directory_created = True
        artifact_relative = f"artifacts/{artifact_uuid}{_safe_suffix(source)}"
        artifact_destination = build_directory / Path(artifact_relative)
        artifact_sha, artifact_size, artifact_audit = _copy_source(
            source, artifact_destination, dataset_root=dataset_root
        )
        files: list[AnnexFile] = [
            AnnexFile(
                artifact_uuid=artifact_uuid,
                role="external_original",
                relative_path=artifact_relative,
                media_type=_media_type(source),
                size_bytes=artifact_size,
                sha256=artifact_sha,
                source_audit=artifact_audit,
            )
        ]

        if validated.external_pulse_times is not None:
            external_pulses = _validate_external_pulses(validated.external_pulse_times)
            pulse_source: dict[str, Any] = {"kind": "user_provided_values"}
        else:
            csv_source = pulse_csv or source
            if csv_source == source:
                csv_copy = artifact_destination
                pulse_source_artifact_uuid = artifact_uuid
            else:
                pulse_evidence_uuid = uuid4()
                pulse_relative = f"evidence/{pulse_evidence_uuid}{_safe_suffix(csv_source)}"
                csv_copy = build_directory / Path(pulse_relative)
                pulse_sha, pulse_size, pulse_audit = _copy_source(
                    csv_source, csv_copy, dataset_root=dataset_root
                )
                files.append(
                    AnnexFile(
                        artifact_uuid=pulse_evidence_uuid,
                        role="pulse_evidence",
                        relative_path=pulse_relative,
                        media_type="text/csv",
                        size_bytes=pulse_size,
                        sha256=pulse_sha,
                        source_audit=pulse_audit,
                    )
                )
                pulse_source_artifact_uuid = pulse_evidence_uuid
            assert validated.pulse_csv_column is not None
            external_pulses = _read_csv_pulses(
                csv_copy,
                column=validated.pulse_csv_column,
                delimiter=validated.csv_delimiter,
                encoding=validated.csv_encoding,
                dataset_root=dataset_root,
            )
            pulse_source = {
                "kind": "csv_column",
                "artifact_uuid": str(pulse_source_artifact_uuid),
                "column": _redact_sensitive_text(validated.pulse_csv_column),
                "delimiter": validated.csv_delimiter or "auto",
                "encoding": validated.csv_encoding,
            }

        time_scale = _time_scale_to_ns(validated)
        mapping_document, quality, offset_only = _mapping_document(
            request=validated,
            mapping_uuid=mapping_uuid,
            external_artifact_uuid=artifact_uuid,
            external_pulses=external_pulses,
            internal_pulses=internal_pulses,
            trial_t0_ns=manifest.timing.start_host_monotonic_ns,
            time_scale_to_ns=time_scale,
        )
        mapping_document["pulse_source"] = pulse_source
        mapping_path = build_directory / "alignment/mapping.json"
        _write_json(mapping_path, mapping_document)
        mapping_sha = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
        mapping_size = mapping_path.stat().st_size

        imported_at = datetime.now(timezone.utc)
        base_relative = manifest_path.relative_to(dataset_root).as_posix()
        annex_manifest = ExternalAnnexManifest(
            annex_uuid=validated.annex_uuid,
            trial_uuid=manifest.trial_uuid,
            base_manifest_uuid=manifest.manifest_uuid,
            base_manifest_schema_version=manifest.schema_version,
            base_manifest_relative_path=base_relative,
            base_manifest_sha256=base_manifest_sha,
            modality=validated.modality,
            other_modality_label=(
                _redact_sensitive_text(validated.other_modality_label)
                if validated.other_modality_label
                else None
            ),
            source_system=_redact_sensitive_text(validated.source_system),
            external_clock_domain=_redact_sensitive_text(validated.external_clock_domain),
            imported_at_utc=imported_at,
            files=tuple(files),
            mapping=MappingReference(
                mapping_uuid=mapping_uuid,
                size_bytes=mapping_size,
                sha256=mapping_sha,
                quality=quality,
                offset_only=offset_only,
                anchor_count=len(internal_pulses),
            ),
        )
        annex_manifest_path = build_directory / "annex_manifest.json"
        _write_json(annex_manifest_path, annex_manifest.model_dump(mode="json"))

        checksum_digests = {item.relative_path: item.sha256 for item in files}
        checksum_digests["alignment/mapping.json"] = mapping_sha
        checksum_digests["annex_manifest.json"] = hashlib.sha256(
            annex_manifest_path.read_bytes()
        ).hexdigest()
        checksum_path = _write_checksums(build_directory, checksum_digests)
        _verify_annex_checksums(checksum_path, dataset_root=dataset_root)

        # Re-check every immutable binding immediately before publication.
        _require_idle(dataset_root)
        if _hash_file_with_idle(manifest_path, dataset_root) != base_manifest_sha:
            raise ExternalImportError(
                "BASE_MANIFEST_CHANGED", "导入期间基准 Manifest 发生变化。"
            )
        if _hash_file_with_idle(sync_path, dataset_root) != sync_sha:
            raise ExternalImportError(
                "SYNC_CHANGED", "导入期间同步 HDF5 发生变化。"
            )
        if final_directory.exists():
            raise ExternalImportError("ANNEX_EXISTS", "目标外部导入包已存在。")
        os.replace(build_directory, final_directory)
        build_directory_created = False
    except ExternalImportError:
        raise
    except Exception as exc:
        raise ExternalImportError(
            "IMPORT_FAILED",
            f"外部模态导入失败（{type(exc).__name__}）；未发布不完整数据包。",
        ) from None
    finally:
        if build_directory_created:
            shutil.rmtree(build_directory, ignore_errors=True)
        # Do not leave empty hierarchy after a failed import.
        for empty in (annex_parent, annex_root):
            try:
                empty.rmdir()
            except OSError:
                pass

    return ExternalImportResult(
        annex_uuid=validated.annex_uuid,
        trial_uuid=manifest.trial_uuid,
        annex_directory=final_directory,
        annex_manifest_path=final_directory / "annex_manifest.json",
        mapping_path=final_directory / "alignment/mapping.json",
        copied_artifact_path=final_directory / Path(artifact_relative),
        base_manifest_sha256=base_manifest_sha,
        copied_artifact_sha256=artifact_sha,
        quality=quality,
        offset_only=offset_only,
        anchor_count=len(internal_pulses),
    )


__all__ = [
    "ALIGNMENT_QUALITY_VERSION",
    "ANNEX_DIRECTORY_NAME",
    "ANNEX_SCHEMA_VERSION",
    "AnnexFile",
    "ExternalAnnexManifest",
    "ExternalImportError",
    "ExternalImportRequest",
    "ExternalImportResult",
    "ExternalModality",
    "MappingReference",
    "SourceAudit",
    "import_external_artifact",
]
