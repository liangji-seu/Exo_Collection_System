"""Append-only human quality reviews kept outside immutable Trial packages."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exo_collection.domain.models import QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import read_activity
from exo_collection.storage.checksum import sha256_file
from exo_collection.storage.layout import (
    name_has_storage_suffix,
    path_has_unpublished_component,
)
from exo_collection.storage.manifest import load_manifest


QUALITY_REVIEW_DIRECTORY = ".studio-records/quality-reviews"


class QualityReviewError(RuntimeError):
    """A review cannot be safely loaded or appended."""


class QualityReviewRecord(BaseModel):
    """One immutable human decision chained to the previous review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    review_uuid: UUID = Field(default_factory=uuid4)
    trial_uuid: UUID
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    computed_grade: QualityGrade | None = None
    reviewed_grade: QualityGrade
    reviewer: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=4000)
    reviewed_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    previous_record_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("reviewer", "reason")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.strip().split()) if "\n" not in value else value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("reviewed_at_utc")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


class SavedQualityReview(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    record: QualityReviewRecord
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _require_idle(data_root: Path) -> None:
    if read_activity(data_root) is not None:
        raise QualityReviewError("检测到 Collector 正在采集，已禁止修改人工审核记录。")


def _validated_trial(
    data_root: str | Path,
    manifest_path: str | Path,
) -> tuple[Path, Path, Any]:
    root = Path(data_root).expanduser().resolve()
    path = Path(manifest_path).expanduser().resolve()
    if path_has_unpublished_component(path):
        raise QualityReviewError("不能审核 .recording/.partial Trial。")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise QualityReviewError("Manifest 不在当前数据根目录中。") from exc
    if path.name != "manifest.json" or not path.is_file():
        raise QualityReviewError("请选择有效的 Trial manifest.json。")
    _require_idle(root)
    manifest = load_manifest(path)
    if manifest.state is not TrialState.FINALIZED:
        raise QualityReviewError("只允许审核 FINALIZED Trial。")
    return root, path, manifest


def _review_directory(root: Path, trial_uuid: UUID) -> Path:
    return root / QUALITY_REVIEW_DIRECTORY / str(trial_uuid)


@contextmanager
def _review_lock(directory: Path) -> Iterator[None]:
    """Serialize the read-head/append transaction across Data Studio processes."""

    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".append.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised by non-Windows CI
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def list_quality_reviews(
    data_root: str | Path,
    manifest_path: str | Path,
) -> tuple[SavedQualityReview, ...]:
    """Load and validate every append-only review for a finalized Trial."""

    root, path, manifest = _validated_trial(data_root, manifest_path)
    expected_manifest_sha = sha256_file(path)
    directory = _review_directory(root, manifest.trial_uuid)
    if not directory.is_dir():
        return ()
    result: list[SavedQualityReview] = []
    previous_sha: str | None = None
    for review_path in sorted(directory.glob("*.json")):
        if review_path.name.startswith(".") or name_has_storage_suffix(
            review_path.name
        ):
            continue
        try:
            payload = review_path.read_text(encoding="utf-8")
            record = QualityReviewRecord.model_validate_json(payload)
        except Exception as exc:
            raise QualityReviewError(
                f"人工审核记录无法验证：{review_path.name}: {exc}"
            ) from exc
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if not review_path.stem.endswith(f"-{digest}"):
            raise QualityReviewError("人工审核记录内容与文件摘要不匹配。")
        if record.trial_uuid != manifest.trial_uuid:
            raise QualityReviewError("人工审核记录的 Trial UUID 不匹配。")
        if record.manifest_sha256 != expected_manifest_sha:
            raise QualityReviewError("人工审核记录引用的 Manifest 已变化。")
        if record.previous_record_sha256 != previous_sha:
            raise QualityReviewError("人工审核链不连续，拒绝隐藏中间记录。")
        result.append(SavedQualityReview(record=record, path=review_path, sha256=digest))
        previous_sha = digest
    _require_idle(root)
    return tuple(result)


def latest_quality_review(
    data_root: str | Path,
    manifest_path: str | Path,
) -> SavedQualityReview | None:
    records = list_quality_reviews(data_root, manifest_path)
    return records[-1] if records else None


def append_quality_review(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    reviewed_grade: QualityGrade | str,
    reviewer: str,
    reason: str,
) -> SavedQualityReview:
    """Atomically append a human review without rewriting the Trial Manifest."""

    root, path, initial_manifest = _validated_trial(data_root, manifest_path)
    directory = _review_directory(root, initial_manifest.trial_uuid)
    with _review_lock(directory):
        # The chain head must be loaded only after acquiring the interprocess
        # lock. Otherwise two Data Studio instances can publish sibling records
        # with the same previous hash and make the append-only ledger invalid.
        root, path, manifest = _validated_trial(root, path)
        previous = list_quality_reviews(root, path)
        record = QualityReviewRecord(
            trial_uuid=manifest.trial_uuid,
            manifest_sha256=sha256_file(path),
            computed_grade=manifest.quality.computed_grade,
            reviewed_grade=QualityGrade(reviewed_grade),
            reviewer=reviewer,
            reason=reason,
            previous_record_sha256=previous[-1].sha256 if previous else None,
        )
        document = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        # ISO timestamps sort chronologically; the digest in the immutable file
        # name lets readers detect an edited record even when it is the newest one.
        timestamp = record.reviewed_at_utc.strftime("%Y%m%dT%H%M%S.%fZ")
        destination = directory / f"{timestamp}-{record.review_uuid}-{digest}.json"
        temporary = destination.with_name(f".{destination.name}.partial")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            _require_idle(root)
            # Recheck the immutable anchor immediately before publication.
            if sha256_file(path) != record.manifest_sha256:
                raise QualityReviewError("Manifest 在审核期间发生变化，记录未发布。")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return SavedQualityReview(
            record=record,
            path=destination,
            sha256=digest,
        )


__all__ = [
    "QUALITY_REVIEW_DIRECTORY",
    "QualityReviewError",
    "QualityReviewRecord",
    "SavedQualityReview",
    "append_quality_review",
    "latest_quality_review",
    "list_quality_reviews",
]
