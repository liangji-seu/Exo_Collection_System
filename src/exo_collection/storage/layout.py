"""Canonical Trial directories and atomic publication primitives."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from exo_collection.domain.models import normalize_relative_path


TRIAL_SUBDIRECTORIES = (
    "raw/external",
    "derived/preview",
    "reports",
    "logs",
)

# NTFS is normally case-insensitive, so every trust boundary must treat these
# lifecycle suffixes identically regardless of spelling.  Keeping the leading
# dot and checking only the end of one path component avoids classifying names
# such as ``recording-notes.txt`` or ``trial.partial.backup`` as package state.
UNPUBLISHED_STORAGE_SUFFIXES = (
    ".recording",
    ".partial",
    ".aborted",
    ".building",
)


def name_has_storage_suffix(
    name: str,
    suffixes: tuple[str, ...] = UNPUBLISHED_STORAGE_SUFFIXES,
) -> bool:
    """Apply Windows-style case-insensitive suffix semantics to one name."""

    folded = name.casefold()
    return any(folded.endswith(suffix.casefold()) for suffix in suffixes)


def path_has_unpublished_component(path: str | Path) -> bool:
    """Return whether any complete path component has a reserved state suffix."""

    return any(name_has_storage_suffix(part) for part in Path(path).parts)


def safe_relative_path(value: str | Path) -> PurePosixPath:
    try:
        return PurePosixPath(normalize_relative_path(str(value)))
    except ValueError as exc:
        raise ValueError(f"Expected a safe Trial-relative path, got {value!r}") from exc


def _safe_path_segment(value: str) -> str:
    """Replace characters unsafe for directory names with underscores."""
    return re.sub(r"[<>:\"/\\|?*\x00-\x1f ]", "_", value).strip("_") or "_"


def _condition_level_segment(level: int | str | None) -> str:
    """Normalize a condition level value into a safe directory segment."""
    if level is None:
        return "L"
    text = str(level).strip()
    return _safe_path_segment(text) if text else "L"


@dataclass(frozen=True, slots=True)
class TrialLayout:
    dataset_root: Path
    project_uuid: UUID
    subject_uuid: UUID
    session_uuid: UUID
    trial_uuid: UUID
    project_partition: str | None = None
    subject_code: str | None = None
    condition_code: str | None = None
    condition_level: int | str | None = None
    repeat_index: int | None = None

    @classmethod
    def build(
        cls,
        dataset_root: str | Path,
        project_uuid: UUID,
        subject_uuid: UUID,
        session_uuid: UUID,
        trial_uuid: UUID,
        project_partition: str | None = None,
        subject_code: str | None = None,
        condition_code: str | None = None,
        condition_level: int | str | None = None,
        repeat_index: int | None = None,
    ) -> TrialLayout:
        partition = None
        if project_partition is not None:
            partition = project_partition.strip().upper()
            if partition not in {"F", "T"}:
                raise ValueError("project_partition must be 'F' or 'T'")
        readable_subject = None
        if subject_code is not None:
            readable_subject = subject_code.strip()
            if re.fullmatch(r"\d{3}", readable_subject) is None:
                raise ValueError("subject_code must contain exactly three digits")
        safe_condition = None
        if condition_code is not None:
            safe_condition = _safe_path_segment(condition_code.strip())
            if not safe_condition:
                safe_condition = None
        return cls(
            Path(dataset_root).expanduser().resolve(),
            project_uuid,
            subject_uuid,
            session_uuid,
            trial_uuid,
            partition,
            readable_subject,
            safe_condition,
            condition_level,
            repeat_index,
        )

    @property
    def session_directory(self) -> Path:
        project_directory = self.project_partition or str(self.project_uuid)
        subject_directory = self.subject_code or str(self.subject_uuid)
        return (
            self.dataset_root
            / project_directory
            / subject_directory
            / str(self.session_uuid)
        )

    @property
    def trials_directory(self) -> Path:
        return self.session_directory / "trials"

    @property
    def _trial_leaf_name(self) -> str:
        """Build the human-readable leaf directory name for this trial.

        When all three grouping fields are available the name is
        ``{condition_code}/{level_segment}/{repeat_index}``.
        When any field is missing the leaf falls back to the trial UUID.
        """
        if self.condition_code and self.repeat_index is not None:
            level_seg = _condition_level_segment(self.condition_level)
            return f"{self.condition_code}/{level_seg}/{self.repeat_index}"
        return str(self.trial_uuid)

    @property
    def _resolved_leaf_name(self) -> str:
        """Leaf name with a short UUID discriminator when a finalized trial
        already occupies the human-readable path."""
        leaf = self._trial_leaf_name
        base = self.trials_directory / leaf
        if base.exists():
            short = str(self.trial_uuid)[:8]
            return f"{leaf}_{short}"
        return leaf

    @property
    def recording_directory(self) -> Path:
        leaf = self._resolved_leaf_name
        parent = self.trials_directory.joinpath(leaf).parent
        return parent / f"{Path(leaf).name}.recording"

    @property
    def final_directory(self) -> Path:
        return self.trials_directory / self._resolved_leaf_name

    def create_recording(self) -> Path:
        if self.final_directory.exists():
            raise FileExistsError(f"Finalized Trial already exists: {self.final_directory}")
        self.recording_directory.mkdir(parents=True, exist_ok=False)
        for relative in TRIAL_SUBDIRECTORIES:
            (self.recording_directory / relative).mkdir(parents=True, exist_ok=True)
        return self.recording_directory

    def path(self, relative: str | Path, *, final: bool = False) -> Path:
        safe = safe_relative_path(relative)
        base = self.final_directory if final else self.recording_directory
        return base.joinpath(*safe.parts)

    def partial_path(self, relative: str | Path) -> Path:
        final_path = self.path(relative)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return final_path.with_name(final_path.name + ".partial")

    def publish_partial(self, relative: str | Path) -> Path:
        destination = self.path(relative)
        source = destination.with_name(destination.name + ".partial")
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(source, destination)
        return destination

    def assert_ready_to_finalize(self) -> None:
        if not self.recording_directory.is_dir():
            raise FileNotFoundError(self.recording_directory)
        unpublished = [
            path
            for path in self.recording_directory.rglob("*")
            if name_has_storage_suffix(path.name)
        ]
        if unpublished:
            display = ", ".join(
                str(path.relative_to(self.recording_directory))
                for path in unpublished[:5]
            )
            raise RuntimeError(f"Trial still contains unpublished paths: {display}")
        if not (self.recording_directory / "manifest.json").is_file():
            raise RuntimeError("manifest.json must be published before finalizing a Trial")
        if not (self.recording_directory / "checksums.sha256").is_file():
            raise RuntimeError("checksums.sha256 must be published before finalizing a Trial")

    def finalize_directory(self) -> Path:
        """Atomically publish a closed Trial on the same filesystem."""

        self.assert_ready_to_finalize()
        if self.final_directory.exists():
            raise FileExistsError(self.final_directory)
        os.replace(self.recording_directory, self.final_directory)
        return self.final_directory


def iter_recording_directories(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    return (
        sorted(
            path
            for path in root.rglob("*")
            if name_has_storage_suffix(path.name, (".recording",))
            and path.is_dir()
        )
        if root.exists()
        else []
    )


def iter_aborted_directories(dataset_root: str | Path) -> list[Path]:
    """Return retained recovery packages explicitly marked ``.aborted``."""

    root = Path(dataset_root).expanduser().resolve()
    return (
        sorted(
            path
            for path in root.rglob("*")
            if name_has_storage_suffix(path.name, (".aborted",))
            and path.is_dir()
        )
        if root.exists()
        else []
    )


def iter_finalized_manifest_paths(dataset_root: str | Path) -> list[Path]:
    """Return only published manifests; never inspect active recording directories."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("manifest.json")
        if path.is_file()
        and not path_has_unpublished_component(path)
    )
