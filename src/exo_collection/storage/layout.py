"""Canonical Trial directories and atomic publication primitives."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from exo_collection.domain.models import normalize_relative_path


# Internal bookkeeping files live under a single hidden subdirectory so the
# operator sees only data artefacts when browsing a trial folder.
EXO_INTERNAL_DIR = ".exo"

TRIAL_SUBDIRECTORIES = (EXO_INTERNAL_DIR,)

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


@dataclass(frozen=True, slots=True)
class TrialLayout:
    """Canonical directory structure for one Trial.

    The operator-visible hierarchy is::

        {data_root}/{project}/{subject}/{condition}/session{repeat}_{timestamp}/

    All internal bookkeeping (manifest, checksums, logs, quality reports,
    session metadata) lives under a single ``.exo/`` subdirectory inside the
    trial folder so the operator sees only data files when browsing.
    """

    dataset_root: Path
    project_uuid: UUID
    subject_uuid: UUID
    session_uuid: UUID
    trial_uuid: UUID
    project_partition: str | None = None
    subject_code: str | None = None
    condition_code: str | None = None
    repeat_index: int | None = None
    started_at_utc: datetime | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

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
        repeat_index: int | None = None,
        started_at_utc: datetime | None = None,
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
            repeat_index,
            started_at_utc,
        )

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @property
    def exo_root(self) -> Path:
        """Global Exo bookkeeping directory (catalog, activity guard)."""
        return self.dataset_root / EXO_INTERNAL_DIR

    @property
    def subject_directory(self) -> Path:
        """Top-level grouping: ``{root}/{project}/{subject}/``."""
        project_dir = self.project_partition or str(self.project_uuid)
        subject_dir = self.subject_code or str(self.subject_uuid)
        return self.dataset_root / project_dir / subject_dir

    @property
    def _trial_leaf_name(self) -> str:
        """Human-readable leaf name: ``{condition}/session{repeat}_{timestamp}``.

        The timestamp suffix (``YYYYmmdd_HHMMSS``) makes every trial directory
        unique without resorting to UUIDs.  Falls back to the trial UUID when
        condition or repeat is missing.
        """
        if self.condition_code and self.repeat_index is not None:
            ts = (
                self.started_at_utc or datetime.now(timezone.utc)
            ).strftime("%Y%m%d_%H%M%S")
            return f"{self.condition_code}/session{self.repeat_index}_{ts}"
        return str(self.trial_uuid)

    @property
    def recording_directory(self) -> Path:
        """Active trial directory (``.recording`` suffix on the leaf)."""
        leaf = self._trial_leaf_name
        parent = self.subject_directory.joinpath(leaf).parent
        return parent / f"{Path(leaf).name}.recording"

    @property
    def final_directory(self) -> Path:
        """Published trial directory (no suffix)."""
        return self.subject_directory / self._trial_leaf_name

    @property
    def exo_directory(self) -> Path:
        """Internal bookkeeping subdirectory inside the trial."""
        return self.recording_directory / EXO_INTERNAL_DIR

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

    def exo_path(self, relative: str | Path) -> Path:
        """Path to a file inside the ``.exo/`` internal directory."""
        safe = safe_relative_path(relative)
        return self.exo_directory.joinpath(*safe.parts)

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
        manifest_path = self.exo_path("manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError("manifest.json must be published before finalizing a Trial")
        checksums_path = self.exo_path("checksums.sha256")
        if not checksums_path.is_file():
            raise RuntimeError("checksums.sha256 must be published before finalizing a Trial")

    def finalize_directory(self) -> Path:
        """Atomically publish a closed Trial on the same filesystem."""

        self.assert_ready_to_finalize()
        destination = self.final_directory
        if destination.exists():
            raise FileExistsError(destination)
        recording = self.recording_directory
        os.replace(recording, destination)
        return destination


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
