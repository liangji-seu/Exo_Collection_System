"""Canonical Trial directories and atomic publication primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID


TRIAL_SUBDIRECTORIES = (
    "raw/external",
    "derived/preview",
    "reports",
    "logs",
)


def safe_relative_path(value: str | Path) -> PurePosixPath:
    text = str(value).replace("\\", "/")
    relative = PurePosixPath(text)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Expected a safe Trial-relative path, got {value!r}")
    if ":" in relative.parts[0]:
        raise ValueError(f"Drive-qualified paths are forbidden: {value!r}")
    return relative


@dataclass(frozen=True, slots=True)
class TrialLayout:
    dataset_root: Path
    project_uuid: UUID
    subject_uuid: UUID
    session_uuid: UUID
    trial_uuid: UUID

    @classmethod
    def build(
        cls,
        dataset_root: str | Path,
        project_uuid: UUID,
        subject_uuid: UUID,
        session_uuid: UUID,
        trial_uuid: UUID,
    ) -> TrialLayout:
        return cls(Path(dataset_root).expanduser().resolve(), project_uuid, subject_uuid, session_uuid, trial_uuid)

    @property
    def session_directory(self) -> Path:
        return self.dataset_root / str(self.project_uuid) / str(self.subject_uuid) / str(self.session_uuid)

    @property
    def trials_directory(self) -> Path:
        return self.session_directory / "trials"

    @property
    def recording_directory(self) -> Path:
        return self.trials_directory / f"{self.trial_uuid}.recording"

    @property
    def final_directory(self) -> Path:
        return self.trials_directory / str(self.trial_uuid)

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
        partials = list(self.recording_directory.rglob("*.partial"))
        if partials:
            display = ", ".join(str(path.relative_to(self.recording_directory)) for path in partials[:5])
            raise RuntimeError(f"Trial still contains partial files: {display}")
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
    return sorted(path for path in root.rglob("*.recording") if path.is_dir()) if root.exists() else []


def iter_finalized_manifest_paths(dataset_root: str | Path) -> list[Path]:
    """Return only published manifests; never inspect active recording directories."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("manifest.json")
        if path.is_file() and not any(part.endswith(".recording") for part in path.parts)
    )

