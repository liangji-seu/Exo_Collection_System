"""Artifact publication and all-or-nothing Trial package finalization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from exo_collection.domain.models import ArtifactKind, UTCDateTime, utc_now
from exo_collection.storage.manifest import (
    ManifestArtifact,
    TrialManifest,
    load_manifest,
    save_manifest,
)

from .checksum import sha256_file, verify_checksum_manifest, write_checksum_manifest
from .layout import TrialLayout, safe_relative_path


@dataclass(frozen=True, slots=True)
class ArtifactDraft:
    trial_uuid: UUID
    modality: str
    kind: ArtifactKind
    media_type: str
    relative_path: str
    artifact_uuid: UUID = field(default_factory=uuid4)
    created_at_utc: datetime = field(default_factory=utc_now)
    source_artifact_uuids: tuple[UUID, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)


def publish_json(layout: TrialLayout, relative_path: str, document: object) -> Path:
    partial = layout.partial_path(relative_path)
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return layout.publish_partial(relative_path)


def publish_artifact(layout: TrialLayout, draft: ArtifactDraft, finalized_at_utc: UTCDateTime) -> ManifestArtifact:
    """Rename a closed partial file and capture immutable integrity metadata."""

    path = layout.path(draft.relative_path)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        path = layout.publish_partial(draft.relative_path)
    elif not path.is_file():
        raise FileNotFoundError(f"Artifact is neither partial nor published: {draft.relative_path}")
    return ManifestArtifact(
        artifact_uuid=draft.artifact_uuid,
        trial_uuid=draft.trial_uuid,
        modality=draft.modality,
        kind=draft.kind,
        media_type=draft.media_type,
        relative_path=draft.relative_path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        created_at_utc=draft.created_at_utc,
        finalized_at_utc=finalized_at_utc,
        source_artifact_uuids=list(draft.source_artifact_uuids),
        metadata=draft.metadata,
        immutable=True,
    )


def validate_artifact_integrity(layout: TrialLayout, artifacts: list[ManifestArtifact]) -> None:
    for artifact in artifacts:
        path = layout.path(artifact.relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact.size_bytes:
            raise RuntimeError(f"Artifact size changed during finalization: {artifact.relative_path}")
        if sha256_file(path) != artifact.sha256:
            raise RuntimeError(f"Artifact digest changed during finalization: {artifact.relative_path}")


def finalize_trial_package(layout: TrialLayout, manifest: TrialManifest) -> Path:
    """Publish Manifest/checksums then atomically remove the `.recording` suffix."""

    if manifest.trial_uuid != layout.trial_uuid:
        raise ValueError("Manifest Trial UUID does not match TrialLayout")
    validate_artifact_integrity(layout, manifest.artifacts)
    manifest_path = layout.path("manifest.json")
    if manifest_path.exists():
        # A crash can occur after Manifest publication but before the atomic
        # directory rename. Retrying is safe only for byte-equivalent content.
        existing = load_manifest(manifest_path)
        if existing != manifest:
            raise FileExistsError(f"conflicting Manifest already exists: {manifest_path}")
    else:
        save_manifest(manifest_path, manifest)
    relative_paths = [artifact.relative_path for artifact in manifest.artifacts]
    relative_paths.append("manifest.json")
    checksums = write_checksum_manifest(layout.recording_directory, relative_paths)
    results = verify_checksum_manifest(checksums)
    if not results or not all(results.values()):
        raise RuntimeError("Trial checksum validation failed before publication")
    return layout.finalize_directory()
