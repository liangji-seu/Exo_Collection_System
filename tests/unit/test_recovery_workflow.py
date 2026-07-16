from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.apps.data_studio.recovery_service import RecoveryBackgroundService
from exo_collection.domain.models import ArtifactKind, Condition
from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file, write_checksum_manifest
from exo_collection.storage.layout import (
    TrialLayout,
    iter_aborted_directories,
    iter_finalized_manifest_paths,
)
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    save_manifest,
)
from exo_collection.storage.recovery_manager import (
    RecoveryConfirmationRequiredError,
    UnsafeRecoveryDecisionError,
    abort_recording_preserving_data,
    discover_recoverable_trials,
    finalize_prepared_recording,
    inspect_recording_directory,
    repair_recording_directory,
)
from exo_collection.writers.binary_block import BLOCK_HEADER_SIZE, BlockBinaryWriter


UTC = timezone.utc
BASE_TIME = datetime(2026, 7, 16, 0, 0, tzinfo=UTC)


def _layout(tmp_path: Path) -> TrialLayout:
    return TrialLayout.build(tmp_path, uuid4(), uuid4(), uuid4(), uuid4())


def _write_prepared_package(layout: TrialLayout, *, complete_checksums: bool = True) -> bytes:
    layout.create_recording()
    raw = layout.recording_directory / "raw/sensor.bin"
    raw.parent.mkdir(parents=True, exist_ok=True)
    payload = b"closed immutable sensor payload\x00\x01"
    raw.write_bytes(payload)
    artifact = ManifestArtifact(
        artifact_uuid=uuid4(),
        trial_uuid=layout.trial_uuid,
        modality="sensor",
        kind=ArtifactKind.RAW,
        media_type="application/octet-stream",
        relative_path="raw/sensor.bin",
        size_bytes=len(payload),
        sha256=sha256_file(raw),
        created_at_utc=BASE_TIME,
        finalized_at_utc=BASE_TIME + timedelta(seconds=2),
    )
    manifest = TrialManifest(
        project_uuid=layout.project_uuid,
        project_code="T",
        project_name="Test",
        subject_uuid=layout.subject_uuid,
        subject_code="001",
        session_uuid=layout.session_uuid,
        trial_uuid=layout.trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code="TEST",
            condition_name="recovery test",
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=BASE_TIME,
        ),
        timing=TrialTiming(
            started_at_utc=BASE_TIME,
            stopped_at_utc=BASE_TIME + timedelta(seconds=1),
            finalized_at_utc=BASE_TIME + timedelta(seconds=2),
            start_host_monotonic_ns=100,
            stop_host_monotonic_ns=200,
            finalize_host_monotonic_ns=300,
        ),
        software=SoftwareProvenance(
            application="test",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="test-commit",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="a" * 64,
        ),
        artifacts=[artifact],
    )
    save_manifest(layout.recording_directory / "manifest.json", manifest)
    checksum_paths = ["manifest.json", "raw/sensor.bin"] if complete_checksums else ["manifest.json"]
    write_checksum_manifest(layout.recording_directory, checksum_paths)
    return payload


def test_discovery_is_read_only_and_active_lock_skips_payload_inspection(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.create_recording()
    source = layout.partial_path("raw/ultrasound.bin")
    with BlockBinaryWriter(
        source,
        dtype="uint16",
        sample_shape=(4,),
        metadata={"clock_domain": "test"},
    ) as writer:
        writer.append(np.ones((2, 4), dtype=np.uint16), host_monotonic_ns=10)
    before = source.read_bytes()

    discovered = discover_recoverable_trials(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].ultrasound is not None
    assert source.read_bytes() == before

    with AcquisitionLock(tmp_path, layout.trial_uuid):
        occupied = inspect_recording_directory(layout.recording_directory)
        assert occupied.active_collection
        assert occupied.active_trial_uuid == str(layout.trial_uuid)
        assert occupied.ultrasound is None
        assert occupied.hdf5_files == ()
        assert occupied.allowed_actions == ()
        with pytest.raises(FileExistsError, match="collector lock"):
            abort_recording_preserving_data(
                layout.recording_directory,
                reason="operator rejected incomplete acquisition",
                confirmed=True,
            )
    assert source.read_bytes() == before


def test_recovery_discovers_and_finalizes_mixed_case_recording_suffix(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _write_prepared_package(layout)
    uppercase = layout.recording_directory.with_name(
        f"{layout.trial_uuid}.RECORDING"
    )
    layout.recording_directory.rename(uppercase)

    reports = discover_recoverable_trials(tmp_path)

    assert len(reports) == 1
    assert reports[0].recording_directory == uppercase
    assert reports[0].can_finalize
    result = finalize_prepared_recording(uppercase, confirmed=True)
    assert result.destination_directory.name == str(layout.trial_uuid)
    assert result.destination_directory.is_dir()


def test_recovery_refuses_prepared_package_with_mixed_case_private_descendant(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _write_prepared_package(layout)
    private = layout.recording_directory / "reports" / "cache.BUILDING"
    private.mkdir()

    report = inspect_recording_directory(layout.recording_directory)

    assert private in report.partial_files
    assert not report.can_finalize


def test_possible_middle_corruption_is_never_truncated(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.create_recording()
    source = layout.partial_path("raw/ultrasound.bin")
    batch = np.arange(8, dtype=np.uint16).reshape(2, 4)
    with BlockBinaryWriter(
        source,
        dtype="uint16",
        sample_shape=(4,),
        metadata={"clock_domain": "test"},
    ) as writer:
        writer.append(batch, host_monotonic_ns=10)
        writer.append(batch + 10, host_monotonic_ns=20)
    damaged = bytearray(source.read_bytes())
    damaged[BLOCK_HEADER_SIZE] ^= 0xFF
    source.write_bytes(damaged)
    before = source.read_bytes()

    report = inspect_recording_directory(layout.recording_directory)
    assert report.ultrasound is not None
    assert report.ultrasound.intermediate_corruption
    assert not report.can_repair
    with pytest.raises(UnsafeRecoveryDecisionError, match="safe tail repair"):
        repair_recording_directory(layout.recording_directory)

    assert source.read_bytes() == before
    assert not list((layout.recording_directory / "reports").glob("recovery-*.json"))


def test_repair_rejects_hard_link_alias_without_touching_external_bytes(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    layout.create_recording()
    external = tmp_path / "external-ultrasound.bin.partial"
    with BlockBinaryWriter(
        external,
        dtype="uint16",
        sample_shape=(4,),
        metadata={"clock_domain": "test"},
    ) as writer:
        writer.append(np.arange(8, dtype=np.uint16).reshape(2, 4), host_monotonic_ns=10)
    with external.open("ab") as stream:
        stream.write(b"incomplete-tail")
    external_before = external.read_bytes()

    aliased = layout.partial_path("raw/ultrasound.bin")
    os.link(external, aliased)
    report = inspect_recording_directory(layout.recording_directory)
    assert report.can_repair

    with pytest.raises(UnsafeRecoveryDecisionError, match="hard-linked"):
        repair_recording_directory(layout.recording_directory)

    assert external.read_bytes() == external_before
    assert aliased.read_bytes() == external_before
    assert not list((layout.recording_directory / "reports").glob("recovery-*.json"))


def test_prepared_package_can_only_be_finalized_after_complete_proof_and_confirmation(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    payload = _write_prepared_package(layout)
    report = inspect_recording_directory(layout.recording_directory)
    assert report.prepared_publication is not None
    assert report.prepared_publication.ready
    assert report.can_finalize

    with pytest.raises(RecoveryConfirmationRequiredError):
        finalize_prepared_recording(layout.recording_directory)
    assert layout.recording_directory.is_dir()

    result = finalize_prepared_recording(
        layout.recording_directory,
        confirmed=True,
        confirmed_by="local recovery operator",
    )
    assert result.destination_directory == layout.final_directory
    assert result.audit_path.is_file()
    assert not layout.recording_directory.exists()
    assert (layout.final_directory / "raw/sensor.bin").read_bytes() == payload
    assert iter_finalized_manifest_paths(tmp_path) == [layout.final_directory / "manifest.json"]


def test_incomplete_checksum_contract_cannot_be_disguised_as_finalized(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_prepared_package(layout, complete_checksums=False)
    report = inspect_recording_directory(layout.recording_directory)
    assert report.prepared_publication is not None
    assert not report.prepared_publication.ready
    assert not report.can_finalize

    with pytest.raises(UnsafeRecoveryDecisionError, match="refusing to publish"):
        finalize_prepared_recording(layout.recording_directory, confirmed=True)
    assert layout.recording_directory.is_dir()
    assert not layout.final_directory.exists()


def test_prepared_package_with_hard_link_alias_cannot_be_finalized(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    payload = _write_prepared_package(layout)
    raw = layout.recording_directory / "raw/sensor.bin"
    external = tmp_path / "external-sensor.bin"
    external.write_bytes(payload)
    raw.unlink()
    os.link(external, raw)

    report = inspect_recording_directory(layout.recording_directory)
    assert report.prepared_publication is not None
    assert not report.prepared_publication.ready
    assert not report.can_finalize
    assert any("hard-linked" in reason for reason in report.issues)

    with pytest.raises(UnsafeRecoveryDecisionError, match="refusing to publish"):
        finalize_prepared_recording(layout.recording_directory, confirmed=True)
    assert external.read_bytes() == payload
    assert layout.recording_directory.is_dir()
    assert not layout.final_directory.exists()


def test_abort_is_append_only_atomic_and_preserves_every_original_byte(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.create_recording()
    raw = layout.partial_path("raw/ultrasound.bin")
    raw_bytes = b"incomplete-but-scientifically-important\x00\xff"
    raw.write_bytes(raw_bytes)
    journal = layout.recording_directory / "logs/trial.jsonl"
    journal.write_bytes(b'{"event":"writer-crashed"}\n')
    original = {
        "raw/ultrasound.bin.partial": raw.read_bytes(),
        "logs/trial.jsonl": journal.read_bytes(),
    }

    result = abort_recording_preserving_data(
        layout.recording_directory,
        reason="synchronization trigger was never observed",
        confirmed=True,
    )
    assert result.destination_directory.name == f"{layout.trial_uuid}.aborted"
    assert not layout.recording_directory.exists()
    assert iter_aborted_directories(tmp_path) == [result.destination_directory]
    for relative, expected in original.items():
        assert (result.destination_directory / relative).read_bytes() == expected

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["state"] == "ABORTED"
    assert audit["reason"] == "synchronization trigger was never observed"
    assert audit["decided_at_utc"].endswith("Z")
    evidence = {item["relative_path"]: item for item in audit["original_evidence"]}
    for relative, expected in original.items():
        assert evidence[relative]["size_bytes"] == len(expected)
        assert evidence[relative]["sha256"] == sha256_file(result.destination_directory / relative)
    assert iter_finalized_manifest_paths(tmp_path) == []


def test_background_service_exposes_manual_spawned_rescan(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.create_recording()
    service = RecoveryBackgroundService()
    service.start_scan(tmp_path)
    result: tuple[str, str, object] | None = None
    deadline = time.monotonic() + 20.0
    while result is None and time.monotonic() < deadline:
        result = service.poll()  # type: ignore[assignment]
        time.sleep(0.02)
    assert result is not None
    status, operation, payload = result
    assert status == "completed"
    assert operation == "scan"
    reports = tuple(payload)  # type: ignore[arg-type]
    assert len(reports) == 1
    assert reports[0].recording_directory == layout.recording_directory
    service.finish()
