from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from exo_collection.apps.data_studio.management import (
    AnnexValidationStatus,
    ConditionCompletionStatus,
    ManagementBusyError,
    ManagementError,
    QualityReviewStatus,
    TrialFilter,
    UploadAuditStatus,
    compute_subject_coverage,
    export_manifest_inventory,
    export_manifest_inventory_checked,
    filter_trial_records,
    load_management_index,
    scan_external_annexes,
    summarize_dataset_states,
)
from exo_collection.apps.data_studio.quality_reviews import append_quality_review
from exo_collection.apps.data_studio.upload import build_upload_plan
from exo_collection.domain.models import ArtifactKind, Condition, QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.external.importer import (
    AnnexFile,
    ExternalAnnexManifest,
    ExternalModality,
    MappingReference,
    SourceAudit,
)
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file, write_checksum_manifest
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    load_manifest,
    save_manifest,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)


def _publish_trial(
    root: Path,
    *,
    project_uuid: UUID,
    project_code: str,
    project_name: str,
    subject_uuid: UUID,
    subject_code: str,
    session_uuid: UUID,
    condition_code: str,
    condition_name: str,
    repeat_index: int,
    quality: QualityGrade,
    started_at: datetime,
) -> Path:
    trial_uuid = uuid4()
    trial_root = (
        root
        / project_code
        / str(subject_uuid)
        / str(session_uuid)
        / "trials"
        / str(trial_uuid)
    )
    artifact_path = trial_root / "reports" / "summary.txt"
    artifact_path.parent.mkdir(parents=True)
    payload = f"{condition_code}:{repeat_index}:{quality.value}".encode()
    artifact_path.write_bytes(payload)
    stopped_at = started_at + timedelta(seconds=2.5)
    finalized_at = stopped_at + timedelta(milliseconds=100)
    artifact = ManifestArtifact(
        artifact_uuid=uuid4(),
        trial_uuid=trial_uuid,
        modality="trial",
        kind=ArtifactKind.REPORT,
        media_type="text/plain",
        relative_path="reports/summary.txt",
        size_bytes=len(payload),
        sha256=sha256_file(artifact_path),
        created_at_utc=started_at,
        finalized_at_utc=finalized_at,
    )
    manifest = TrialManifest(
        project_uuid=project_uuid,
        project_code=project_code,
        project_name=project_name,
        subject_uuid=subject_uuid,
        subject_code=subject_code,
        session_uuid=session_uuid,
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code=condition_code,
            condition_name=condition_name,
            condition_level=1,
            parameters={},
            repeat_index=repeat_index,
            protocol_version="1.0.0",
            selected_at_utc=started_at,
        ),
        timing=TrialTiming(
            started_at_utc=started_at,
            stopped_at_utc=stopped_at,
            finalized_at_utc=finalized_at,
            start_host_monotonic_ns=1_000_000_000,
            stop_host_monotonic_ns=3_500_000_000,
            finalize_host_monotonic_ns=3_600_000_000,
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="management-test",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="c" * 64,
        ),
        artifacts=[artifact],
        quality=QualitySummary(
            computed_grade=quality,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-test",
            assessed_at_utc=finalized_at,
        ),
    )
    manifest_path = save_manifest(trial_root / "manifest.json", manifest)
    write_checksum_manifest(
        trial_root,
        ("reports/summary.txt", "manifest.json"),
    )
    return manifest_path


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "dataset"
    project_f = uuid4()
    project_t = uuid4()
    subject_1 = uuid4()
    subject_2 = uuid4()
    session_1 = uuid4()
    session_2 = uuid4()
    session_3 = uuid4()
    paths = {
        "stand_a": _publish_trial(
            root,
            project_uuid=project_f,
            project_code="F",
            project_name="正式",
            subject_uuid=subject_1,
            subject_code="001",
            session_uuid=session_1,
            condition_code="STAND",
            condition_name="静止站立",
            repeat_index=1,
            quality=QualityGrade.A,
            started_at=BASE_TIME,
        ),
        "walk_c": _publish_trial(
            root,
            project_uuid=project_f,
            project_code="F",
            project_name="正式",
            subject_uuid=subject_1,
            subject_code="001",
            session_uuid=session_1,
            condition_code="WALK_LEVEL",
            condition_name="平地行走",
            repeat_index=1,
            quality=QualityGrade.C,
            started_at=BASE_TIME + timedelta(days=1),
        ),
        "walk_b": _publish_trial(
            root,
            project_uuid=project_f,
            project_code="F",
            project_name="正式",
            subject_uuid=subject_1,
            subject_code="001",
            session_uuid=session_2,
            condition_code="WALK_LEVEL",
            condition_name="平地行走",
            repeat_index=2,
            quality=QualityGrade.B,
            started_at=BASE_TIME + timedelta(days=2),
        ),
        "test_stand_c": _publish_trial(
            root,
            project_uuid=project_t,
            project_code="T",
            project_name="测试",
            subject_uuid=subject_2,
            subject_code="002",
            session_uuid=session_3,
            condition_code="STAND",
            condition_name="静止站立",
            repeat_index=1,
            quality=QualityGrade.C,
            started_at=BASE_TIME + timedelta(days=3),
        ),
    }
    return root, paths


def _write_upload_audit(
    root: Path,
    manifest_path: Path,
    *,
    verified: bool,
    valid: bool = True,
) -> Path:
    plan = build_upload_plan(manifest_path)
    transfer_uuid = uuid4()
    status = "VERIFIED" if verified else "FAILED"
    payload = {
        "schema_version": "1.0.0",
        "transfer_batch_uuid": str(transfer_uuid),
        "trial_uuid": str(plan.trial_uuid),
        "status": status,
        "started_at_utc_ns": 100,
        "completed_at_utc_ns": 200,
        "remote": {
            "host": "example.test",
            "port": 22 if valid else 0,
            "username": "researcher",
            "authentication_method": "PRIVATE_KEY",
            "trial_directory": f"/archive/{plan.trial_uuid}",
        },
        "files": [
            {
                "relative_path": item.relative_path.as_posix(),
                "size_bytes": item.size_bytes,
                "local_sha256": item.sha256,
                "remote_sha256": item.sha256 if verified else None,
            }
            for item in plan.files
        ],
        "error": None if verified else {"code": "NETWORK", "message": "failed"},
    }
    path = root / ".upload-audit" / str(plan.trial_uuid) / f"{transfer_uuid}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_index_and_composable_filters(
    dataset: tuple[Path, dict[str, Path]],
) -> None:
    root, paths = dataset
    index = load_management_index(root)
    assert len(index.records) == 4
    assert not index.catalog_scan_failures
    assert not index.manifest_failures

    assert len(filter_trial_records(index.records, {"projects": ["F"]})) == 3
    assert len(filter_trial_records(index.records, {"subjects": "001"})) == 3
    session_uuid = str(load_manifest(paths["stand_a"]).session_uuid)
    assert len(filter_trial_records(index.records, {"sessions": session_uuid})) == 2
    assert len(filter_trial_records(index.records, {"conditions": "WALK_LEVEL"})) == 2
    assert len(
        filter_trial_records(
            index.records,
            {"start_date": "2026-07-02", "end_date": date(2026, 7, 3)},
        )
    ) == 2
    assert len(filter_trial_records(index.records, {"qualities": "B"})) == 1
    assert len(filter_trial_records(index.records, {"text": "平地行走"})) == 2
    combined = filter_trial_records(
        index.records,
        TrialFilter(projects=("正式",), qualities=("A",)),
    )
    assert [item.condition_code for item in combined] == ["STAND"]
    with pytest.raises(ValidationError, match="end_date"):
        TrialFilter(start_date=date(2026, 7, 2), end_date=date(2026, 7, 1))


def test_verified_sidecars_coverage_and_state_summary(
    dataset: tuple[Path, dict[str, Path]],
) -> None:
    root, paths = dataset
    reviewed_trial = append_quality_review(
        root,
        paths["walk_c"],
        reviewed_grade="B",
        reviewer="reviewer",
        reason="人工复核通过",
    )
    _write_upload_audit(root, paths["stand_a"], verified=True)
    _write_upload_audit(root, paths["walk_b"], verified=True, valid=False)

    index = load_management_index(root)
    by_uuid = {item.trial_uuid: item for item in index.records}
    reviewed = by_uuid[str(reviewed_trial.record.trial_uuid)]
    assert reviewed.quality_review_status is QualityReviewStatus.REVIEWED
    assert reviewed.computed_quality_grade == "C"
    assert reviewed.effective_quality_grade == "B"
    uploaded = by_uuid[str(load_manifest(paths["stand_a"]).trial_uuid)]
    assert uploaded.upload_status is UploadAuditStatus.VERIFIED
    invalid_upload = by_uuid[str(load_manifest(paths["walk_b"]).trial_uuid)]
    assert invalid_upload.upload_status is UploadAuditStatus.INVALID_SIDECAR
    assert invalid_upload.sidecar_errors

    coverage = compute_subject_coverage(index.records)
    formal = next(item for item in coverage if item.project_code == "F")
    assert formal.total_trial_count == 3
    assert formal.valid_trial_count == 3
    assert formal.completed_condition_codes == ("STAND", "WALK_LEVEL")
    walk = next(item for item in formal.conditions if item.condition_code == "WALK_LEVEL")
    assert walk.status is ConditionCompletionStatus.COMPLETED
    assert walk.trial_count == 2
    assert walk.valid_trial_count == 2
    assert walk.repeat_indices == (1, 2)
    test_subject = next(item for item in coverage if item.project_code == "T")
    assert test_subject.completed_condition_codes == ()
    assert test_subject.missing_condition_codes == ("STAND", "WALK_LEVEL")
    assert test_subject.attempted_without_valid_condition_codes == ("STAND",)
    assert test_subject.never_attempted_condition_codes == ("WALK_LEVEL",)
    stand = next(item for item in test_subject.conditions if item.condition_code == "STAND")
    walk_missing = next(
        item for item in test_subject.conditions if item.condition_code == "WALK_LEVEL"
    )
    assert stand.status is ConditionCompletionStatus.ATTEMPTED_NO_VALID_TRIAL
    assert walk_missing.status is ConditionCompletionStatus.MISSING

    summary = summarize_dataset_states(root, index.records)
    assert summary.finalized_count == 4
    assert summary.pending_quality_count == 3
    assert summary.pending_upload_count == 3
    assert summary.reviewed_trial_uuids == (str(reviewed_trial.record.trial_uuid),)
    assert summary.verified_upload_trial_uuids == (uploaded.trial_uuid,)
    assert invalid_upload.trial_uuid in summary.sidecar_error_trial_uuids


def test_tampered_quality_sidecar_is_not_treated_as_reviewed(
    dataset: tuple[Path, dict[str, Path]],
) -> None:
    root, paths = dataset
    saved = append_quality_review(
        root,
        paths["walk_c"],
        reviewed_grade="A",
        reviewer="reviewer",
        reason="initial decision",
    )
    saved.path.write_text(saved.path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    record = next(
        item
        for item in load_management_index(root).records
        if item.trial_uuid == str(saved.record.trial_uuid)
    )
    assert record.quality_review_status is QualityReviewStatus.INVALID_SIDECAR
    assert record.effective_quality_grade == "C"
    assert record.pending_quality_review
    assert record.sidecar_errors


def _make_package_state_directory(root: Path, suffix: str, *, valid: bool = True) -> Path:
    trial_uuid = uuid4()
    path = root / "F" / str(uuid4()) / str(uuid4()) / "trials" / f"{trial_uuid}{suffix}"
    path.mkdir(parents=True)
    if suffix == ".aborted":
        retained = path / "raw" / "ultrasound.partial"
        retained.parent.mkdir()
        retained.write_bytes(b"retained crash tail")
        operation_uuid = uuid4()
        audit = path / "reports" / f"recovery-abort-{operation_uuid}.json"
        audit.parent.mkdir()
        payload = {
            "schema_version": "1.0.0",
            "operation_uuid": str(operation_uuid),
            "action": "ABORT_PRESERVING_DATA",
            "state": "ABORTED",
            "confirmed": valid,
            "reason": "operator decision",
            "decided_at_utc": "2026-07-04T08:00:00Z",
            "trial_uuid": str(trial_uuid),
            "destination_directory": str(path.resolve()),
            "original_evidence": [
                {
                    "relative_path": "raw/ultrasound.partial",
                    "size_bytes": retained.stat().st_size,
                    "sha256": sha256_file(retained),
                }
            ],
        }
        audit.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_recovery_aborted_and_busy_state_identification(
    dataset: tuple[Path, dict[str, Path]],
) -> None:
    root, _paths = dataset
    recording = _make_package_state_directory(root, ".recording")
    aborted = _make_package_state_directory(root, ".aborted", valid=True)
    invalid_aborted = _make_package_state_directory(root, ".aborted", valid=False)
    index = load_management_index(root)
    summary = summarize_dataset_states(root, index.records)
    assert [item.path for item in summary.pending_recovery] == [recording]
    by_path = {item.path: item for item in summary.aborted}
    assert by_path[aborted].evidence_verified
    assert not by_path[invalid_aborted].evidence_verified

    with AcquisitionLock(root, uuid4()):
        with pytest.raises(ManagementBusyError):
            load_management_index(root)
        with pytest.raises(ManagementBusyError):
            summarize_dataset_states(root, index.records)
        with pytest.raises(ManagementBusyError):
            scan_external_annexes(root)


def test_inventory_export_is_atomic_and_never_writes_into_trial(
    dataset: tuple[Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    root, _paths = dataset
    index = load_management_index(root)
    selected = filter_trial_records(index.records, {"projects": "F"})
    result = export_manifest_inventory(selected, tmp_path / "exports" / "formal")
    assert result.record_count == 3
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 3
    assert {
        "trial_uuid",
        "manifest_path",
        "state",
        "effective_quality_grade",
        "condition_code",
        "date_utc",
        "duration_s",
        "artifact_count",
        "artifact_total_bytes",
    }.issubset(payload["records"][0])
    with result.csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    before = result.json_path.read_bytes()
    with pytest.raises(FileExistsError):
        export_manifest_inventory(selected, result.json_path)
    assert result.json_path.read_bytes() == before
    export_manifest_inventory(selected[:1], result.json_path, overwrite=True)
    assert json.loads(result.json_path.read_text(encoding="utf-8"))["record_count"] == 1
    assert not list(result.json_path.parent.glob(".*.tmp"))

    artifact = selected[0].manifest_path.parent / "reports" / "summary.txt"
    artifact_before = artifact.read_bytes()
    with pytest.raises(ManagementError, match="Trial"):
        export_manifest_inventory(selected, selected[0].manifest_path.parent / "inventory")
    assert artifact.read_bytes() == artifact_before

    unselected_trial = next(
        record for record in index.records if record.trial_uuid != selected[0].trial_uuid
    )
    with pytest.raises(ManagementError, match="任何不可变 Trial"):
        export_manifest_inventory_checked(
            root,
            selected[:1],
            unselected_trial.manifest_path.parent / "inventory",
        )

    for suffix in ("RECORDING", "PARTIAL", "ABORTED", "BUILDING"):
        with pytest.raises(ManagementError, match="recording|partial|aborted"):
            export_manifest_inventory(
                selected,
                tmp_path / f"unsafe.{suffix}" / "inventory",
            )

    ordinary = export_manifest_inventory(
        selected,
        tmp_path / "exports.partial.backup" / "inventory",
    )
    assert ordinary.record_count == 3


def _publish_annex(root: Path, manifest_path: Path) -> tuple[Path, Path]:
    base = load_manifest(manifest_path)
    annex_uuid = uuid4()
    annex_root = root / "external_annexes" / str(base.trial_uuid) / str(annex_uuid)
    artifact = annex_root / "artifacts" / "force.csv"
    mapping = annex_root / "alignment" / "mapping.json"
    artifact.parent.mkdir(parents=True)
    mapping.parent.mkdir()
    artifact.write_bytes(b"time,force\n0,10\n")
    mapping.write_text('{"scale_a":1.0,"offset_b_ns":0}\n', encoding="utf-8")
    file_record = AnnexFile(
        artifact_uuid=uuid4(),
        role="external_original",
        relative_path="artifacts/force.csv",
        media_type="text/csv",
        size_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        source_audit=SourceAudit(
            source_path_redacted="force.csv",
            source_path_sha256="d" * 64,
            original_filename_redacted="force.csv",
            source_size_bytes=artifact.stat().st_size,
            source_mtime_ns=1,
        ),
    )
    annex = ExternalAnnexManifest(
        annex_uuid=annex_uuid,
        trial_uuid=base.trial_uuid,
        base_manifest_uuid=base.manifest_uuid,
        base_manifest_schema_version=base.schema_version,
        base_manifest_relative_path=manifest_path.relative_to(root).as_posix(),
        base_manifest_sha256=sha256_file(manifest_path),
        modality=ExternalModality.FORCE_PLATE,
        source_system="force plate export",
        external_clock_domain="force_plate_clock",
        imported_at_utc=BASE_TIME,
        files=(file_record,),
        mapping=MappingReference(
            mapping_uuid=uuid4(),
            size_bytes=mapping.stat().st_size,
            sha256=sha256_file(mapping),
            quality="GOOD",
            offset_only=False,
            anchor_count=3,
        ),
    )
    annex_manifest = annex_root / "annex_manifest.json"
    annex_manifest.write_text(
        json.dumps(annex.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_checksum_manifest(
        annex_root,
        ("artifacts/force.csv", "alignment/mapping.json", "annex_manifest.json"),
    )
    return annex_root, artifact


def test_annex_scan_groups_by_trial_skips_temporary_and_detects_tampering(
    dataset: tuple[Path, dict[str, Path]],
) -> None:
    root, paths = dataset
    annex_root, artifact = _publish_annex(root, paths["stand_a"])
    temporary_parent = annex_root.parent / f".{uuid4()}.building"
    temporary_parent.mkdir()
    (annex_root.parent / f"{uuid4()}.partial").mkdir()
    (annex_root.parent / f"{uuid4()}.RECORDING").mkdir()
    (annex_root.parent / f"{uuid4()}.ABORTED").mkdir()
    (annex_root.parent / f"{uuid4()}.BUILDING").mkdir()

    result = scan_external_annexes(root)
    assert len(result.annexes) == 1
    summary = result.annexes[0]
    assert summary.validation_status is AnnexValidationStatus.VERIFIED
    assert summary.trial_uuid == str(load_manifest(paths["stand_a"]).trial_uuid)
    assert summary.modality == "force_plate"
    assert summary.mapping_quality == "GOOD"
    assert summary.file_count == 1
    assert result.by_trial_uuid()[summary.trial_uuid] == (summary,)
    scoped = scan_external_annexes(root, trial_uuid=summary.trial_uuid)
    assert scoped.annexes == (summary,)

    artifact.write_bytes(b"tampered")
    invalid = scan_external_annexes(root).annexes[0]
    assert invalid.validation_status is AnnexValidationStatus.INVALID
    assert invalid.errors
