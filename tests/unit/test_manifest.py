from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from exo_collection.domain.models import (
    ArtifactKind,
    Condition,
    QualityGrade,
)
from exo_collection.domain.states import TrialState
from exo_collection.storage.manifest import (
    ClockAndAlignment,
    ClockDomainKind,
    ClockDomainManifest,
    ConfigurationSnapshot,
    DeviceProvenance,
    ManifestArtifact,
    ModalityManifest,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    export_manifest_json_schema,
    load_manifest,
    manifest_json_schema,
    save_manifest,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)


def make_manifest() -> TrialManifest:
    trial_uuid = uuid4()
    artifact_uuid = uuid4()
    device_id = "imu_left_shank"
    artifact = ManifestArtifact(
        artifact_uuid=artifact_uuid,
        trial_uuid=trial_uuid,
        modality="imu",
        kind=ArtifactKind.RAW,
        media_type="application/x-hdf5",
        relative_path="raw/imu.h5",
        size_bytes=4096,
        sha256="a" * 64,
        created_at_utc=BASE_TIME,
        finalized_at_utc=BASE_TIME + timedelta(seconds=12),
    )
    return TrialManifest(
        project_uuid=uuid4(),
        project_code="F",
        project_name="正式",
        subject_uuid=uuid4(),
        subject_code="001",
        session_uuid=uuid4(),
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code="WALK_LEVEL",
            condition_name="平地行走",
            condition_level=2,
            levels={"load": "L1", "speed": "S2", "assist": "A3"},
            parameters={"speed_mps": 0.8, "assist_level": 3},
            repeat_index=2,
            protocol_version="1.0.0",
            selected_at_utc=BASE_TIME,
        ),
        timing=TrialTiming(
            started_at_utc=BASE_TIME,
            stopped_at_utc=BASE_TIME + timedelta(seconds=10),
            finalized_at_utc=BASE_TIME + timedelta(seconds=12),
            start_host_monotonic_ns=1_000,
            stop_host_monotonic_ns=10_001_000,
            finalize_host_monotonic_ns=12_001_000,
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="0123456789abcdef",
            python_version="3.11",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="b" * 64,
        ),
        devices=[
            DeviceProvenance(
                device_id=device_id,
                modality="imu",
                adapter_type="exo_collection.adapters.simulated.SimulatedImuAdapter",
            )
        ],
        modalities=[
            ModalityManifest(
                modality="imu",
                required=True,
                adapter_type="exo_collection.adapters.simulated.SimulatedImuAdapter",
                writer_type="hdf5_signal",
                clock_domain="imu_left_device_clock",
                device_ids=[device_id],
                artifact_uuids=[artifact_uuid],
                channels=["ax", "ay", "az", "gx", "gy", "gz"],
                units=["m/s2", "m/s2", "m/s2", "rad/s", "rad/s", "rad/s"],
                sample_count=1_000,
                first_sample_index=0,
                last_sample_index=999,
                nominal_sample_rate_hz=100,
            )
        ],
        artifacts=[artifact],
        clock_and_alignment=ClockAndAlignment(
            clock_domains=[
                ClockDomainManifest(
                    clock_domain="imu_left_device_clock",
                    kind=ClockDomainKind.DEVICE_TICK,
                    unit="tick",
                    device_id=device_id,
                    nominal_rate_hz=100,
                )
            ]
        ),
        quality=QualitySummary(
            computed_grade=QualityGrade.A,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-0.1.0",
            assessed_at_utc=BASE_TIME + timedelta(seconds=12),
        ),
    )


def test_manifest_round_trip_and_uuid_links(tmp_path) -> None:
    manifest = make_manifest()
    path = tmp_path / "manifest.json"

    save_manifest(path, manifest)
    restored = load_manifest(path)

    assert restored == manifest
    assert restored.artifacts[0].trial_uuid == restored.trial_uuid
    assert restored.modalities[0].artifact_uuids == [
        restored.artifacts[0].artifact_uuid
    ]
    assert restored.condition.condition_code == "WALK_LEVEL"
    assert restored.project_code == "F"
    assert restored.subject_code == "001"
    assert restored.quality.computed_grade is QualityGrade.A
    assert restored.created_at_utc.utcoffset() == timedelta(0)


def test_manifest_rejects_draft_state() -> None:
    payload = make_manifest().model_dump()
    payload["state"] = TrialState.RECORDING
    with pytest.raises(ValidationError, match="Manifest state"):
        TrialManifest.model_validate(payload)


def test_manifest_rejects_unknown_schema_version() -> None:
    payload = make_manifest().model_dump()
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError, match="unsupported Manifest schema_version"):
        TrialManifest.model_validate(payload)


@pytest.mark.parametrize("project_code", [None, "", "X", "f", "FT"])
def test_manifest_v1_1_requires_formal_or_test_project_code(
    project_code: str | None,
) -> None:
    payload = make_manifest().model_dump()
    payload["project_code"] = project_code

    with pytest.raises(ValidationError):
        TrialManifest.model_validate(payload)


@pytest.mark.parametrize("subject_code", [None, "", "1", "01", "0001", "00A", "００１"])
def test_manifest_v1_1_requires_three_ascii_digit_subject_code(
    subject_code: str | None,
) -> None:
    payload = make_manifest().model_dump()
    payload["subject_code"] = subject_code

    with pytest.raises(ValidationError):
        TrialManifest.model_validate(payload)


def test_manifest_reader_accepts_legacy_v1_0_without_new_labels() -> None:
    payload = make_manifest().model_dump(mode="json")
    payload["schema_version"] = "1.0.0"
    payload.pop("project_code")
    payload.pop("project_name")
    payload.pop("subject_code")

    restored = TrialManifest.model_validate(payload)

    assert restored.schema_version == "1.0.0"
    assert restored.project_code is None
    assert restored.subject_code is None


@pytest.mark.parametrize(
    "path",
    [
        "C:/absolute/raw/imu.h5",
        "/absolute/raw/imu.h5",
        "raw/../outside.bin",
        "raw/data.partial",
        "raw/imu.h5:hidden",
        "raw/CON",
        "raw/report?.json",
    ],
)
def test_manifest_rejects_unsafe_or_temporary_artifact_path(path: str) -> None:
    payload = make_manifest().model_dump()
    payload["artifacts"][0]["relative_path"] = path
    with pytest.raises(ValidationError):
        TrialManifest.model_validate(payload)


def test_manifest_rejects_wrong_trial_artifact_reference() -> None:
    payload = make_manifest().model_dump()
    payload["artifacts"][0]["trial_uuid"] = uuid4()
    with pytest.raises(ValidationError, match="Manifest Trial UUID"):
        TrialManifest.model_validate(payload)


def test_condition_is_a_frozen_snapshot_and_separate_from_quality() -> None:
    manifest = make_manifest()
    with pytest.raises(ValidationError):
        manifest.condition.repeat_index = 3  # type: ignore[misc]
    with pytest.raises(TypeError, match="frozen"):
        manifest.condition.parameters["speed_mps"] = 1.2
    assert not hasattr(manifest.condition, "quality_grade")
    assert manifest.quality.computed_grade is QualityGrade.A


def test_all_datetimes_require_timezone_awareness() -> None:
    payload = make_manifest().model_dump()
    payload["timing"]["started_at_utc"] = datetime(2026, 7, 15, 8, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        TrialManifest.model_validate(payload)


def test_save_is_immutable_by_default_and_partial_load_is_refused(tmp_path) -> None:
    manifest = make_manifest()
    path = tmp_path / "manifest.json"
    save_manifest(path, manifest)
    with pytest.raises(FileExistsError):
        save_manifest(path, manifest)

    partial = tmp_path / "manifest.json.partial"
    partial.write_text(manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        load_manifest(partial)


@pytest.mark.parametrize(
    "relative_path",
    [
        "raw/data.RECORDING",
        "raw/data.PaRtIaL",
        "raw/data.AbOrTeD",
        "raw/.data.BUILDING",
    ],
)
def test_manifest_artifact_rejects_mixed_case_unpublished_suffixes(
    relative_path: str,
) -> None:
    values = make_manifest().artifacts[0].model_dump(mode="python")
    values["relative_path"] = relative_path

    with pytest.raises(ValidationError, match="temporary paths"):
        ManifestArtifact.model_validate(values)


def test_manifest_partial_guard_is_case_insensitive_without_prefix_false_positive(
    tmp_path,
) -> None:
    manifest = make_manifest()
    uppercase_partial = tmp_path / "manifest.json.PARTIAL"
    uppercase_partial.write_text(manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        load_manifest(uppercase_partial)

    ordinary = tmp_path / "trial.partial.backup" / "manifest.json"
    ordinary.parent.mkdir()
    save_manifest(ordinary, manifest)
    assert load_manifest(ordinary) == manifest


def test_json_schema_can_be_exported(tmp_path) -> None:
    schema = manifest_json_schema()
    assert schema["$id"].endswith("/1.1.0.json")
    assert "schema_version" in schema["properties"]
    assert "condition" in schema["properties"]
    assert "clock_and_alignment" in schema["properties"]
    assert schema["properties"]["schema_version"]["enum"] == ["1.0.0", "1.1.0"]
    versioned_identity = schema["allOf"][0]
    assert versioned_identity["if"]["properties"]["schema_version"] == {
        "const": "1.1.0"
    }
    assert versioned_identity["then"]["required"] == [
        "project_code",
        "subject_code",
    ]
    assert versioned_identity["then"]["properties"]["project_code"] == {
        "enum": ["F", "T"]
    }
    assert versioned_identity["then"]["properties"]["subject_code"] == {
        "pattern": r"^[0-9]{3}$",
        "type": "string",
    }

    output = export_manifest_json_schema(tmp_path / "manifest.schema.json")
    assert output.is_file()
    assert '"$defs"' in output.read_text(encoding="utf-8")
