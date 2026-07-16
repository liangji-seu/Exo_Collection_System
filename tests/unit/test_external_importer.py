from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
from uuid import UUID, uuid4

import h5py
import numpy as np
import pytest

from exo_collection.domain.models import ArtifactKind, Condition
from exo_collection.domain.states import TrialState
from exo_collection.external import (
    ExternalAnnexManifest,
    ExternalImportError,
    ExternalImportRequest,
    ExternalModality,
    import_external_artifact,
)
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file, verify_checksum_manifest
from exo_collection.storage.manifest import (
    AbnormalTermination,
    ClockAndAlignment,
    ConfigurationSnapshot,
    ManifestArtifact,
    ModalityManifest,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    save_manifest,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 7, 16, 0, 0, tzinfo=UTC)


def _write_sync_h5(
    path: Path,
    trial_uuid: UUID,
    host_times_ns: list[int],
    *,
    edge_type: str = "rising",
) -> None:
    path.parent.mkdir(parents=True)
    records = [
        json.dumps(
            {
                "event_type": "sync_pulse",
                "trial_uuid": str(trial_uuid),
                "pulse_id": f"sync-device:{index:06d}",
                "edge_type": edge_type,
                "host_monotonic_ns": host_time,
            },
            separators=(",", ":"),
        )
        for index, host_time in enumerate(host_times_ns, start=1)
    ]
    with h5py.File(path, "w") as handle:
        handle.attrs["closed_cleanly"] = True
        events = handle.create_group("events")
        events.create_dataset(
            "records",
            data=np.asarray(records, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _make_finalized_trial(
    data_root: Path,
    host_times_ns: list[int],
    *,
    edge_type: str = "rising",
    state: TrialState = TrialState.FINALIZED,
) -> Path:
    trial_uuid = uuid4()
    subject_uuid = uuid4()
    session_uuid = uuid4()
    trial_root = data_root / "T" / str(subject_uuid) / str(session_uuid) / "trials" / str(trial_uuid)
    sync_path = trial_root / "raw/sync_pulse.h5"
    _write_sync_h5(sync_path, trial_uuid, host_times_ns, edge_type=edge_type)
    artifact_uuid = uuid4()
    start_ns = host_times_ns[0] if host_times_ns else 1_000_000_000
    stop_ns = (host_times_ns[-1] + 1_000_000) if host_times_ns else start_ns + 1_000_000
    artifact = ManifestArtifact(
        artifact_uuid=artifact_uuid,
        trial_uuid=trial_uuid,
        modality="sync_pulse",
        kind=ArtifactKind.RAW,
        media_type="application/x-hdf5",
        relative_path="raw/sync_pulse.h5",
        size_bytes=sync_path.stat().st_size,
        sha256=sha256_file(sync_path),
        created_at_utc=BASE_TIME,
        finalized_at_utc=BASE_TIME + timedelta(seconds=3),
    )
    abnormal = (
        AbnormalTermination()
        if state is TrialState.FINALIZED
        else AbnormalTermination(
            occurred=True,
            reason="test aborted trial",
            last_state=TrialState.RECORDING,
            occurred_at_utc=BASE_TIME + timedelta(seconds=2),
        )
    )
    manifest = TrialManifest(
        project_uuid=uuid4(),
        project_code="T",
        project_name="测试",
        subject_uuid=subject_uuid,
        subject_code="001",
        session_uuid=session_uuid,
        trial_uuid=trial_uuid,
        state=state,
        condition=Condition(
            condition_code="WALK_LEVEL",
            condition_name="平地行走",
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=BASE_TIME,
        ),
        timing=TrialTiming(
            started_at_utc=BASE_TIME,
            stopped_at_utc=BASE_TIME + timedelta(seconds=2),
            finalized_at_utc=(
                BASE_TIME + timedelta(seconds=3)
                if state is TrialState.FINALIZED
                else None
            ),
            start_host_monotonic_ns=start_ns,
            stop_host_monotonic_ns=stop_ns,
            finalize_host_monotonic_ns=(
                stop_ns + 1_000_000 if state is TrialState.FINALIZED else None
            ),
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="test-commit",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="b" * 64,
        ),
        modalities=[
            ModalityManifest(
                modality="sync_pulse",
                required=True,
                adapter_type="test.sync.Adapter",
                writer_type="hdf5_signal",
                clock_domain="sync_device_clock",
                artifact_uuids=[artifact_uuid],
                channels=["voltage"],
                units=["V"],
                sample_count=1,
            )
        ],
        artifacts=[artifact],
        clock_and_alignment=ClockAndAlignment(
            raw_sync_pulse_artifact_uuids=[artifact_uuid],
            sync_event_artifact_uuids=[artifact_uuid],
        ),
        abnormal_termination=abnormal,
    )
    manifest_path = trial_root / "manifest.json"
    save_manifest(manifest_path, manifest)
    return manifest_path


def _manual_request(
    data_root: Path,
    manifest_path: Path,
    source_path: Path,
    pulse_times: list[float],
    **updates: object,
) -> ExternalImportRequest:
    values: dict[str, object] = {
        "dataset_root": data_root,
        "trial_manifest_path": manifest_path,
        "source_path": source_path,
        "modality": ExternalModality.FORCE_PLATE,
        "source_system": "generic force plate export",
        "external_clock_domain": "force_plate_clock",
        "external_time_unit": "s",
        "external_pulse_times": pulse_times,
    }
    values.update(updates)
    return ExternalImportRequest.model_validate(values)


def test_import_copies_exact_bytes_without_modifying_source_or_trial(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [10_000_000_000])
    source = tmp_path / "force-data.bin"
    source.write_bytes(bytes(range(256)) * 40)
    source_before = source.read_bytes()
    source_stat = source.stat()
    manifest_before = manifest_path.read_bytes()

    result = import_external_artifact(
        _manual_request(data_root, manifest_path, source, [2.0])
    )

    assert result.copied_artifact_path.read_bytes() == source_before
    assert source.read_bytes() == source_before
    assert source.stat().st_mtime_ns == source_stat.st_mtime_ns
    assert manifest_path.read_bytes() == manifest_before
    assert result.annex_directory.parent.parent.name == "external_annexes"
    assert not result.annex_directory.is_relative_to(manifest_path.parent)
    assert all(verify_checksum_manifest(result.annex_directory / "checksums.sha256").values())
    annex = ExternalAnnexManifest.model_validate_json(
        result.annex_manifest_path.read_text(encoding="utf-8")
    )
    assert annex.base_manifest_sha256 == sha256_file(manifest_path)
    assert annex.trial_uuid == UUID(manifest_path.parent.name)
    assert annex.files[0].sha256 == sha256_file(result.copied_artifact_path)


def test_import_rejects_trial_annex_parent_link_that_escapes_dataset(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [10_000_000_000])
    source = tmp_path / "force-data.bin"
    source.write_bytes(b"external measurement")
    annex_root = data_root / "external_annexes"
    annex_root.mkdir()
    outside = tmp_path / "outside-annex-target"
    outside.mkdir()
    linked_parent = annex_root / manifest_path.parent.name
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_parent), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"directory junctions are unavailable: {completed.stderr}")
    else:
        linked_parent.symlink_to(outside, target_is_directory=True)

    try:
        with pytest.raises(ExternalImportError) as captured:
            import_external_artifact(
                _manual_request(data_root, manifest_path, source, [2.0])
            )

        assert captured.value.code == "ANNEX_PARENT_ESCAPE"
        assert not any(outside.iterdir())
    finally:
        # Remove only the link/junction itself; never recurse through it.
        linked_parent.rmdir()


def test_single_pulse_estimates_offset_only_with_nominal_unit_scale(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [10_000_000_000])
    source = tmp_path / "force.csv"
    source.write_text("sample,value\n0,1\n", encoding="utf-8")

    result = import_external_artifact(
        _manual_request(data_root, manifest_path, source, [2.0])
    )
    mapping = json.loads(result.mapping_path.read_text(encoding="utf-8"))

    assert result.offset_only is True
    assert result.quality == "UNAVAILABLE"
    assert mapping["scale_estimated"] is False
    assert mapping["scale_a"] == pytest.approx(1_000_000_000.0)
    assert mapping["offset_b_ns"] == pytest.approx(8_000_000_000.0)
    assert mapping["anchors"][0]["trial_relative_ns"] == 0


def test_multiple_pulses_fit_affine_drift_and_residuals(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    external = [0.0, 1.0, 2.0, 3.0]
    host = [int(5_000_000_000 + value * 1_000_200_000) for value in external]
    manifest_path = _make_finalized_trial(data_root, host)
    source = tmp_path / "mocap.c3d"
    source.write_bytes(b"generic mocap bytes")

    result = import_external_artifact(
        _manual_request(
            data_root,
            manifest_path,
            source,
            external,
            modality=ExternalModality.MOCAP,
            source_system="generic motion capture export",
            external_clock_domain="mocap_clock",
        )
    )
    mapping = json.loads(result.mapping_path.read_text(encoding="utf-8"))

    assert result.offset_only is False
    assert result.anchor_count == 4
    assert result.quality == "GOOD"
    assert mapping["scale_a"] == pytest.approx(1_000_200_000.0)
    assert mapping["offset_b_ns"] == pytest.approx(5_000_000_000.0)
    assert mapping["residuals"]["max_absolute_ns"] < 1e-3
    assert mapping["normalized_fit"]["algorithm_version"].startswith("affine")


def test_csv_column_pulses_and_separate_evidence_are_checksumming_inputs(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    host = [4_000_000_000, 5_000_000_000, 6_000_000_000]
    manifest_path = _make_finalized_trial(data_root, host)
    source = tmp_path / "force.vendor"
    source.write_bytes(b"opaque vendor export")
    pulse_csv = tmp_path / "events.csv"
    pulse_csv.write_text(
        "event_time;label\n0.0;start\n1.0;middle\n2.0;stop\n",
        encoding="utf-8",
    )
    request = ExternalImportRequest(
        dataset_root=data_root,
        trial_manifest_path=manifest_path,
        source_path=source,
        modality=ExternalModality.FORCE_PLATE,
        source_system="generic export",
        external_clock_domain="external_clock",
        external_time_unit="s",
        pulse_csv_path=pulse_csv,
        pulse_csv_column="event_time",
    )

    result = import_external_artifact(request)
    annex = ExternalAnnexManifest.model_validate_json(
        result.annex_manifest_path.read_text(encoding="utf-8")
    )
    mapping = json.loads(result.mapping_path.read_text(encoding="utf-8"))

    assert [item.role for item in annex.files] == [
        "external_original",
        "pulse_evidence",
    ]
    evidence = result.annex_directory / annex.files[1].relative_path
    assert evidence.read_bytes() == pulse_csv.read_bytes()
    assert mapping["pulse_source"]["kind"] == "csv_column"
    assert mapping["pulse_source"]["column"] == "event_time"
    assert all(verify_checksum_manifest(result.annex_directory / "checksums.sha256").values())


def test_single_column_csv_is_supported_when_no_delimiter_can_be_sniffed(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [2_000_000_000, 3_000_000_000])
    source = tmp_path / "pulse.csv"
    source.write_text("pulse_time\n0\n1\n", encoding="utf-8")

    result = import_external_artifact(
        ExternalImportRequest(
            dataset_root=data_root,
            trial_manifest_path=manifest_path,
            source_path=source,
            modality=ExternalModality.OTHER,
            other_modality_label="generic pressure insole",
            external_clock_domain="insole_clock",
            external_time_unit="s",
            pulse_csv_column="pulse_time",
        )
    )

    assert result.anchor_count == 2
    assert result.quality == "ACCEPTABLE"


def test_pulse_count_mismatch_cleans_private_build_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(
        data_root, [1_000_000_000, 2_000_000_000]
    )
    source = tmp_path / "external.bin"
    source.write_bytes(b"copy me then reject alignment")

    with pytest.raises(ExternalImportError) as caught:
        import_external_artifact(
            _manual_request(data_root, manifest_path, source, [0.0])
        )

    assert caught.value.code == "PULSE_COUNT_MISMATCH"
    annex_root = data_root / "external_annexes"
    assert not annex_root.exists()
    assert not list(data_root.rglob("*.building"))


@pytest.mark.parametrize(
    ("pulses", "code"),
    [
        ([], "NO_EXTERNAL_PULSES"),
        ([0.0, 0.0], "NON_MONOTONIC_EXTERNAL_PULSES"),
        ([0.0, float("nan")], "INVALID_EXTERNAL_PULSE"),
        ([1.0, 0.0], "NON_MONOTONIC_EXTERNAL_PULSES"),
    ],
)
def test_external_pulse_shortage_and_invalid_values_are_rejected(
    tmp_path: Path,
    pulses: list[float],
    code: str,
) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [1_000_000_000])
    source = tmp_path / "external.bin"
    source.write_bytes(b"data")

    with pytest.raises(ExternalImportError) as caught:
        import_external_artifact(
            _manual_request(data_root, manifest_path, source, pulses)
        )

    assert caught.value.code == code
    assert not (data_root / "external_annexes").exists()


def test_no_internal_rising_edge_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(
        data_root, [1_000_000_000], edge_type="falling"
    )
    source = tmp_path / "external.bin"
    source.write_bytes(b"data")

    with pytest.raises(ExternalImportError) as caught:
        import_external_artifact(
            _manual_request(data_root, manifest_path, source, [0.0])
        )

    assert caught.value.code == "NO_INTERNAL_RISING_EDGE"


def test_active_acquisition_blocks_import_before_any_annex_write(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [1_000_000_000])
    source = tmp_path / "external.bin"
    source.write_bytes(b"data")

    with AcquisitionLock(data_root, heartbeat_interval_s=0.1, stale_after_s=1.0):
        with pytest.raises(ExternalImportError) as caught:
            import_external_artifact(
                _manual_request(data_root, manifest_path, source, [0.0])
            )

    assert caught.value.code == "ACQUISITION_ACTIVE"
    assert not (data_root / "external_annexes").exists()


def test_non_finalized_trial_and_temporary_paths_are_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    aborted_manifest = _make_finalized_trial(
        data_root, [1_000_000_000], state=TrialState.ABORTED
    )
    source = tmp_path / "external.bin"
    source.write_bytes(b"data")

    with pytest.raises(ExternalImportError) as aborted:
        import_external_artifact(
            _manual_request(data_root, aborted_manifest, source, [0.0])
        )
    assert aborted.value.code == "NOT_FINALIZED"

    finalized = _make_finalized_trial(data_root, [1_000_000_000])
    recording = finalized.parent.with_name(finalized.parent.name + ".recording")
    recording.mkdir(parents=True)
    recording_manifest = recording / "manifest.json"
    recording_manifest.write_bytes(finalized.read_bytes())
    with pytest.raises(ExternalImportError) as active_trial:
        import_external_artifact(
            _manual_request(data_root, recording_manifest, source, [0.0])
        )
    assert active_trial.value.code == "TEMPORARY_PATH_REJECTED"

    partial_source = tmp_path / "external.bin.partial"
    partial_source.write_bytes(b"data")
    with pytest.raises(ExternalImportError) as partial:
        import_external_artifact(
            _manual_request(data_root, finalized, partial_source, [0.0])
        )
    assert partial.value.code == "TEMPORARY_PATH_REJECTED"


@pytest.mark.parametrize(
    "state_suffix",
    [".RECORDING", ".PaRtIaL", ".AbOrTeD", ".BUILDING"],
)
def test_external_import_rejects_all_mixed_case_unpublished_path_states(
    tmp_path: Path,
    state_suffix: str,
) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    finalized = _make_finalized_trial(data_root, [1_000_000_000])
    safe_source = tmp_path / "external.bin"
    safe_source.write_bytes(b"data")

    unsafe_manifest = finalized.parent.with_name(
        finalized.parent.name + state_suffix
    ) / "manifest.json"
    with pytest.raises(ExternalImportError) as manifest_error:
        import_external_artifact(
            _manual_request(data_root, unsafe_manifest, safe_source, [0.0])
        )
    assert manifest_error.value.code == "TEMPORARY_PATH_REJECTED"

    unsafe_source = tmp_path / f"external.bin{state_suffix}"
    unsafe_source.write_bytes(b"data")
    with pytest.raises(ExternalImportError) as source_error:
        import_external_artifact(
            _manual_request(data_root, finalized, unsafe_source, [0.0])
        )
    assert source_error.value.code == "TEMPORARY_PATH_REJECTED"


def test_source_path_and_text_audit_redact_credentials(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [1_000_000_000])
    secret_directory = tmp_path / "password=hunter2"
    secret_directory.mkdir()
    source = secret_directory / "token=abc123.bin"
    source.write_bytes(b"non-sensitive measurement bytes")

    result = import_external_artifact(
        _manual_request(
            data_root,
            manifest_path,
            source,
            [0.0],
            source_system="server password=hunter2",
            external_clock_domain="clock token=abc123",
        )
    )
    audit_text = (
        result.annex_manifest_path.read_text(encoding="utf-8")
        + result.mapping_path.read_text(encoding="utf-8")
        + (result.annex_directory / "checksums.sha256").read_text(encoding="utf-8")
    ).casefold()

    assert "hunter2" not in audit_text
    assert "abc123" not in audit_text
    assert "password=***" in audit_text
    assert "token=***" in audit_text


def test_corrupt_sync_file_is_rejected_without_touching_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    manifest_path = _make_finalized_trial(data_root, [1_000_000_000])
    manifest_before = manifest_path.read_bytes()
    sync_path = manifest_path.parent / "raw/sync_pulse.h5"
    sync_path.write_bytes(sync_path.read_bytes() + b"tampered")
    source = tmp_path / "external.bin"
    source.write_bytes(b"data")

    with pytest.raises(ExternalImportError) as caught:
        import_external_artifact(
            _manual_request(data_root, manifest_path, source, [0.0])
        )

    assert caught.value.code == "SYNC_INTEGRITY_FAILED"
    assert manifest_path.read_bytes() == manifest_before
    assert not (data_root / "external_annexes").exists()
