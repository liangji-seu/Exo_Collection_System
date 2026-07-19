from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from exo_collection.apps.data_studio.upload import (
    OfflineUploadRequest,
    ParamikoScpSession,
    RemoteDatasetStatusScanner,
    RemoteTrialStatus,
    SshScpTrialUploader,
    UnknownHostKeyError,
    UploadError,
    UploadWorkerHandle,
    UploadWorkerEventType,
    _remote_join,
    build_remote_trial_directory,
    build_upload_plan,
    validate_finalized_trial,
    validate_remote_directory,
)
from exo_collection.domain.models import ArtifactKind, Condition, QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    save_manifest,
)


def _publish_trial(root: Path) -> Path:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    project_uuid = uuid4()
    subject_uuid = uuid4()
    session_uuid = uuid4()
    trial_uuid = uuid4()
    trial_dir = root / "F" / str(subject_uuid) / str(session_uuid) / "trials" / str(trial_uuid)
    artifact_path = trial_dir / "raw" / "imu.h5"
    artifact_path.parent.mkdir(parents=True)
    payload = b"immutable-imu-payload"
    artifact_path.write_bytes(payload)
    manifest = TrialManifest(
        project_uuid=project_uuid,
        project_code="F",
        project_name="正式",
        subject_uuid=subject_uuid,
        subject_code="001",
        session_uuid=session_uuid,
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code="WALK_LEVEL",
            condition_name="Level walking",
            condition_level=1,
            parameters={},
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=now,
        ),
        timing=TrialTiming(
            started_at_utc=now,
            stopped_at_utc=now + timedelta(seconds=2),
            finalized_at_utc=now + timedelta(seconds=3),
            start_host_monotonic_ns=1_000,
            stop_host_monotonic_ns=2_000_001_000,
            finalize_host_monotonic_ns=3_000_001_000,
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="test",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="b" * 64,
        ),
        artifacts=[
            ManifestArtifact(
                artifact_uuid=uuid4(),
                trial_uuid=trial_uuid,
                modality="imu",
                kind=ArtifactKind.RAW,
                media_type="application/x-hdf5",
                relative_path="raw/imu.h5",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                created_at_utc=now,
                finalized_at_utc=now + timedelta(seconds=3),
            )
        ],
        quality=QualitySummary(
            computed_grade=QualityGrade.A,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-test",
            assessed_at_utc=now + timedelta(seconds=3),
        ),
    )
    return save_manifest(trial_dir / "manifest.json", manifest)


class _FakeRemoteSession:
    def __init__(self, *, corrupt_remote_hash: bool = False) -> None:
        self.directories = {"/"}
        self.files: dict[str, bytes] = {}
        self.corrupt_remote_hash = corrupt_remote_hash
        self.closed = False

    def ensure_directory(self, remote_path: str) -> None:
        path = PurePosixPath(remote_path)
        current = PurePosixPath("/")
        for part in path.parts[1:]:
            current /= part
            self.directories.add(str(current))

    def exists(self, remote_path: str) -> bool:
        return remote_path in self.directories or remote_path in self.files

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        progress: object | None = None,
    ) -> None:
        assert str(PurePosixPath(remote_path).parent) in self.directories
        self.files[remote_path] = local_path.read_bytes()
        if callable(progress):
            progress(len(self.files[remote_path]), len(self.files[remote_path]))

    def remote_sha256(
        self,
        remote_path: str,
        *,
        progress: object | None = None,
    ) -> str:
        payload = self.files[remote_path]
        if callable(progress):
            progress(len(payload), len(payload))
        if self.corrupt_remote_hash:
            payload += b"corrupt"
        return hashlib.sha256(payload).hexdigest()

    def rename(self, source: str, destination: str) -> None:
        if source in self.files:
            assert destination not in self.files
            self.files[destination] = self.files.pop(source)
            return
        assert source in self.directories
        moved_files = {
            destination + path[len(source) :]: payload
            for path, payload in self.files.items()
            if path == source or path.startswith(source + "/")
        }
        self.files = {
            path: payload
            for path, payload in self.files.items()
            if not (path == source or path.startswith(source + "/"))
        }
        self.files.update(moved_files)
        moved_directories = {
            destination + path[len(source) :]
            for path in self.directories
            if path == source or path.startswith(source + "/")
        }
        self.directories = {
            path
            for path in self.directories
            if not (path == source or path.startswith(source + "/"))
        }
        self.directories.update(moved_directories)

    def remove_file(self, remote_path: str) -> None:
        self.files.pop(remote_path, None)

    def remove_directory(self, remote_path: str) -> None:
        self.directories.discard(remote_path)

    def close(self) -> None:
        self.closed = True


def _password_request(root: Path, manifest_path: Path, secret: str = "S3cret!unique") -> OfflineUploadRequest:
    return OfflineUploadRequest(
        dataset_root=root,
        manifest_path=manifest_path,
        host="example.internal",
        port=22,
        username="researcher",
        remote_workdir="/srv/exo-data",
        password=secret,
    )


def test_upload_uses_manifest_hierarchy_verifies_every_file_and_writes_safe_audit(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "never-persist-this-secret"
    request = _password_request(tmp_path, manifest_path, secret)
    session = _FakeRemoteSession()
    uploader = SshScpTrialUploader(lambda received: session)

    result = uploader.upload(request)
    plan = build_upload_plan(manifest_path)
    expected = build_remote_trial_directory("/srv/exo-data", plan, tmp_path)

    assert result.remote_trial_directory == expected
    assert expected == "/srv/exo-data/" + plan.trial_directory.relative_to(
        tmp_path
    ).as_posix()
    assert result.file_count == 2
    assert session.closed
    assert all(".partial-" not in path for path in session.directories)
    assert f"{expected}/manifest.json" in session.files
    assert f"{expected}/raw/imu.h5" in session.files

    audit_text = result.audit_record_path.read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert audit["status"] == "VERIFIED"
    assert audit["remote"]["authentication_method"] == "PASSWORD"
    assert len(audit["files"]) == 2
    assert all(item["local_sha256"] == item["remote_sha256"] for item in audit["files"])
    assert secret not in audit_text
    assert '"password":' not in audit_text.casefold()


def test_existing_remote_trial_is_merged_additively_without_deleting_extras(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path)
    plan = build_upload_plan(manifest_path)
    remote = build_remote_trial_directory(request.remote_workdir, plan, tmp_path)
    session = _FakeRemoteSession()
    session.ensure_directory(remote)
    manifest_item = next(
        item for item in plan.files if item.relative_path.as_posix() == "manifest.json"
    )
    session.files[f"{remote}/manifest.json"] = manifest_item.local_path.read_bytes()
    session.files[f"{remote}/server-only-note.txt"] = b"keep-me"

    result = SshScpTrialUploader(lambda _request: session).upload(request)

    assert result.remote_trial_directory == remote
    assert session.files[f"{remote}/server-only-note.txt"] == b"keep-me"
    assert session.files[f"{remote}/raw/imu.h5"] == b"immutable-imu-payload"
    assert session.files[f"{remote}/manifest.json"] == manifest_item.local_path.read_bytes()
    assert all(".partial-" not in path for path in session.files)


def test_existing_same_path_with_different_bytes_is_never_overwritten(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path)
    plan = build_upload_plan(manifest_path)
    remote = build_remote_trial_directory(request.remote_workdir, plan, tmp_path)
    session = _FakeRemoteSession()
    session.ensure_directory(remote)
    conflicting_path = f"{remote}/raw/imu.h5"
    session.ensure_directory(str(PurePosixPath(conflicting_path).parent))
    session.files[conflicting_path] = b"remote-different-and-must-survive"

    with pytest.raises(UploadError) as captured:
        SshScpTrialUploader(lambda _request: session).upload(request)

    assert captured.value.code == "REMOTE_CONTENT_CONFLICT"
    assert session.files[conflicting_path] == b"remote-different-and-must-survive"
    assert all(".partial-" not in path for path in session.files)


def test_remote_status_scan_distinguishes_uploaded_missing_partial_and_conflict(
    tmp_path: Path,
) -> None:
    manifests = tuple(_publish_trial(tmp_path) for _ in range(4))
    request = replace(
        _password_request(tmp_path, manifests[0]),
        additional_manifest_paths=manifests[1:],
    )
    session = _FakeRemoteSession()
    plans = [build_upload_plan(path) for path in manifests]
    remote_dirs = [
        build_remote_trial_directory(request.remote_workdir, plan, tmp_path)
        for plan in plans
    ]

    # Trial 1 is an exact mirror.
    session.ensure_directory(remote_dirs[0])
    for item in plans[0].files:
        session.files[f"{remote_dirs[0]}/{item.relative_path.as_posix()}"] = (
            item.local_path.read_bytes()
        )
    # Trial 2 is absent. Trial 3 has only one file. Trial 4 has conflicting bytes.
    session.ensure_directory(remote_dirs[2])
    partial_item = plans[2].files[0]
    session.files[f"{remote_dirs[2]}/{partial_item.relative_path.as_posix()}"] = (
        partial_item.local_path.read_bytes()
    )
    session.ensure_directory(remote_dirs[3])
    for item in plans[3].files:
        session.files[f"{remote_dirs[3]}/{item.relative_path.as_posix()}"] = (
            b"different" if item is plans[3].files[0] else item.local_path.read_bytes()
        )

    result = RemoteDatasetStatusScanner(lambda _request: session).scan(request)

    assert [record.status for record in result.records] == [
        RemoteTrialStatus.UPLOADED,
        RemoteTrialStatus.NOT_UPLOADED,
        RemoteTrialStatus.PARTIAL,
        RemoteTrialStatus.CONFLICT,
    ]
    assert session.closed


@pytest.mark.parametrize(
    "state_suffix",
    [".RECORDING", ".PaRtIaL", ".AbOrTeD", ".BUILDING"],
)
def test_upload_rejects_mixed_case_unpublished_path_and_package_content(
    tmp_path: Path,
    state_suffix: str,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    unsafe_manifest = manifest_path.parent.with_name(
        manifest_path.parent.name + state_suffix
    ) / "manifest.json"
    with pytest.raises(UploadError) as path_error:
        validate_finalized_trial(unsafe_manifest)
    assert path_error.value.code == "ACTIVE_TRIAL"

    leftover = manifest_path.parent / "raw" / f"leftover{state_suffix}"
    leftover.write_bytes(b"not published")
    with pytest.raises(UploadError) as content_error:
        build_upload_plan(manifest_path)
    assert content_error.value.code == "INCOMPLETE_PACKAGE"


def test_upload_does_not_misclassify_suffix_text_in_the_middle(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    ordinary = manifest_path.parent / "notes.partial.backup"
    ordinary.write_bytes(b"operator notes")

    plan = build_upload_plan(manifest_path)

    assert "notes.partial.backup" in {
        item.relative_path.as_posix() for item in plan.files
    }


def test_private_key_and_passphrase_are_ephemeral_and_not_in_audit(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("fake-key-for-injected-session", encoding="utf-8")
    passphrase = "private-passphrase-never-store"
    request = OfflineUploadRequest(
        dataset_root=tmp_path,
        manifest_path=manifest_path,
        host="example.internal",
        port=2202,
        username="researcher",
        remote_workdir="/srv/exo-data",
        private_key_path=private_key,
        private_key_passphrase=passphrase,
    )
    assert passphrase not in repr(request)
    assert str(private_key) not in repr(request)
    assert request.password is None

    result = SshScpTrialUploader(lambda _request: _FakeRemoteSession()).upload(request)
    audit_text = result.audit_record_path.read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert audit["remote"]["authentication_method"] == "PRIVATE_KEY"
    assert passphrase not in audit_text
    assert str(private_key) not in audit_text
    assert '"passphrase":' not in audit_text.casefold()


@pytest.mark.parametrize(
    "remote_path",
    [
        "relative/path",
        "/srv/data;rm",
        "/srv/../data",
        "/srv/./data",
        "/srv/data with spaces",
        "/srv/$HOME",
    ],
)
def test_remote_workdir_rejects_paths_that_are_not_scp_safe(remote_path: str) -> None:
    with pytest.raises(ValueError):
        validate_remote_directory(remote_path)


def test_remote_directory_is_exact_trial_path_relative_to_data_root(
    tmp_path: Path,
) -> None:
    plan = build_upload_plan(_publish_trial(tmp_path))

    remote = build_remote_trial_directory("/srv/data", plan, tmp_path)

    assert PurePosixPath(remote).relative_to("/srv/data").parts == (
        "F",
        str(plan.subject_uuid),
        str(plan.session_uuid),
        "trials",
        str(plan.trial_uuid),
    )


def test_readable_project_subject_condition_session_path_is_preserved(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    plan = build_upload_plan(_publish_trial(tmp_path / "fixture"))
    readable_trial = (
        data_root / "T" / "001" / "WALK_LEVEL" / "session1_20260719_070112"
    )
    readable_trial.mkdir(parents=True)
    plan = replace(plan, trial_directory=readable_trial)

    remote = build_remote_trial_directory("/srv/archive/data", plan, data_root)

    assert remote == (
        "/srv/archive/data/T/001/WALK_LEVEL/session1_20260719_070112"
    )


def test_remote_directory_rejects_trial_outside_selected_data_root(
    tmp_path: Path,
) -> None:
    plan = build_upload_plan(_publish_trial(tmp_path / "actual-data"))

    with pytest.raises(UploadError) as captured:
        build_remote_trial_directory("/srv/data", plan, tmp_path / "other-data")

    assert captured.value.code == "TRIAL_OUTSIDE_DATA_ROOT"


@pytest.mark.parametrize("unsafe_segment", ["", ".", ".."])
def test_remote_join_rejects_empty_and_navigational_segments(
    unsafe_segment: str,
) -> None:
    with pytest.raises(UploadError) as captured:
        _remote_join("/srv/exo-data", unsafe_segment)

    assert captured.value.code == "UNSAFE_REMOTE_PATH"


def test_collector_activity_blocks_upload_before_any_network_connection(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path)
    factory_called = False

    def forbidden_factory(_request: OfflineUploadRequest) -> _FakeRemoteSession:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("network session must not be created")

    with AcquisitionLock(tmp_path):
        with pytest.raises(UploadError, match="Collector") as captured:
            SshScpTrialUploader(forbidden_factory).upload(request)
    assert captured.value.code == "COLLECTOR_ACTIVE"
    assert not factory_called


def test_remote_checksum_failure_cleans_staging_and_records_failure(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "audit-redaction-secret"
    request = _password_request(tmp_path, manifest_path, secret)
    session = _FakeRemoteSession(corrupt_remote_hash=True)

    with pytest.raises(UploadError) as captured:
        SshScpTrialUploader(lambda _request: session).upload(request)

    assert captured.value.code == "REMOTE_INTEGRITY_FAILED"
    assert session.closed
    assert not session.files
    assert all(".partial-" not in path for path in session.directories)
    audit_path = (
        tmp_path
        / ".upload-audit"
        / str(build_upload_plan(manifest_path).trial_uuid)
        / f"{request.transfer_batch_uuid}.json"
    )
    text = audit_path.read_text(encoding="utf-8")
    assert json.loads(text)["status"] == "FAILED"
    assert secret not in text


def test_cancel_is_observed_inside_one_scp_file_and_partial_staging_is_cleaned(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "in-file-cancel-secret"
    request = _password_request(tmp_path, manifest_path, secret)
    cancellation = {"inside_file": False}

    class CancellingRemoteSession(_FakeRemoteSession):
        def upload_file(
            self,
            local_path: Path,
            remote_path: str,
            *,
            progress: object | None = None,
        ) -> None:
            assert callable(progress)
            payload = local_path.read_bytes()
            self.files[remote_path] = payload[: max(1, len(payload) // 2)]
            cancellation["inside_file"] = True
            # This represents SCPClient's buffer-level callback while the same
            # large file is still open on the remote host.
            progress(len(self.files[remote_path]), len(payload))
            raise AssertionError("cancellation callback should interrupt SCP")

    session = CancellingRemoteSession()
    with pytest.raises(UploadError) as captured:
        SshScpTrialUploader(lambda _request: session).upload(
            request,
            cancelled=lambda: cancellation["inside_file"],
        )

    assert captured.value.code == "CANCELLED"
    assert session.closed
    assert not session.files
    assert all(".partial-" not in path for path in session.directories)
    audit_path = (
        tmp_path
        / ".upload-audit"
        / str(build_upload_plan(manifest_path).trial_uuid)
        / f"{request.transfer_batch_uuid}.json"
    )
    audit_text = audit_path.read_text(encoding="utf-8")
    assert json.loads(audit_text)["error"]["code"] == "CANCELLED"
    assert secret not in audit_text


def test_collector_start_is_observed_inside_one_scp_file(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "activity-mid-file-secret"
    request = _password_request(tmp_path, manifest_path, secret)

    class CollectorStartsDuringFile(_FakeRemoteSession):
        def upload_file(
            self,
            local_path: Path,
            remote_path: str,
            *,
            progress: object | None = None,
        ) -> None:
            assert callable(progress)
            payload = local_path.read_bytes()
            self.files[remote_path] = payload[:1]
            with AcquisitionLock(tmp_path):
                progress(1, len(payload))
            raise AssertionError("Collector activity should interrupt SCP")

    session = CollectorStartsDuringFile()
    with pytest.raises(UploadError) as captured:
        SshScpTrialUploader(lambda _request: session).upload(request)

    assert captured.value.code == "COLLECTOR_ACTIVE"
    assert session.closed
    assert not session.files
    audit_path = (
        tmp_path
        / ".upload-audit"
        / str(build_upload_plan(manifest_path).trial_uuid)
        / f"{request.transfer_batch_uuid}.json"
    )
    audit_text = audit_path.read_text(encoding="utf-8")
    assert json.loads(audit_text)["error"]["code"] == "COLLECTOR_ACTIVE"
    assert secret not in audit_text


def test_cancel_is_observed_inside_remote_checksum_and_staging_is_cleaned(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "checksum-cancel-secret"
    request = _password_request(tmp_path, manifest_path, secret)
    cancellation = {"inside_checksum": False}

    class CancellingChecksumSession(_FakeRemoteSession):
        def remote_sha256(
            self,
            remote_path: str,
            *,
            progress: object | None = None,
        ) -> str:
            assert callable(progress)
            payload = self.files[remote_path]
            cancellation["inside_checksum"] = True
            progress(min(1, len(payload)), len(payload))
            raise AssertionError("cancellation callback should interrupt checksum")

    session = CancellingChecksumSession()
    with pytest.raises(UploadError) as captured:
        SshScpTrialUploader(lambda _request: session).upload(
            request,
            cancelled=lambda: cancellation["inside_checksum"],
        )

    assert captured.value.code == "CANCELLED"
    assert session.closed
    assert not session.files
    assert all(".partial-" not in path for path in session.directories)
    audit_path = (
        tmp_path
        / ".upload-audit"
        / str(build_upload_plan(manifest_path).trial_uuid)
        / f"{request.transfer_batch_uuid}.json"
    )
    audit_text = audit_path.read_text(encoding="utf-8")
    assert json.loads(audit_text)["error"]["code"] == "CANCELLED"
    assert secret not in audit_text


def test_collector_start_is_observed_inside_remote_checksum(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "checksum-activity-secret"
    request = _password_request(tmp_path, manifest_path, secret)

    class CollectorStartsDuringChecksum(_FakeRemoteSession):
        def remote_sha256(
            self,
            remote_path: str,
            *,
            progress: object | None = None,
        ) -> str:
            assert callable(progress)
            payload = self.files[remote_path]
            with AcquisitionLock(tmp_path):
                progress(min(1, len(payload)), len(payload))
            raise AssertionError("Collector activity should interrupt checksum")

    session = CollectorStartsDuringChecksum()
    with pytest.raises(UploadError) as captured:
        SshScpTrialUploader(lambda _request: session).upload(request)

    assert captured.value.code == "COLLECTOR_ACTIVE"
    assert session.closed
    assert not session.files
    assert all(".partial-" not in path for path in session.directories)
    audit_path = (
        tmp_path
        / ".upload-audit"
        / str(build_upload_plan(manifest_path).trial_uuid)
        / f"{request.transfer_batch_uuid}.json"
    )
    audit_text = audit_path.read_text(encoding="utf-8")
    assert json.loads(audit_text)["error"]["code"] == "COLLECTOR_ACTIVE"
    assert secret not in audit_text


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False

    def send(self, value: object) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True

    def poll(self) -> bool:
        return False


class _FakeProcess:
    def __init__(self, **arguments: object) -> None:
        self.arguments = arguments
        self.started = False
        self._alive = False
        self.exitcode: int | None = None
        self.joined = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -9

    def join(
        self, _timeout: float | None = None, *, timeout: float | None = None
    ) -> None:
        self.joined = True

    def close(self) -> None:
        self.closed = True


class _FakeSpawnContext:
    def __init__(self) -> None:
        self.pipe_pairs = [
            (_FakeConnection(), _FakeConnection()),
            (_FakeConnection(), _FakeConnection()),
        ]
        self.process: _FakeProcess | None = None

    def Pipe(self, *, duplex: bool) -> tuple[_FakeConnection, _FakeConnection]:  # noqa: N802
        return self.pipe_pairs.pop(0)

    def Process(self, **arguments: object) -> _FakeProcess:  # noqa: N802
        self.process = _FakeProcess(**arguments)
        return self.process


def test_worker_starts_before_credentials_are_sent_and_secret_is_not_in_process_args(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    secret = "not-in-process-argv-or-spawn-args"
    request = _password_request(tmp_path, manifest_path, secret)
    context = _FakeSpawnContext()
    # Static duck typing is intentionally irrelevant to this runtime contract.
    handle = UploadWorkerHandle(context=context)  # type: ignore[arg-type]

    handle.start(request)

    assert context.process is not None and context.process.started
    process_args = context.process.arguments["args"]
    assert request not in process_args
    assert secret not in repr(process_args)
    parent_command = context.process.arguments["args"][0]
    # The parent endpoint is the first endpoint from the first pipe, already
    # held by the handle. It receives the request only after Process.start().
    assert handle._command is not None  # noqa: SLF001 - security contract test
    assert handle._command.sent == [request]  # type: ignore[attr-defined]  # noqa: SLF001
    assert parent_command is not handle._command  # child has only the IPC handle

    handle.terminate_for_shutdown()
    handle.close()


def test_worker_shutdown_kills_process_that_ignores_terminate(tmp_path: Path) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path, "shutdown-secret")

    class StubbornProcess(_FakeProcess):
        def __init__(self, **arguments: object) -> None:
            super().__init__(**arguments)
            self.kill_called = False

        def terminate(self) -> None:
            # Model a native network wait that did not honor terminate within
            # the bounded grace interval.
            return None

        def kill(self) -> None:
            self.kill_called = True
            super().kill()

    class StubbornContext(_FakeSpawnContext):
        def Process(self, **arguments: object) -> StubbornProcess:  # noqa: N802
            self.process = StubbornProcess(**arguments)
            return self.process

    context = StubbornContext()
    handle = UploadWorkerHandle(context=context)  # type: ignore[arg-type]
    handle.start(request)

    assert handle.terminate_for_shutdown(timeout=0) == -9
    assert isinstance(context.process, StubbornProcess)
    assert context.process.kill_called
    handle.close()


def test_worker_start_failure_closes_every_pipe_and_process_handle(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path, "spawn-failure-secret")

    class FailingStartProcess(_FakeProcess):
        def start(self) -> None:
            # Model a Windows spawn failure after the OS process handle became
            # live: cleanup must terminate, join, and close it as well as all
            # four anonymous-pipe endpoints.
            self.started = True
            self._alive = True
            raise OSError("simulated CreateProcess failure")

    class FailingStartContext:
        def __init__(self) -> None:
            self.connections = [_FakeConnection() for _index in range(4)]
            self._pipe_index = 0
            self.process: FailingStartProcess | None = None

        def Pipe(  # noqa: N802
            self, *, duplex: bool
        ) -> tuple[_FakeConnection, _FakeConnection]:
            del duplex
            start = self._pipe_index
            self._pipe_index += 2
            return self.connections[start], self.connections[start + 1]

        def Process(self, **arguments: object) -> FailingStartProcess:  # noqa: N802
            self.process = FailingStartProcess(**arguments)
            return self.process

    context = FailingStartContext()
    handle = UploadWorkerHandle(context=context)  # type: ignore[arg-type]

    with pytest.raises(OSError, match="CreateProcess"):
        handle.start(request)

    assert context.process is not None
    assert context.process.exitcode == -15
    assert context.process.joined
    assert context.process.closed
    assert all(connection.closed for connection in context.connections)
    assert handle._process is None  # noqa: SLF001 - lifecycle contract test
    assert handle._command is None  # noqa: SLF001 - lifecycle contract test
    assert handle._events is None  # noqa: SLF001 - lifecycle contract test


def test_upload_dialog_has_no_server_defaults_and_clears_all_secrets(
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLineEdit

    from exo_collection.apps.data_studio.upload_dialog import OfflineUploadDialog

    manifest_path = _publish_trial(tmp_path)
    _app = QApplication.instance() or QApplication(["test-upload-dialog"])
    dialog = OfflineUploadDialog(manifest_path)

    assert dialog.host_edit.text() == ""
    assert dialog.username_edit.text() == ""
    assert dialog.remote_workdir_edit.text() == ""
    assert dialog.port_spin.value() == 22  # protocol default, not a real endpoint
    assert dialog.password_edit.echoMode() is QLineEdit.EchoMode.Password
    assert dialog.passphrase_edit.echoMode() is QLineEdit.EchoMode.Password

    dialog.host_edit.setText("example.internal")
    dialog.username_edit.setText("researcher")
    dialog.remote_workdir_edit.setText("/srv/exo-data")
    dialog.password_edit.setText("dialog-secret")
    request = dialog.take_request(tmp_path)
    assert request.password == "dialog-secret"
    assert dialog.password_edit.text() == ""
    assert dialog.passphrase_edit.text() == ""
    assert "dialog-secret" not in repr(request)
    dialog.close()


def test_first_host_key_requires_confirmation_then_is_saved_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path)
    state: dict[str, object] = {"trusted": False, "clients": []}

    class FakeMissingHostKeyPolicy:
        pass

    class FakeBadHostKeyException(Exception):
        pass

    class FakeAuthenticationException(Exception):
        pass

    class FakeKey:
        def asbytes(self) -> bytes:
            return b"server-public-key"

        def get_name(self) -> str:
            return "ssh-ed25519"

        def get_base64(self) -> str:
            return "c2VydmVyLXB1YmxpYy1rZXk="

    class FakeHostKeys:
        def add(self, hostname: str, algorithm: str, key: FakeKey) -> None:
            state["added"] = (hostname, algorithm, key.get_base64())

    class FakeTransport:
        @staticmethod
        def is_active() -> bool:
            return True

    class FakeSftp:
        @staticmethod
        def close() -> None:
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.policy: object | None = None
            self.host_keys = FakeHostKeys()
            self.closed = False
            state["clients"].append(self)  # type: ignore[union-attr]

        def load_system_host_keys(self, _path: str | None = None) -> None:
            return None

        def load_host_keys(self, _path: str) -> None:
            return None

        def set_missing_host_key_policy(self, policy: object) -> None:
            self.policy = policy

        def connect(self, **_arguments: object) -> None:
            if state["trusted"]:
                return
            assert self.policy is not None
            self.policy.missing_host_key(  # type: ignore[attr-defined]
                self, "example.internal", FakeKey()
            )

        def get_host_keys(self) -> FakeHostKeys:
            return self.host_keys

        def save_host_keys(self, path: str) -> None:
            Path(path).write_text("saved", encoding="utf-8")
            state["trusted"] = True

        @staticmethod
        def get_transport() -> FakeTransport:
            return FakeTransport()

        @staticmethod
        def open_sftp() -> FakeSftp:
            return FakeSftp()

        def close(self) -> None:
            self.closed = True

    class FakeScpClient:
        def __init__(self, _transport: object, **arguments: object) -> None:
            self.progress4 = arguments.get("progress4")
            state["scp_client"] = self

        def put(self, _local_path: str, **_arguments: object) -> None:
            assert callable(self.progress4)
            self.progress4(b"payload.bin", 100, 40, ("host", 22))
            self.progress4(b"payload.bin", 100, 100, ("host", 22))

        @staticmethod
        def close() -> None:
            return None

    fake_paramiko = ModuleType("paramiko")
    fake_paramiko.SSHClient = FakeClient  # type: ignore[attr-defined]
    fake_paramiko.MissingHostKeyPolicy = FakeMissingHostKeyPolicy  # type: ignore[attr-defined]
    fake_paramiko.BadHostKeyException = FakeBadHostKeyException  # type: ignore[attr-defined]
    fake_paramiko.AuthenticationException = FakeAuthenticationException  # type: ignore[attr-defined]
    fake_scp = ModuleType("scp")
    fake_scp.SCPClient = FakeScpClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)
    monkeypatch.setitem(sys.modules, "scp", fake_scp)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    with pytest.raises(UnknownHostKeyError) as first_use:
        ParamikoScpSession(request)
    info = first_use.value.host_key
    assert info.algorithm == "ssh-ed25519"
    assert info.sha256_fingerprint.startswith("SHA256:")

    accepted = replace(request, accepted_host_key=info)
    session = ParamikoScpSession(accepted)
    observed_progress: list[tuple[int, int]] = []
    session.upload_file(
        manifest_path,
        "/srv/exo-data/manifest.json",
        progress=lambda sent, total: observed_progress.append((sent, total)),
    )
    assert observed_progress == [(40, 100), (100, 100)]
    session.close()
    assert state["trusted"] is True
    assert (tmp_path / ".exo_collection_system" / "known_hosts").is_file()

    # A later request needs no acceptance payload because the saved key is
    # loaded and normal strict host-key verification handles the connection.
    subsequent = ParamikoScpSession(request)
    subsequent.close()


def test_real_spawn_worker_receives_credentials_by_pipe_and_honors_activity_guard(
    tmp_path: Path,
) -> None:
    manifest_path = _publish_trial(tmp_path)
    request = _password_request(tmp_path, manifest_path, "spawn-pipe-only-secret")
    worker = UploadWorkerHandle()
    terminal = None

    with AcquisitionLock(tmp_path):
        worker.start(request)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and terminal is None:
            for event in worker.poll_events():
                if event.event_type in {
                    UploadWorkerEventType.COMPLETED,
                    UploadWorkerEventType.FAILED,
                }:
                    terminal = event
                    break
            if terminal is None:
                time.sleep(0.02)

    worker.join(5)
    try:
        assert terminal is not None
        assert terminal.event_type is UploadWorkerEventType.FAILED
        assert terminal.error_code == "COLLECTOR_ACTIVE"
        assert "spawn-pipe-only-secret" not in (terminal.message or "")
        assert worker.exitcode == 0
    finally:
        if worker.is_alive:
            worker.terminate_for_shutdown()
        worker.close()
