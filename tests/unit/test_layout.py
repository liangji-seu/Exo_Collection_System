from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from exo_collection.storage.activity import AcquisitionLock, _pid_is_alive, read_activity
from exo_collection.storage.checksum import verify_checksum_manifest, write_checksum_manifest
from exo_collection.storage.layout import (
    TrialLayout,
    iter_aborted_directories,
    iter_finalized_manifest_paths,
    iter_recording_directories,
    name_has_storage_suffix,
    path_has_unpublished_component,
    safe_relative_path,
)


def make_layout(tmp_path) -> TrialLayout:
    return TrialLayout.build(tmp_path, uuid4(), uuid4(), uuid4(), uuid4())


def test_partial_publish_and_atomic_trial_finalize(tmp_path) -> None:
    layout = make_layout(tmp_path)
    layout.create_recording()
    partial = layout.partial_path("raw/imu.h5")
    partial.write_bytes(b"immutable samples")
    artifact = layout.publish_partial("raw/imu.h5")
    assert artifact.name == "imu.h5"
    assert not partial.exists()

    (layout.recording_directory / "manifest.json").write_text("{}", encoding="utf-8")
    write_checksum_manifest(
        layout.recording_directory,
        ["raw/imu.h5", "manifest.json"],
    )
    final = layout.finalize_directory()

    assert final == layout.final_directory
    assert not layout.recording_directory.exists()
    assert (final / "raw/imu.h5").read_bytes() == b"immutable samples"
    assert all(verify_checksum_manifest(final / "checksums.sha256").values())
    assert iter_finalized_manifest_paths(tmp_path) == [final / "manifest.json"]


def test_finalization_rejects_open_partial_file(tmp_path) -> None:
    layout = make_layout(tmp_path)
    layout.create_recording()
    layout.partial_path("raw/ultrasound.bin").write_bytes(b"still open")
    (layout.recording_directory / "manifest.json").write_text("{}", encoding="utf-8")
    (layout.recording_directory / "checksums.sha256").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial"):
        layout.finalize_directory()


@pytest.mark.parametrize(
    "name",
    [
        "trial.RECORDING",
        "signal.PaRtIaL",
        "trial.AbOrTeD",
        ".annex.BUILDING",
    ],
)
def test_storage_state_suffixes_use_windows_case_semantics(name: str) -> None:
    assert name_has_storage_suffix(name)
    assert path_has_unpublished_component(Path("safe") / name / "payload.bin")


@pytest.mark.parametrize(
    "name",
    [
        "recording",
        "trial.recording.notes",
        "signal.partial.backup",
        "aborted_reason.txt",
        ".building-work",
    ],
)
def test_storage_state_suffixes_do_not_match_ordinary_names(name: str) -> None:
    assert not name_has_storage_suffix(name)
    assert not path_has_unpublished_component(Path("safe") / name / "payload.bin")


def test_package_scanners_are_case_insensitive_and_keep_ordinary_names(
    tmp_path,
) -> None:
    recording = tmp_path / f"{uuid4()}.RECORDING"
    aborted = tmp_path / f"{uuid4()}.AbOrTeD"
    recording.mkdir()
    aborted.mkdir()

    visible = tmp_path / "trial.partial.backup" / "manifest.json"
    visible.parent.mkdir()
    visible.write_text("{}", encoding="utf-8")
    hidden_directories: dict[str, Path] = {}
    for suffix in ("RECORDING", "PARTIAL", "ABORTED", "BUILDING"):
        hidden_directory = tmp_path / f"hidden.{suffix}"
        hidden_directories[suffix] = hidden_directory
        hidden = hidden_directory / "manifest.json"
        hidden.parent.mkdir()
        hidden.write_text("{}", encoding="utf-8")

    assert iter_recording_directories(tmp_path) == sorted(
        (recording, hidden_directories["RECORDING"])
    )
    assert iter_aborted_directories(tmp_path) == sorted(
        (aborted, hidden_directories["ABORTED"])
    )
    assert iter_finalized_manifest_paths(tmp_path) == [visible]


def test_finalization_rejects_mixed_case_unpublished_descendant(tmp_path) -> None:
    layout = make_layout(tmp_path)
    layout.create_recording()
    (layout.recording_directory / "raw" / "leftover.PARTIAL").write_bytes(b"open")
    (layout.recording_directory / "manifest.json").write_text("{}", encoding="utf-8")
    (layout.recording_directory / "checksums.sha256").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unpublished"):
        layout.finalize_directory()


@pytest.mark.parametrize("partition", ["F", "T"])
def test_project_partition_is_a_readable_top_level_folder(
    tmp_path, partition: str
) -> None:
    layout = TrialLayout.build(
        tmp_path,
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        project_partition=partition,
    )

    assert layout.session_directory.relative_to(tmp_path).parts[0] == partition
    assert str(layout.project_uuid) not in layout.session_directory.parts


def test_subject_code_is_the_readable_folder_without_becoming_the_primary_key(
    tmp_path,
) -> None:
    subject_uuid = uuid4()
    layout = TrialLayout.build(
        tmp_path,
        uuid4(),
        subject_uuid,
        uuid4(),
        uuid4(),
        project_partition="F",
        subject_code="001",
    )

    relative = layout.session_directory.relative_to(tmp_path)
    assert relative.parts[:2] == ("F", "001")
    assert str(subject_uuid) not in relative.parts


@pytest.mark.parametrize("subject_code", ["1", "01", "0001", "A01", "../001"])
def test_subject_code_folder_rejects_noncanonical_values(
    tmp_path, subject_code: str
) -> None:
    with pytest.raises(ValueError, match="subject_code"):
        TrialLayout.build(
            tmp_path,
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
            project_partition="T",
            subject_code=subject_code,
        )


@pytest.mark.parametrize("partition", ["", "formal", "../F", "F/T"])
def test_project_partition_rejects_noncanonical_values(
    tmp_path, partition: str
) -> None:
    with pytest.raises(ValueError, match="project_partition"):
        TrialLayout.build(
            tmp_path,
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
            project_partition=partition,
        )


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "raw/../escape",
        "raw/ultrasound.bin:hidden",
        "raw/CON",
        "raw/report. ",
        "raw/report?.json",
    ],
)
def test_trial_paths_reject_escape(path: str) -> None:
    with pytest.raises(ValueError):
        safe_relative_path(path)


def test_activity_lock_announces_lightweight_mode(tmp_path) -> None:
    trial_uuid = uuid4()
    with AcquisitionLock(tmp_path, trial_uuid):
        activity = read_activity(tmp_path)
        assert activity is not None
        assert activity.trial_uuid == str(trial_uuid)
    assert read_activity(tmp_path) is None


def test_activity_lock_heartbeats_during_slow_finalization(tmp_path) -> None:
    with AcquisitionLock(
        tmp_path,
        uuid4(),
        stale_after_s=0.25,
        heartbeat_interval_s=0.04,
    ):
        first = read_activity(tmp_path, stale_after_s=0.1)
        assert first is not None
        time.sleep(0.16)
        second = read_activity(tmp_path, stale_after_s=0.1)
        assert second is not None
        assert second.heartbeat_monotonic_ns > first.heartbeat_monotonic_ns


def test_activity_lock_reclaims_stale_dead_owner(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stale_path = tmp_path / ".collector-active.json"
    stale_path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "hostname": socket.gethostname(),
                "trial_uuid": str(uuid4()),
                "heartbeat_monotonic_ns": time.perf_counter_ns() - 10_000_000_000,
                "heartbeat_utc_ns": time.time_ns() - 10_000_000_000,
                "owner_token": "dead-owner",
            }
        ),
        encoding="utf-8",
    )
    with AcquisitionLock(tmp_path, uuid4()) as replacement:
        activity = read_activity(tmp_path)
        assert activity is not None
        assert activity.owner_token == replacement.owner_token
    assert not stale_path.exists()


def test_read_activity_keeps_lightweight_mode_for_stale_but_live_local_owner(
    tmp_path,
) -> None:
    lock_path = tmp_path / ".collector-active.json"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "trial_uuid": str(uuid4()),
                "heartbeat_monotonic_ns": time.perf_counter_ns() - 10_000_000_000,
                "heartbeat_utc_ns": time.time_ns() - 10_000_000_000,
                "owner_token": "temporarily-delayed-live-owner",
            }
        ),
        encoding="utf-8",
    )

    activity = read_activity(tmp_path, stale_after_s=0.1)

    assert activity is not None
    assert activity.pid == os.getpid()


def test_recent_malformed_activity_lock_fails_closed(tmp_path) -> None:
    lock_path = tmp_path / ".collector-active.json"
    lock_path.write_text("{incomplete", encoding="utf-8")

    activity = read_activity(tmp_path, stale_after_s=5.0)

    assert activity is not None
    assert activity.pid == 0
    assert activity.hostname == "unreadable-lock"


def test_old_malformed_activity_lock_is_ignored(tmp_path) -> None:
    lock_path = tmp_path / ".collector-active.json"
    lock_path.write_text("{incomplete", encoding="utf-8")
    old = time.time() - 10.0
    os.utime(lock_path, (old, old))

    assert read_activity(tmp_path, stale_after_s=1.0) is None


def test_read_activity_rejects_nonpositive_stale_timeout(tmp_path) -> None:
    with pytest.raises(ValueError, match="stale_after_s"):
        read_activity(tmp_path, stale_after_s=0)


@pytest.mark.skipif(os.name != "nt", reason="Windows PID probe regression")
def test_windows_pid_probe_never_terminates_the_process_being_checked() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        close_fds=True,
    )
    try:
        assert _pid_is_alive(process.pid)
        time.sleep(0.1)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
