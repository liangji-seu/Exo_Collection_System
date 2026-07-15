from __future__ import annotations

import json
import socket
import time
from uuid import uuid4

import pytest

from exo_collection.storage.activity import AcquisitionLock, read_activity
from exo_collection.storage.checksum import verify_checksum_manifest, write_checksum_manifest
from exo_collection.storage.layout import (
    TrialLayout,
    iter_finalized_manifest_paths,
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


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/absolute", "raw/../escape"])
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
