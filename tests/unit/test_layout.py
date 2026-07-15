from __future__ import annotations

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

