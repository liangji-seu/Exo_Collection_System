from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio import DataStudioWindow, load_catalog_snapshot
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


def _make_manifest(*, condition_code: str = "WALK_LEVEL") -> TrialManifest:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    trial_uuid = uuid4()
    return TrialManifest(
        project_uuid=uuid4(),
        subject_uuid=uuid4(),
        session_uuid=uuid4(),
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code=condition_code,
            condition_name="Level walking",
            condition_level=2,
            parameters={"speed_mps": 0.8},
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=now,
        ),
        timing=TrialTiming(
            started_at_utc=now,
            stopped_at_utc=now + timedelta(seconds=4),
            finalized_at_utc=now + timedelta(seconds=5),
            start_host_monotonic_ns=1_000,
            stop_host_monotonic_ns=4_000_001_000,
            finalize_host_monotonic_ns=5_000_001_000,
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
        artifacts=[
            ManifestArtifact(
                artifact_uuid=uuid4(),
                trial_uuid=trial_uuid,
                modality="imu",
                kind=ArtifactKind.RAW,
                media_type="application/x-hdf5",
                relative_path="raw/imu.h5",
                size_bytes=1234,
                sha256="a" * 64,
                created_at_utc=now,
                finalized_at_utc=now + timedelta(seconds=5),
            )
        ],
        quality=QualitySummary(
            computed_grade=QualityGrade.A,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-0.1.0",
            assessed_at_utc=now + timedelta(seconds=5),
        ),
    )


def _publish_manifest(root: Path, manifest: TrialManifest) -> Path:
    trial_dir = (
        root
        / str(manifest.project_uuid)
        / str(manifest.subject_uuid)
        / str(manifest.session_uuid)
        / "trials"
        / str(manifest.trial_uuid)
    )
    return save_manifest(trial_dir / "manifest.json", manifest)


def _wait_until(app: QApplication, predicate: object, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:  # type: ignore[operator]
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()  # type: ignore[operator]


def test_snapshot_scans_only_published_manifests_and_never_artifacts(
    tmp_path: Path, monkeypatch: object
) -> None:
    manifest = _make_manifest()
    path = _publish_manifest(tmp_path, manifest)
    partial_payload = path.parent / "raw" / "ultrasound.bin.partial"
    partial_payload.parent.mkdir(parents=True)
    partial_payload.write_bytes(b"must not be opened")

    recording_manifest = _make_manifest(condition_code="SHOULD_NOT_BE_INDEXED")
    recording_path = (
        path.parent.parent
        / f"{recording_manifest.trial_uuid}.recording"
        / "manifest.json"
    )
    recording_path.parent.mkdir(parents=True)
    recording_path.write_text(
        recording_manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    original_open = Path.open

    def guarded_open(candidate: Path, *args: object, **kwargs: object) -> object:
        if candidate.suffix == ".partial" or any(
            part.endswith(".recording") for part in candidate.parts
        ):
            raise AssertionError(f"Data Studio opened active data: {candidate}")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)  # type: ignore[attr-defined]
    snapshot = load_catalog_snapshot(tmp_path)

    assert snapshot.scan_report.indexed == 1
    assert snapshot.statistics == {
        "trial_count": 1,
        "finalized_count": 1,
        "total_duration_s": 4.0,
        "by_condition": {
            "WALK_LEVEL": {"trial_count": 1, "duration_s": 4.0}
        },
    }
    assert snapshot.tree[0]["children"][0]["children"][0]["children"][0][
        "children"
    ][0]["label"] == "raw/imu.h5"


def test_window_refreshes_in_background_and_enforces_lightweight_mode(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest()
    _publish_manifest(tmp_path, manifest)
    app = QApplication.instance() or QApplication(["test-data-studio"])

    with AcquisitionLock(tmp_path, manifest.trial_uuid):
        window = DataStudioWindow(tmp_path, autostart_refresh=False)
        completions: list[bool] = []
        window.refresh_finished.connect(completions.append)
        window.refresh_catalog()

        assert window.refresh_in_progress
        assert window.isEnabled()  # refresh must not disable/block the whole UI
        _wait_until(app, lambda: bool(completions))

        assert completions == [True]
        assert window.tree_widget.topLevelItemCount() == 1
        assert window.statistics["trial_count"] == 1
        assert window.condition_table.item(0, 0).text() == "WALK_LEVEL"
        assert window.lightweight_mode
        assert "轻量模式" in window.activity_banner.text()
        assert all(not action.isEnabled() for action in window._restricted_actions)

    _wait_until(app, lambda: not window.lightweight_mode, timeout_s=2.0)
    assert all(action.isEnabled() for action in window._restricted_actions)

    completions.clear()
    window.refresh_catalog()
    _wait_until(app, lambda: bool(completions))
    assert completions == [True]
    assert not window.lightweight_mode
    window.close()
