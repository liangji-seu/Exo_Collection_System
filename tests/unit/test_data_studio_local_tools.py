from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QTableWidget,
)

from exo_collection.apps.data_studio.local_dialogs import (
    ChecksumDialog,
    FullStatisticsDialog,
    PlaybackDialog,
    QualityAuditDialog,
)
from exo_collection.apps.data_studio.window import DataStudioWindow
from exo_collection.apps.data_studio.local_tools import (
    AcquisitionBecameActiveError,
    ChecksumReport,
    DataStudioToolError,
    TrialPlayback,
    _read_ultrasound,
    compute_full_statistics,
    load_quality_audit,
    load_trial_playback,
    verify_trial_checksums,
)
from exo_collection.apps.data_studio.process_workers import DataStudioProcessWorker
from exo_collection.apps.data_studio.quality_reviews import (
    QualityReviewError,
    append_quality_review,
    list_quality_reviews,
)
from exo_collection.domain.models import ArtifactKind, Condition, QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file, write_checksum_manifest
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    QualityIssue,
    QualityIssueSeverity,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    load_manifest,
    save_manifest,
)
from exo_collection.writers import BlockBinaryWriter, Hdf5SignalWriter


def _write_hdf5(path: Path, modality: str, columns: int, count: int = 100) -> None:
    channels = tuple(f"{modality}_{index + 1}" for index in range(columns))
    with Hdf5SignalWriter(
        path,
        channels=channels,
        units=("a.u.",) * columns,
        device_metadata={"device_id": f"{modality}_sim"},
        sample_shape=(columns,),
        nominal_rate_hz=100.0,
    ) as writer:
        events = (
            [
                {
                    "event_type": "sync_pulse",
                    "edge_type": "rising",
                    "host_monotonic_ns": 1_500_000_000,
                }
            ]
            if modality == "sync_pulse"
            else None
        )
        writer.append(
            np.arange(count * columns, dtype=np.float32).reshape(count, columns),
            sample_index=0,
            host_monotonic_ns=np.arange(count, dtype=np.uint64) * 10_000_000
            + 900_000_000,
            events=events,
        )


def _build_finalized_trial(data_root: Path) -> tuple[Path, TrialManifest]:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    project_uuid = uuid4()
    subject_uuid = uuid4()
    session_uuid = uuid4()
    trial_uuid = uuid4()
    trial_root = (
        data_root
        / str(project_uuid)
        / str(subject_uuid)
        / str(session_uuid)
        / "trials"
        / str(trial_uuid)
    )
    raw = trial_root / "raw"
    reports = trial_root / "reports"
    raw.mkdir(parents=True)
    reports.mkdir()

    ultrasound_path = raw / "ultrasound.bin"
    with BlockBinaryWriter(
        ultrasound_path,
        dtype=np.uint16,
        sample_shape=(4, 64),
        metadata={
            "clock_domain": "ultrasound_clock",
            "channels": ["ch_1", "ch_2", "ch_3", "ch_4"],
            "nominal_frame_rate_hz": 20.0,
        },
    ) as writer:
        for index in range(30):
            writer.append(
                np.full((1, 4, 64), index, dtype=np.uint16),
                host_monotonic_ns=900_000_000 + index * 50_000_000,
                host_utc_ns=1_000_000_000 + index * 50_000_000,
            )
    _write_hdf5(raw / "imu.h5", "imu", 6)
    _write_hdf5(raw / "encoder.h5", "encoder", 2)
    _write_hdf5(raw / "sync_pulse.h5", "sync_pulse", 1)

    (reports / "quality_report.json").write_text(
        json.dumps(
            {
                "computed_grade": "B",
                "algorithm_version": "quality-test",
                "issues": [
                    {
                        "code": "US_SIGNAL_WEAK",
                        "severity": "WARNING",
                        "message": "signal weak",
                        "modality": "ultrasound",
                    }
                ],
                "soft_metrics": {"us_peak": 42.0},
            }
        ),
        encoding="utf-8",
    )
    (reports / "device_status.csv").write_text(
        "modality,health_status,actual_sample_rate_hz,persisted_item_count,"
        "dropped_item_count,sequence_gap_count,fault\n"
        "imu,HEALTHY,100,100,0,0,\n",
        encoding="utf-8",
    )
    (reports / "sync_check.csv").write_text(
        "status,quality,trigger_count,pulse_event_count,pretrigger_duration_s,"
        "formal_duration_s,source_device,confidence\n"
        "TRIGGERED,PASS,1,1,0.1,2.0,force_plate,0.99\n",
        encoding="utf-8",
    )
    (reports / "warnings.txt").write_text(
        "[WARNING] US_SIGNAL_WEAK", encoding="utf-8"
    )

    relative_paths = (
        "raw/ultrasound.bin",
        "raw/ultrasound.meta.json",
        "raw/ultrasound.idx",
        "raw/imu.h5",
        "raw/encoder.h5",
        "raw/sync_pulse.h5",
        "reports/quality_report.json",
        "reports/device_status.csv",
        "reports/sync_check.csv",
        "reports/warnings.txt",
    )
    modalities = {
        "raw/ultrasound.bin": "ultrasound",
        "raw/ultrasound.meta.json": "ultrasound",
        "raw/ultrasound.idx": "ultrasound",
        "raw/imu.h5": "imu",
        "raw/encoder.h5": "encoder",
        "raw/sync_pulse.h5": "sync_pulse",
        "reports/quality_report.json": "trial",
        "reports/device_status.csv": "trial",
        "reports/sync_check.csv": "sync_pulse",
        "reports/warnings.txt": "trial",
    }
    artifacts = [
        ManifestArtifact(
            artifact_uuid=uuid4(),
            trial_uuid=trial_uuid,
            modality=modalities[relative],
            kind=(
                ArtifactKind.RAW
                if relative.startswith("raw/")
                else ArtifactKind.REPORT
            ),
            media_type="application/octet-stream",
            relative_path=relative,
            size_bytes=(trial_root / relative).stat().st_size,
            sha256=sha256_file(trial_root / relative),
            created_at_utc=now,
            finalized_at_utc=now + timedelta(seconds=3),
        )
        for relative in relative_paths
    ]
    manifest = TrialManifest(
        project_uuid=project_uuid,
        project_code="T",
        project_name="测试",
        subject_uuid=subject_uuid,
        subject_code="001",
        session_uuid=session_uuid,
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code="WALK_LEVEL",
            condition_name="Level walking",
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=now,
        ),
        timing=TrialTiming(
            started_at_utc=now,
            stopped_at_utc=now + timedelta(seconds=2),
            finalized_at_utc=now + timedelta(seconds=3),
            start_host_monotonic_ns=1_000_000_000,
            stop_host_monotonic_ns=3_000_000_000,
            finalize_host_monotonic_ns=3_100_000_000,
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
        artifacts=artifacts,
        quality=QualitySummary(
            computed_grade=QualityGrade.B,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-test",
            assessed_at_utc=now + timedelta(seconds=3),
            issues=[
                QualityIssue(
                    code="US_SIGNAL_WEAK",
                    severity=QualityIssueSeverity.WARNING,
                    message="signal weak",
                    modality="ultrasound",
                )
            ],
        ),
    )
    manifest_path = save_manifest(trial_root / "manifest.json", manifest)
    write_checksum_manifest(trial_root, [*relative_paths, "manifest.json"])
    return manifest_path, manifest


def test_playback_is_bounded_and_contains_all_four_modalities(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)

    playback = load_trial_playback(
        manifest_path,
        data_root=tmp_path,
        max_signal_points=20,
        max_ultrasound_frames=10,
        max_ultrasound_depth_points=16,
    )

    assert playback.ultrasound is not None
    assert playback.ultrasound.waterfall.shape == (4, 10, 16)
    assert playback.ultrasound.latest_frame.shape == (4, 16)
    assert playback.imu is not None and playback.imu.values.shape == (20, 6)
    assert playback.encoder is not None and playback.encoder.values.shape == (20, 2)
    assert playback.sync is not None and playback.sync.values.shape == (20, 1)
    np.testing.assert_allclose(playback.sync_trigger_times_s, [0.5])


def test_ultrasound_downsampling_preserves_source_frame_time_offsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ultrasound.bin"
    with BlockBinaryWriter(
        path,
        dtype=np.uint16,
        sample_shape=(1, 8),
        metadata={
            "clock_domain": "ultrasound_clock",
            "channels": ["ch_1"],
            "nominal_frame_rate_hz": 10.0,
        },
    ) as writer:
        first = np.broadcast_to(
            np.arange(6, dtype=np.uint16)[:, None, None], (6, 1, 8)
        ).copy()
        second = np.broadcast_to(
            (100 + np.arange(6, dtype=np.uint16))[:, None, None], (6, 1, 8)
        ).copy()
        writer.append(
            first,
            host_monotonic_ns=1_000_000_000,
            host_utc_ns=10_000_000_000,
        )
        writer.append(
            second,
            host_monotonic_ns=2_000_000_000,
            host_utc_ns=11_000_000_000,
        )
        meta_path = writer.meta_path
        index_path = writer.index_path

    playback = _read_ultrasound(
        path,
        meta_path=meta_path,
        index_path=index_path,
        formal_t0_ns=1_000_000_000,
        max_frames=4,
        max_depth_points=8,
        idle_check=lambda: None,
    )

    # Each six-frame source block is reduced to original positions [0, 5].
    # The second retained frame is therefore 0.5 s, not 0.1 s, after its
    # block timestamp.
    np.testing.assert_allclose(playback.time_s, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(
        playback.waterfall[0, :, 0], [0, 5, 100, 105]
    )
    assert playback.source_frame_count == 12


def test_checksum_quality_and_full_statistics_are_manifest_driven(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)

    checksum = verify_trial_checksums(manifest_path, data_root=tmp_path)
    assert checksum.passed
    assert any(item.relative_path == "manifest.json" for item in checksum.items)

    quality = load_quality_audit(manifest_path, data_root=tmp_path)
    assert quality.computed_grade == "B"
    assert quality.issues[0]["code"] == "US_SIGNAL_WEAK"
    assert quality.devices[0]["health_status"] == "HEALTHY"
    assert quality.sync_checks[0]["quality"] == "PASS"
    assert "US_SIGNAL_WEAK" in quality.warnings_text

    statistics = compute_full_statistics(tmp_path)
    assert statistics.projects == 1
    assert statistics.subjects == 1
    assert statistics.sessions == 1
    assert statistics.trials == statistics.finalized_trials == 1
    assert statistics.artifact_count == 10
    assert statistics.by_modality["ultrasound"]["artifact_count"] == 3


def test_human_quality_reviews_are_append_only_hash_chained_and_visible(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _build_finalized_trial(tmp_path)
    manifest_before = manifest_path.read_bytes()
    checksums_before = (manifest_path.parent / "checksums.sha256").read_bytes()

    first = append_quality_review(
        tmp_path,
        manifest_path,
        reviewed_grade="C",
        reviewer="reviewer-01",
        reason="Ultrasound probe shifted near the end.",
    )
    second = append_quality_review(
        tmp_path,
        manifest_path,
        reviewed_grade=QualityGrade.B,
        reviewer="reviewer-02",
        reason="Independent review accepted the formal window.",
    )

    records = list_quality_reviews(tmp_path, manifest_path)
    assert [item.record.review_uuid for item in records] == [
        first.record.review_uuid,
        second.record.review_uuid,
    ]
    assert records[1].record.previous_record_sha256 == records[0].sha256
    audit = load_quality_audit(manifest_path, data_root=tmp_path)
    assert audit.reviewed_grade == "B"
    assert audit.reviewed_by == "reviewer-02"
    assert audit.review_count == 2
    assert "Independent review" in (audit.review_reason or "")
    assert manifest_path.read_bytes() == manifest_before
    assert (manifest_path.parent / "checksums.sha256").read_bytes() == checksums_before
    assert load_manifest(manifest_path) == manifest


def test_quality_review_rejects_active_collection_and_tampered_chain(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _build_finalized_trial(tmp_path)
    with AcquisitionLock(tmp_path, manifest.trial_uuid):
        with pytest.raises(QualityReviewError, match="Collector"):
            append_quality_review(
                tmp_path,
                manifest_path,
                reviewed_grade="C",
                reviewer="reviewer-01",
                reason="must not be written while collecting",
            )

    saved = append_quality_review(
        tmp_path,
        manifest_path,
        reviewed_grade="C",
        reviewer="reviewer-01",
        reason="first review",
    )
    document = json.loads(saved.path.read_text(encoding="utf-8"))
    document["reason"] = "tampered"
    saved.path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(QualityReviewError, match="内容|链"):
        append_quality_review(
            tmp_path,
            manifest_path,
            reviewed_grade="B",
            reviewer="reviewer-02",
            reason="second review",
        )


@pytest.mark.parametrize(
    "package_suffix",
    [".RECORDING", ".PaRtIaL", ".AbOrTeD", ".BUILDING"],
)
def test_quality_review_rejects_mixed_case_unpublished_package_components(
    tmp_path: Path,
    package_suffix: str,
) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)
    unsafe = manifest_path.parent.with_name(
        manifest_path.parent.name + package_suffix
    ) / "manifest.json"

    with pytest.raises(QualityReviewError, match="recording|partial"):
        list_quality_reviews(tmp_path, unsafe)


def test_checksum_reports_mutation_and_tools_reject_active_paths(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)
    imu_path = manifest_path.parent / "raw/imu.h5"
    with imu_path.open("ab") as stream:
        stream.write(b"changed")
    report = verify_trial_checksums(manifest_path, data_root=tmp_path)
    assert not report.passed
    assert next(item for item in report.items if item.relative_path == "raw/imu.h5").message == "SHA-256 不匹配"

    recording = manifest_path.parent.with_name(manifest_path.parent.name + ".recording")
    manifest_path.parent.rename(recording)
    with pytest.raises(DataStudioToolError, match="recording"):
        load_trial_playback(recording / "manifest.json", data_root=tmp_path)


@pytest.mark.parametrize(
    "package_suffix",
    [".RECORDING", ".PaRtIaL", ".AbOrTeD", ".BUILDING"],
)
def test_local_tools_reject_mixed_case_unpublished_package_components(
    tmp_path: Path,
    package_suffix: str,
) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)
    unsafe = manifest_path.parent.with_name(
        manifest_path.parent.name + package_suffix
    ) / "manifest.json"

    with pytest.raises(DataStudioToolError, match="recording|partial"):
        load_trial_playback(unsafe, data_root=tmp_path)


def test_disk_heavy_tools_abort_when_collector_is_active(tmp_path: Path) -> None:
    manifest_path, manifest = _build_finalized_trial(tmp_path)
    with AcquisitionLock(tmp_path, manifest.trial_uuid):
        with pytest.raises(AcquisitionBecameActiveError):
            load_trial_playback(manifest_path, data_root=tmp_path)
        with pytest.raises(AcquisitionBecameActiveError):
            verify_trial_checksums(manifest_path, data_root=tmp_path)
        with pytest.raises(AcquisitionBecameActiveError):
            compute_full_statistics(tmp_path)


def test_checksum_process_worker_uses_spawn_and_returns_pickled_report(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)
    worker = DataStudioProcessWorker(
        "checksum",
        manifest_path=str(manifest_path),
        data_root=str(tmp_path),
    )
    worker.start()
    result: tuple[str, object] | None = None
    deadline = time.monotonic() + 20.0
    while result is None and time.monotonic() < deadline:
        result = worker.poll_result()
        time.sleep(0.02)
    assert result is not None
    status, payload = result
    assert status == "completed"
    assert isinstance(payload, ChecksumReport)
    assert payload.passed
    worker.join(5.0)
    assert worker.exitcode == 0
    worker.close()


def test_playback_process_worker_returns_bounded_numpy_payload(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_finalized_trial(tmp_path)
    worker = DataStudioProcessWorker(
        "playback",
        manifest_path=str(manifest_path),
        data_root=str(tmp_path),
        max_signal_points=12,
        max_ultrasound_frames=8,
        max_ultrasound_depth_points=16,
    )
    worker.start()
    result: tuple[str, object] | None = None
    deadline = time.monotonic() + 20.0
    while result is None and time.monotonic() < deadline:
        result = worker.poll_result()
        time.sleep(0.02)
    assert result is not None
    status, payload = result
    assert status == "completed"
    assert isinstance(payload, TrialPlayback)
    assert payload.ultrasound is not None
    assert payload.ultrasound.waterfall.shape == (4, 8, 16)
    worker.join(5.0)
    assert worker.exitcode == 0
    worker.close()


def test_local_result_dialogs_render_all_published_evidence(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication(["test-local-dialogs"])
    manifest_path, _manifest = _build_finalized_trial(tmp_path)
    playback = load_trial_playback(manifest_path, data_root=tmp_path)
    checksum = verify_trial_checksums(manifest_path, data_root=tmp_path)
    statistics = compute_full_statistics(tmp_path)
    quality = load_quality_audit(manifest_path, data_root=tmp_path)

    dialogs = [
        PlaybackDialog(playback),
        ChecksumDialog(checksum),
        FullStatisticsDialog(statistics),
        QualityAuditDialog(quality),
    ]
    assert dialogs[0].findChild(QTabWidget, "playback_tabs") is not None
    slider = dialogs[0].findChild(QSlider, "playback_timeline")
    play_button = dialogs[0].findChild(QPushButton, "playback_play_pause")
    speed = dialogs[0].findChild(QComboBox, "playback_speed")
    channel = dialogs[0].findChild(QComboBox, "playback_ultrasound_channel")
    assert slider is not None and play_button is not None and speed is not None
    assert channel is not None and channel.count() == 4
    slider.setValue(5000)
    assert dialogs[0]._time_min < dialogs[0]._current_time < dialogs[0]._time_max
    play_button.click()
    assert dialogs[0]._timer.isActive()
    play_button.click()
    assert not dialogs[0]._timer.isActive()
    assert dialogs[1].findChild(QTableWidget, "checksum_results") is not None
    assert dialogs[2].findChild(QTableWidget, "full_statistics_table") is not None
    assert dialogs[3].findChild(QTableWidget, "quality_sync") is not None
    for dialog in dialogs:
        dialog.close()
    app.processEvents()


def test_window_polls_spawned_checksum_worker_without_blocking_gui(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["test-window-process-tool"])
    _build_finalized_trial(tmp_path)
    window = DataStudioWindow(tmp_path, autostart_refresh=False)
    refreshes: list[bool] = []
    window.refresh_finished.connect(refreshes.append)
    window.refresh_catalog()
    deadline = time.monotonic() + 10.0
    while not refreshes and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert refreshes == [True]
    trial_item = (
        window.tree_widget.topLevelItem(0).child(0).child(0).child(0)
    )
    window.tree_widget.setCurrentItem(trial_item)

    completions: list[tuple[str, bool]] = []
    window.local_tool_finished.connect(
        lambda name, succeeded: completions.append((name, succeeded))
    )
    window.verify_selected_trial()
    assert window.isEnabled()
    deadline = time.monotonic() + 20.0
    while not completions and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert completions == [("SHA-256 校验", True)]
    assert window.findChild(ChecksumDialog, "checksum_dialog") is not None
    window.close()
    app.processEvents()
