from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
from threading import Event, Timer
import time

import h5py
import numpy as np
import pytest

from exo_collection import __version__
import exo_collection.orchestration.simulated as simulated_module
from exo_collection.acquisition.messages import WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.adapters.base import AdapterError
from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository
from exo_collection.domain.events import FrameBatch
from exo_collection.domain.models import ArtifactKind
from exo_collection.domain.states import TrialState
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.orchestration.simulated import run_simulated_trial
from exo_collection.quality import DiskSpaceEvidence, InsufficientDiskSpaceError
from exo_collection.readers.binary_block import BlockBinaryReader
from exo_collection.readers.binary_block import scan_binary_file
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file, verify_checksum_manifest
from exo_collection.storage.manifest import MANIFEST_SCHEMA_VERSION, load_manifest
from exo_collection.writers.block_binary_process import BlockBinaryWriterProcess
from exo_collection.writers.hdf5_signal import HDF5_SIGNAL_VERSION, Hdf5SignalWriter


def _raw_ultrasound_frame_batch(channel: int | None) -> FrameBatch:
    return FrameBatch(
        device_id="raw_ultrasound",
        modality="ultrasound",
        clock_domain="host",
        first_frame_index=0,
        frame_count=1,
        sequence_number=0,
        data=np.zeros((1, 1000), dtype=np.uint8),
        channel=channel,
        tail_flags=1 if channel is not None else 0,
    )


def test_recording_preview_rate_limit_is_independent_per_raw_ultrasound_channel() -> None:
    last_sent: dict[tuple[str, int | None], int] = {}
    channel_events = [_raw_ultrasound_frame_batch(channel) for channel in range(4)]

    assert all(
        simulated_module._acquisition_preview_is_due(
            "ultrasound",
            event,
            now_ns=1_000_000_000,
            last_sent_by_stream=last_sent,
            interval_ns=66_000_000,
        )
        for event in channel_events
    )
    assert set(last_sent) == {
        ("ultrasound", 0),
        ("ultrasound", 1),
        ("ultrasound", 2),
        ("ultrasound", 3),
    }

    assert not simulated_module._acquisition_preview_is_due(
        "ultrasound",
        channel_events[0],
        now_ns=1_010_000_000,
        last_sent_by_stream=last_sent,
        interval_ns=66_000_000,
    )
    assert simulated_module._acquisition_preview_is_due(
        "ultrasound",
        channel_events[0],
        now_ns=1_066_000_000,
        last_sent_by_stream=last_sent,
        interval_ns=66_000_000,
    )


def test_recording_preview_rate_limit_keeps_batched_ultrasound_on_one_stream() -> None:
    last_sent: dict[tuple[str, int | None], int] = {}
    first = _raw_ultrasound_frame_batch(None)
    second = _raw_ultrasound_frame_batch(None)

    assert simulated_module._acquisition_preview_is_due(
        "ultrasound",
        first,
        now_ns=2_000_000_000,
        last_sent_by_stream=last_sent,
        interval_ns=66_000_000,
    )
    assert not simulated_module._acquisition_preview_is_due(
        "ultrasound",
        second,
        now_ns=2_010_000_000,
        last_sent_by_stream=last_sent,
        interval_ns=66_000_000,
    )
    assert last_sent == {("ultrasound", None): 2_000_000_000}


def test_simulated_trial_produces_complete_immutable_package(tmp_path) -> None:
    request = TrialRunRequest(data_root=tmp_path, duration_s=0.5)
    result = run_simulated_trial(request)

    manifest = load_manifest(result.manifest_path)
    assert manifest.state is TrialState.FINALIZED
    assert result.trial_directory.relative_to(tmp_path).parts[0] == "T"
    assert result.trial_directory.relative_to(tmp_path).parts[1] == "001"
    assert manifest.project_code == "T"
    assert manifest.project_name == "测试"
    assert manifest.subject_code == "001"
    assert result.quality_grade == "A"
    assert result.pulse_event_count >= 2
    assert not list(tmp_path.rglob("*.partial"))
    assert not list(tmp_path.rglob("*.recording"))
    checksum_results = verify_checksum_manifest(
        result.trial_directory / "checksums.sha256"
    )
    assert all(checksum_results.values())
    assert {item.modality for item in manifest.modalities} == {
        "ultrasound",
        "imu",
        "encoder",
        "sync_pulse",
    }
    for artifact in manifest.artifacts:
        assert (result.trial_directory / artifact.relative_path).stat().st_size == artifact.size_bytes

    artifact_by_path = {artifact.relative_path: artifact for artifact in manifest.artifacts}
    configuration_path = result.trial_directory / "derived/configuration_snapshot.json"
    configuration_sha256 = sha256_file(configuration_path)
    assert manifest.configuration.content_sha256 == configuration_sha256
    assert (
        artifact_by_path["derived/configuration_snapshot.json"].sha256
        == configuration_sha256
    )
    qc_paths = {
        "reports/device_status.csv",
        "reports/imu_encoder_preview.png",
        "reports/sync_check.csv",
        "reports/sync_manifest.json",
        "reports/us_quality_preview.png",
        "reports/warnings.txt",
    }
    assert qc_paths <= artifact_by_path.keys()
    assert qc_paths <= checksum_results.keys()
    assert all(artifact_by_path[path].kind is ArtifactKind.REPORT for path in qc_paths)
    assert all(artifact_by_path[path].immutable for path in qc_paths)
    assert all(artifact_by_path[path].source_artifact_uuids for path in qc_paths)
    quality_rules_path = result.trial_directory / "derived/quality_rules_snapshot.json"
    quality_rules_sha256 = sha256_file(quality_rules_path)
    quality_rules_artifact = artifact_by_path["derived/quality_rules_snapshot.json"]
    assert quality_rules_artifact.kind is ArtifactKind.DERIVED
    assert quality_rules_artifact.sha256 == quality_rules_sha256
    assert quality_rules_artifact.metadata["content_sha256"] == quality_rules_sha256

    raw_ultrasound = artifact_by_path["raw/ultrasound.bin"]
    raw_imu = artifact_by_path["raw/imu.h5"]
    raw_encoder = artifact_by_path["raw/encoder.h5"]
    us_preview = artifact_by_path["reports/us_quality_preview.png"]
    signal_preview = artifact_by_path["reports/imu_encoder_preview.png"]
    assert us_preview.source_artifact_uuids == [raw_ultrasound.artifact_uuid]
    assert set(signal_preview.source_artifact_uuids) == {
        raw_imu.artifact_uuid,
        raw_encoder.artifact_uuid,
    }
    for relative_path, expected_size in (
        ("reports/us_quality_preview.png", (1000, 720)),
        ("reports/imu_encoder_preview.png", (1000, 600)),
    ):
        png_header = (result.trial_directory / relative_path).read_bytes()[:24]
        assert png_header[:8] == b"\x89PNG\r\n\x1a\n"
        assert png_header[12:16] == b"IHDR"
        assert struct.unpack(">II", png_header[16:24]) == expected_size
        artifact = artifact_by_path[relative_path]
        assert artifact.media_type == "image/png"
        assert artifact.sha256 == sha256_file(result.trial_directory / relative_path)
        assert artifact.metadata["generated_after_acquisition_stop"] is True
        assert artifact.metadata["raw_file_scan_performed"] is False

    statistics = json.loads(
        (result.trial_directory / "derived/statistics.json").read_text(
            encoding="utf-8"
        )
    )
    quality_report = json.loads(
        (result.trial_directory / "reports/quality_report.json").read_text(
            encoding="utf-8"
        )
    )
    us_metrics = statistics["soft_quality_metrics"]["ultrasound"]
    assert us_metrics["hard_thresholds_applied"] is False
    assert us_metrics["frames_retained"] <= us_metrics["retention_capacity_frames"]
    assert us_metrics["frames_retained"] > 0
    assert us_metrics["channel_count"] == 4
    assert us_metrics["mean_intensity"] > 0
    assert us_metrics["standard_deviation"] > 0
    assert 0 <= us_metrics["saturation_fraction"] <= 1
    assert len(us_metrics["channels"]) == 4
    assert quality_report["soft_metrics"] == statistics["soft_quality_metrics"]
    assert quality_report["algorithm_version"] == "exo-quality-rules-1.0.0"
    assert quality_report["rules_snapshot_sha256"] == quality_rules_sha256
    assert quality_report["unassessed_rule_count"] > 0
    assert quality_report["rule_count"] == len(quality_report["rule_results"])
    required_rule_results = [
        item
        for item in quality_report["rule_results"]
        if item["required_for_grade_a"]
    ]
    assert required_rule_results
    assert all(item["status"] == "PASS" for item in required_rule_results)
    assert statistics["quality_assessment"]["rules_snapshot_sha256"] == (
        quality_rules_sha256
    )
    assert statistics["quality_assessment"]["computed_grade"] == "A"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    assert configuration["quality_assessment_configuration"][
        "rules_snapshot_sha256"
    ] == quality_rules_sha256
    assert manifest.quality.algorithm_version == "exo-quality-rules-1.0.0"
    assert manifest.quality.metric_count == quality_report["rule_count"]

    with (result.trial_directory / "reports/device_status.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        device_rows = list(csv.DictReader(stream))
    assert {row["modality"] for row in device_rows} == {
        "ultrasound",
        "imu",
        "encoder",
        "sync_pulse",
    }
    assert all(row["health_status"] == "HEALTHY" for row in device_rows)
    assert all(row["adapter_state"] == "stopped" for row in device_rows)
    assert all(float(row["actual_sample_rate_hz"]) > 0 for row in device_rows)
    assert all(row["fault"] == "" for row in device_rows)
    assert {
        row["modality"]: int(row["persisted_item_count"]) for row in device_rows
    } == result.modality_counts

    with (result.trial_directory / "reports/sync_check.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        sync_rows = list(csv.DictReader(stream))
    assert len(sync_rows) == 1
    sync_row = sync_rows[0]
    assert sync_row["status"] == "TRIGGERED"
    assert sync_row["quality"] == "PASS"
    assert int(sync_row["trigger_count"]) == result.trigger_count
    assert int(sync_row["pulse_event_count"]) == result.pulse_event_count
    assert int(sync_row["first_trigger_host_monotonic_ns"]) == (
        result.first_trigger_host_monotonic_ns
    )
    assert sync_row["first_trigger_time_utc"].endswith("Z")
    assert sync_row["source_device"]
    sync_manifest = json.loads(
        (result.trial_directory / "reports/sync_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert sync_manifest["edge_count"] == result.pulse_event_count
    assert len(sync_manifest["edges"]) == result.pulse_event_count
    assert sync_manifest["complete_pulse_count"] >= 1
    assert sync_manifest["clock_mappings"]
    assert all(
        "mapped_host_utc_ns" in edge and "within_formal_window" in edge
        for edge in sync_manifest["edges"]
    )
    assert any(edge["accepted_as_formal_t0"] for edge in sync_manifest["edges"])

    warnings_text = (result.trial_directory / "reports/warnings.txt").read_text(
        encoding="utf-8"
    )
    assert f"trial_uuid: {result.trial_uuid}" in warnings_text
    assert "quality_grade: A" in warnings_text
    assert "No warnings or errors were detected." in warnings_text

    with BlockBinaryReader(result.trial_directory / "raw/ultrasound.bin") as reader:
        blocks = list(reader.iter_blocks())
        assert sum(block.header.sample_count for block in blocks) == result.modality_counts["ultrasound"]
        assert all(block.header.host_monotonic_ns > 0 for block in blocks)
        assert all(block.header.host_utc_ns > 0 for block in blocks)

    embedded_trial_metadata: dict[str, dict[str, object]] = {}
    for modality in ("imu", "encoder", "sync_pulse"):
        with h5py.File(result.trial_directory / f"raw/{modality}.h5", "r") as file:
            assert bool(file.attrs["closed_cleanly"])
            assert file["samples/data"].shape[0] == result.modality_counts[modality]
            assert file["samples/sample_index"].shape[0] == result.modality_counts[modality]
            trial_metadata = json.loads(file["metadata/trial"].asstr()[()])
            embedded_trial_metadata[modality] = trial_metadata
            assert trial_metadata["schema_version"] == "1.1.0"
            assert trial_metadata["project_uuid"] == str(manifest.project_uuid)
            assert trial_metadata["subject_uuid"] == str(manifest.subject_uuid)
            assert trial_metadata["session_uuid"] == str(manifest.session_uuid)
            assert trial_metadata["trial_uuid"] == str(result.trial_uuid)
            assert trial_metadata["project_code"] == "T"
            assert trial_metadata["subject_code"] == "001"
            assert trial_metadata["condition"]["condition_code"] == "WALK_LEVEL"
            assert "experiment_metadata" in trial_metadata
            assert trial_metadata["versions"] == {
                "application": "Exo Collector",
                "application_version": __version__,
                "condition_definition_version": "1.0.0",
                "configuration_version": "1.0.0",
                "core_version": __version__,
                "hdf5_signal_format_version": HDF5_SIGNAL_VERSION,
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "protocol_version": "1.0.0",
            }
            assert trial_metadata["clock_policy"] == {
                "device_clock_mapping": (
                    "post_acquisition_affine_mapping_to_host_monotonic"
                ),
                "formal_t0": (
                    "first_qualified_sync_rising_edge_or_recording_gate_start"
                ),
                "host_monotonic_api": "time.perf_counter_ns",
                "per_sample_host_monotonic_timestamps": True,
                "pretrigger_raw_data_preserved": True,
                "primary_timeline": "host_monotonic_ns",
                "utc_role": "audit_only_not_interval_measurement",
            }
    assert embedded_trial_metadata["imu"] == embedded_trial_metadata["encoder"]
    assert embedded_trial_metadata["imu"] == embedded_trial_metadata["sync_pulse"]
    with h5py.File(result.trial_directory / "raw/sync_pulse.h5", "r") as file:
        assert file["events/records"].shape[0] >= 2

    with Catalog(tmp_path / "catalog.sqlite3") as catalog:
        repository = CatalogRepository(catalog)
        statistics = repository.statistics()
        tree = repository.tree()
    assert statistics["trial_count"] == 1
    assert statistics["finalized_count"] == 1
    assert len(tree[0]["children"][0]["children"][0]["children"][0]["children"]) == len(
        manifest.artifacts
    )


def test_collector_worker_runs_acquisition_outside_ui_process(tmp_path) -> None:
    request = TrialRunRequest(data_root=tmp_path, duration_s=0.4)
    worker = CollectorWorker(request)
    worker.start()
    events = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        events.extend(worker.poll_events())
        if any(
            event.event_type in {WorkerEventType.COMPLETED, WorkerEventType.FAILED}
            for event in events
        ):
            break
        time.sleep(0.02)
    exitcode = worker.join(timeout=10)
    events.extend(worker.poll_events())
    try:
        failures = [event for event in events if event.event_type is WorkerEventType.FAILED]
        assert not failures, failures[0].payload.get("traceback") if failures else ""
        assert exitcode == 0
        completed = [event for event in events if event.event_type is WorkerEventType.COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["state"] == "FINALIZED"
        assert any(event.event_type is WorkerEventType.PREVIEW for event in events)
        assert any(event.event_type is WorkerEventType.HEALTH for event in events)
        ultrasound_previews = [
            event
            for event in events
            if event.event_type is WorkerEventType.PREVIEW
            and event.modality == "ultrasound"
        ]
        assert ultrasound_previews
        assert ultrasound_previews[-1].payload["geometry"] == "a_line"
        assert ultrasound_previews[-1].payload["channel_count"] == 4
        assert len(ultrasound_previews[-1].payload["format_metrics"]) == 4
        assert all(
            metric["dtype"] == "uint16"
            and metric["nonfinite_fraction"] == 0.0
            and metric["all_zero"] is False
            for metric in ultrasound_previews[-1].payload["format_metrics"]
        )
        shared_marker = ultrasound_previews[-1].payload["shared_preview"]
        assert shared_marker["channel_count"] == 4
        assert shared_marker["generation"] % 2 == 0
        assert shared_marker["observed_generation"] == shared_marker["generation"]
        assert len(ultrasound_previews[-1].payload["channels"]) == 4
        assert all(
            len(channel) == 1000
            for channel in ultrasound_previews[-1].payload["channels"]
        )
        signal_previews = [
            event
            for event in events
            if event.event_type is WorkerEventType.PREVIEW
            and event.modality in {"imu", "encoder"}
            and "values" in event.payload
        ]
        assert signal_previews
        for preview in signal_previews:
            marker = preview.payload["shared_preview"]
            assert marker["observed_generation"] == marker["generation"]
            assert len(preview.payload["channels"]) == marker["channel_count"]
            assert len(preview.payload["labels"]) == marker["channel_count"]
            assert all(
                len(channel) == marker["points_per_channel"]
                for channel in preview.payload["channels"]
            )
        latest_imu = next(
            event for event in reversed(signal_previews) if event.modality == "imu"
        )
        latest_encoder = next(
            event for event in reversed(signal_previews) if event.modality == "encoder"
        )
        assert latest_imu.payload["labels"] == [
            "imu_trunk_acc_x",
            "imu_trunk_acc_y",
            "imu_trunk_acc_z",
            "imu_left_acc_x",
            "imu_left_acc_y",
            "imu_left_acc_z",
            "imu_right_acc_x",
            "imu_right_acc_y",
            "imu_right_acc_z",
        ]
        assert latest_encoder.payload["labels"] == [
            "left_position",
            "right_position",
        ]
    finally:
        worker.close()


def test_collector_worker_exits_when_parent_does_not_consume_telemetry(tmp_path) -> None:
    request = TrialRunRequest(data_root=tmp_path, duration_s=1.5)
    worker = CollectorWorker(request, queue_capacity=64)
    worker.start()

    # Deliberately do not poll either queue while acquisition runs.  The lossy
    # telemetry pipe will fill beyond the Windows pipe buffer; it must not keep
    # an otherwise completed child process alive during feeder-thread shutdown.
    exitcode = worker.join(timeout=20)
    try:
        assert exitcode == 0
        events = worker.poll_events(limit=200)
        completed = [
            event for event in events if event.event_type is WorkerEventType.COMPLETED
        ]
        assert len(completed) == 1
        assert completed[0].payload["state"] == "FINALIZED"
    finally:
        if worker.is_alive:
            # Keep a failing regression test from leaking a non-daemon process.
            deadline = time.monotonic() + 5
            while worker.is_alive and time.monotonic() < deadline:
                worker.poll_events(limit=1000)
                worker.join(timeout=0.05)
        if not worker.is_alive:
            worker.close()


def test_finalization_failure_is_recoverable_and_preserves_original_error(
    tmp_path, monkeypatch
) -> None:
    request = TrialRunRequest(data_root=tmp_path, duration_s=0.2)

    def fail_finalization(*_args, **_kwargs):
        raise OSError("forced-finalize-sentinel")

    monkeypatch.setattr(simulated_module, "finalize_trial_package", fail_finalization)
    with pytest.raises(OSError, match="forced-finalize-sentinel"):
        run_simulated_trial(request)

    recordings = list(tmp_path.rglob("*.recording"))
    assert len(recordings) == 1
    records = [
        json.loads(line)
        for line in (recordings[0] / "logs/trial.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["event_type"] == "trial_publication_intent" for record in records)
    assert not any(
        record.get("to_state") == TrialState.FINALIZED.value for record in records
    )
    failure_reports = list((recordings[0] / "reports").glob("finalization-failure-*.json"))
    assert len(failure_reports) == 1
    failure = json.loads(failure_reports[0].read_text(encoding="utf-8"))
    assert failure["state"] == TrialState.FINALIZING.value
    assert failure["recovery_state"] == TrialState.RECOVERABLE.value
    assert not (tmp_path / ".collector-active.json").exists()


def test_existing_activity_lock_causes_no_trial_side_effects(tmp_path) -> None:
    request = TrialRunRequest(data_root=tmp_path, duration_s=0.1)
    with AcquisitionLock(tmp_path):
        with pytest.raises(FileExistsError, match="collector lock"):
            run_simulated_trial(request)
    assert not list(tmp_path.rglob("*.recording"))
    assert not (tmp_path / "catalog.sqlite3").exists()


def test_insufficient_disk_space_fails_before_trial_or_catalog_side_effects(
    tmp_path, monkeypatch
) -> None:
    evidence = DiskSpaceEvidence(
        path=str(tmp_path),
        free_bytes=1,
        required_free_bytes=2 * 1024**3,
    )

    def fail_preflight(*_args, **_kwargs):
        raise InsufficientDiskSpaceError(evidence)

    monkeypatch.setattr(simulated_module, "check_disk_space", fail_preflight)
    with pytest.raises(InsufficientDiskSpaceError):
        run_simulated_trial(TrialRunRequest(data_root=tmp_path, duration_s=0.1))
    assert not (tmp_path / "catalog.sqlite3").exists()
    assert not list(tmp_path.rglob("*.recording"))


def test_injected_source_gaps_produce_an_auditable_finalized_trial(tmp_path) -> None:
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=0.45,
        simulation={
            "ultrasound": {"drop_every_n_batches": 2},
            "imu": {"drop_every_n_batches": 2},
            "encoder": {"drop_every_n_batches": 2},
        },
    )
    result = run_simulated_trial(request)
    manifest = load_manifest(result.manifest_path)
    modalities = {item.modality: item for item in manifest.modalities}

    assert manifest.state is TrialState.FINALIZED
    assert result.quality_grade == "C"
    ultrasound_scan = scan_binary_file(result.trial_directory / "raw/ultrasound.bin")
    assert ultrasound_scan.is_clean
    assert ultrasound_scan.sequence_gap_count > 0
    assert modalities["ultrasound"].sequence_gap_count == ultrasound_scan.sequence_gap_count
    assert modalities["ultrasound"].last_sample_index == ultrasound_scan.headers[-1].first_sample_index

    for modality in ("imu", "encoder"):
        with h5py.File(result.trial_directory / f"raw/{modality}.h5", "r") as file:
            stored_indices = file["samples/sample_index"][:]
        assert modalities[modality].first_sample_index == int(stored_indices[0])
        assert modalities[modality].last_sample_index == int(stored_indices[-1])
        assert modalities[modality].sequence_gap_count > 0

    warnings_text = (result.trial_directory / "reports/warnings.txt").read_text(
        encoding="utf-8"
    )
    assert "quality_grade: C" in warnings_text
    assert "[ERROR] SEQUENCE_CONTINUITY (ultrasound)" in warnings_text
    assert "[ERROR] DROPPED_BATCHES (imu)" in warnings_text
    with (result.trial_directory / "reports/device_status.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        device_rows = list(csv.DictReader(stream))
    assert any(int(row["injected_dropped_batches"]) > 0 for row in device_rows)
    assert any(int(row["sequence_gap_count"]) > 0 for row in device_rows)


def test_unbounded_trial_waits_for_sync_then_ends_only_on_manual_stop(tmp_path) -> None:
    stop = Event()
    events = []
    stop_timer: Timer | None = None

    def publish(event) -> None:
        nonlocal stop_timer
        events.append(event)
        if (
            event.event_type is WorkerEventType.SYNC
            and event.payload["status"] == "TRIGGERED"
            and stop_timer is None
        ):
            stop_timer = Timer(0.15, stop.set)
            stop_timer.start()

    result = run_simulated_trial(
        TrialRunRequest(data_root=tmp_path, duration_s=None),
        stop_requested=stop,
        publish=publish,
    )
    if stop_timer is not None:
        stop_timer.join(timeout=2)

    state_sequence = [
        event.payload["state"]
        for event in events
        if event.event_type is WorkerEventType.STATE
    ]
    assert state_sequence == [
        "PREPARING",
        "READY",
        "WAITING_SYNC",
        "RECORDING",
        "STOPPING",
        "FINALIZING",
        "FINALIZED",
    ]
    recording_event_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is WorkerEventType.STATE
        and event.payload["state"] == "RECORDING"
    )
    trigger_event_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is WorkerEventType.SYNC
        and event.payload["status"] == "TRIGGERED"
    )
    assert recording_event_index < trigger_event_index
    sync_events = [
        event for event in events if event.event_type is WorkerEventType.SYNC
    ]
    assert [event.payload["status"] for event in sync_events] == [
        "WAITING_SYNC",
        "TRIGGERED",
    ]
    assert sync_events[1].payload["quality"] == "PASS"
    assert sync_events[1].payload["trigger_count"] == 1
    assert sync_events[1].payload["first_trigger_host_monotonic_ns"] == (
        result.first_trigger_host_monotonic_ns
    )
    assert sync_events[1].payload["trigger_time_utc"].endswith("Z")
    assert result.trigger_count >= 1
    assert result.duration_s >= 0.1

    manifest = load_manifest(result.manifest_path)
    assert manifest.timing.start_host_monotonic_ns == result.first_trigger_host_monotonic_ns
    statistics = json.loads(
        (result.trial_directory / "derived/statistics.json").read_text(encoding="utf-8")
    )
    assert statistics["duration_s"] == pytest.approx(result.duration_s)
    assert statistics["pretrigger_duration_s"] > 0
    assert statistics["trigger_count"] >= 1


def test_stop_immediately_after_trigger_keeps_timing_order_and_marks_zero_window_invalid(
    tmp_path,
) -> None:
    stop = Event()

    def publish(event) -> None:
        if (
            event.event_type is WorkerEventType.SYNC
            and event.payload["status"] == "TRIGGERED"
        ):
            stop.set()

    result = run_simulated_trial(
        TrialRunRequest(
            data_root=tmp_path,
            duration_s=None,
            simulation={"sync_pulse": {"first_pulse_s": 0.049}},
        ),
        stop_requested=stop,
        publish=publish,
    )

    manifest = load_manifest(result.manifest_path)
    assert result.duration_s == 0.0
    assert result.quality_grade == "INVALID"
    assert manifest.timing.started_at_utc <= manifest.timing.stopped_at_utc
    assert manifest.timing.stopped_at_utc <= manifest.timing.finalized_at_utc
    assert (
        manifest.timing.start_host_monotonic_ns
        <= manifest.timing.stop_host_monotonic_ns
        <= manifest.timing.finalize_host_monotonic_ns
    )
    assert any(
        issue.code == "FORMAL_RECORDING_WINDOW"
        for issue in manifest.quality.issues
    )
    assert manifest.quality.algorithm_version == "exo-quality-rules-1.0.0"


def test_missing_sync_is_optional_and_finalizes_from_recording_gate(tmp_path) -> None:
    events = []
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=0.15,
        sync_wait_timeout_s=0.03,
        simulation={"sync_pulse": {"first_pulse_s": 10.0}},
    )

    result = run_simulated_trial(request, publish=events.append)

    assert result.state == "FINALIZED"
    assert result.first_trigger_host_monotonic_ns is None
    assert result.trigger_count == 0
    assert result.duration_s == pytest.approx(0.15, abs=0.02)
    assert result.quality_grade == "A"
    assert not list(tmp_path.rglob("*.recording"))
    manifest = load_manifest(result.manifest_path)
    assert manifest.state is TrialState.FINALIZED
    assert manifest.timing.start_host_monotonic_ns < manifest.timing.stop_host_monotonic_ns
    sync_events = [
        event for event in events if event.event_type is WorkerEventType.SYNC
    ]
    assert [event.payload["status"] for event in sync_events] == [
        "WAITING_SYNC",
        "NOT_RECEIVED",
    ]
    assert sync_events[-1].payload["quality"] == "OPTIONAL"
    assert sync_events[-1].payload["trigger_count"] == 0
    assert sync_events[-1].payload["first_trigger_host_monotonic_ns"] is None
    assert sync_events[-1].payload["formal_t0_source"] == "recording_gate_start"
    assert (
        sync_events[-1].payload["formal_start_host_monotonic_ns"]
        == manifest.timing.start_host_monotonic_ns
    )
    recording_event_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is WorkerEventType.STATE
        and event.payload["state"] == "RECORDING"
    )
    optional_sync_event_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is WorkerEventType.SYNC
        and event.payload["status"] == "NOT_RECEIVED"
    )
    assert recording_event_index < optional_sync_event_index
    state_sequence = [
        event.payload["state"]
        for event in events
        if event.event_type is WorkerEventType.STATE
    ]
    assert state_sequence[-3:] == ["STOPPING", "FINALIZING", "FINALIZED"]
    assert state_sequence.index("RECORDING") < state_sequence.index("STOPPING")

    report = json.loads(
        (result.trial_directory / "reports/sync_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "NOT_RECEIVED"
    assert report["quality"] == "OPTIONAL"
    assert report["trigger_count"] == 0
    assert report["formal_t0_source"] == "recording_gate_start"
    assert (
        report["formal_start_host_monotonic_ns"]
        == manifest.timing.start_host_monotonic_ns
    )
    assert not (result.trial_directory / "reports/sync_failure.json").exists()
    with h5py.File(result.trial_directory / "raw/sync_pulse.h5", "r") as file:
        assert bool(file.attrs["closed_cleanly"])
        assert file["samples/data"].shape[0] > 0
        assert file["events/records"].shape[0] == 0
    journal_records = [
        json.loads(line)
        for line in (result.trial_directory / "logs/trial.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(record["event_type"] == "trial_failure" for record in journal_records)
    assert any(record.get("to_state") == "FINALIZING" for record in journal_records)


def test_trigger_after_sync_wait_deadline_remains_raw_but_sync_is_optional(
    tmp_path,
) -> None:
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=0.15,
        sync_wait_timeout_s=0.1,
        simulation={"sync_pulse": {"first_pulse_s": 0.101}},
    )

    result = run_simulated_trial(request)

    assert result.state == "FINALIZED"
    assert result.trigger_count == 0
    assert result.first_trigger_host_monotonic_ns is None
    report = json.loads(
        (result.trial_directory / "reports/sync_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    statistics = json.loads(
        (result.trial_directory / "derived/statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "NOT_RECEIVED"
    assert report["quality"] == "OPTIONAL"
    assert report["formal_t0_source"] == "recording_gate_start"
    assert report["trigger_count"] == 0
    assert statistics["pulse_event_count"] >= 1
    assert not (result.trial_directory / "reports/sync_failure.json").exists()


def test_device_failure_while_waiting_sync_still_fails_with_adapter_error(
    tmp_path,
) -> None:
    events = []
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=None,
        simulation={
            "sync_pulse": {
                "disconnect_after_batches": 1,
                "first_pulse_s": 0.5,
            }
        },
    )

    with pytest.raises(AdapterError, match="injected disconnect"):
        run_simulated_trial(request, publish=events.append)

    recording = next(tmp_path.rglob("*.recording"))
    assert not (recording / "reports/sync_failure.json").exists()
    journal_records = [
        json.loads(line)
        for line in (recording / "logs/trial.jsonl.partial")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure = next(
        record for record in journal_records if record["event_type"] == "trial_failure"
    )
    assert failure["state"] == "RECORDING"
    assert failure["recovery_state"] == "ABORTED"
    assert failure["exception_type"] == "AdapterError"
    assert any(record.get("to_state") == "ABORTED" for record in journal_records)
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list((recording / "reports").glob("finalization-failure-*.json"))


def test_adapter_fault_between_health_poll_and_stop_is_never_finalized(
    tmp_path,
) -> None:
    # The first periodic health poll is immediate and the next is 0.5 s later.
    # This injected disconnect occurs at about 0.15 s, just before the formal
    # stop at about 0.21 s, so the post-stop fault check is the only reliable
    # place to catch it. These rates are the built-in simulator profile only.
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=0.2,
        simulation={
            "ultrasound": {"disconnect_after_batches": 3},
            "sync_pulse": {"first_pulse_s": 0.01},
        },
    )

    with pytest.raises(AdapterError, match="ultrasound"):
        run_simulated_trial(request)

    recordings = list(tmp_path.rglob("*.recording"))
    assert len(recordings) == 1
    recording = recordings[0]
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("checksums.sha256"))
    scan = scan_binary_file(recording / "raw/ultrasound.bin.partial")
    assert scan.is_clean
    assert scan.complete_block_count > 0
    with h5py.File(recording / "raw/imu.h5.partial", "r") as file:
        assert bool(file.attrs["closed_cleanly"])

    journal_records = [
        json.loads(line)
        for line in (recording / "logs/trial.jsonl.partial")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure = next(
        record for record in journal_records if record["event_type"] == "trial_failure"
    )
    assert failure["recovery_state"] == "RECOVERABLE"
    assert "injected disconnect" in failure["message"]
    assert failure["stop_reports"]["ultrasound"]["fault"] is not None


def test_simulated_raw_queue_saturation_is_explicit_and_never_finalized(
    tmp_path,
    monkeypatch,
) -> None:
    # Model a brief slow Writer against the deterministic simulator. This
    # validates bounded backpressure/failure propagation only; it is not a
    # benchmark or a claim about a real ultrasound device or disk.
    original_append = BlockBinaryWriterProcess.append
    delayed_once = False
    append_calls = 0

    def briefly_slow_append(self, *args, **kwargs):
        nonlocal append_calls, delayed_once
        append_calls += 1
        # Inject the stall well after the deliberately early sync pulse so the
        # test deterministically exercises recording queue-overflow semantics.
        if not delayed_once and append_calls == 10:
            delayed_once = True
            time.sleep(0.3)
        return original_append(self, *args, **kwargs)

    monkeypatch.setattr(BlockBinaryWriterProcess, "append", briefly_slow_append)
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=None,
        sync_wait_timeout_s=1.0,
        simulation={
            "ultrasound": {
                "queue_capacity": 1,
                "frame_rate_hz": 50.0,
            },
            "sync_pulse": {"first_pulse_s": 0.01},
        },
    )

    with pytest.raises(AdapterError, match="raw queue overflow"):
        run_simulated_trial(request)

    recording = next(tmp_path.rglob("*.recording"))
    assert not list(tmp_path.rglob("manifest.json"))
    journal_records = [
        json.loads(line)
        for line in (recording / "logs/trial.jsonl.partial")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure = next(
        record for record in journal_records if record["event_type"] == "trial_failure"
    )
    assert failure["recovery_state"] == "ABORTED"
    assert "raw queue overflow" in failure["message"]


def test_mid_acquisition_disk_write_error_is_explicit_and_leaves_recovery_data(
    tmp_path,
    monkeypatch,
) -> None:
    # HDF5 modalities are intentionally written in the collector-core process.
    # Inject an OS-level write failure after formal t0 to prove it cannot be
    # converted into a successful or immutable Trial.
    original_append = Hdf5SignalWriter.append_batch
    imu_batches = 0

    def fail_after_trigger(self, batch):
        nonlocal imu_batches
        if batch.modality == "imu":
            imu_batches += 1
            if imu_batches == 6:
                raise OSError("simulated mid-acquisition disk full")
        return original_append(self, batch)

    monkeypatch.setattr(Hdf5SignalWriter, "append_batch", fail_after_trigger)

    with pytest.raises(OSError, match="simulated mid-acquisition disk full"):
        run_simulated_trial(TrialRunRequest(data_root=tmp_path, duration_s=0.5))

    recording = next(tmp_path.rglob("*.recording"))
    assert not list(tmp_path.rglob("manifest.json"))
    with h5py.File(recording / "raw/imu.h5.partial", "r") as file:
        assert not bool(file.attrs["closed_cleanly"])
        assert int(file.attrs["sample_count"]) > 0
    journal_records = [
        json.loads(line)
        for line in (recording / "logs/trial.jsonl.partial")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure = next(
        record for record in journal_records if record["event_type"] == "trial_failure"
    )
    assert failure["exception_type"] == "OSError"
    assert "disk full" in failure["message"]


def test_worker_completes_without_sync_and_reports_optional_status(tmp_path) -> None:
    worker = CollectorWorker(
        TrialRunRequest(
            data_root=tmp_path,
            duration_s=0.15,
            sync_wait_timeout_s=0.03,
            simulation={"sync_pulse": {"first_pulse_s": 10.0}},
        )
    )
    worker.start()
    events = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        events.extend(worker.poll_events(limit=1000))
        if any(
            event.event_type in {WorkerEventType.COMPLETED, WorkerEventType.FAILED}
            for event in events
        ):
            break
        time.sleep(0.02)
    exitcode = worker.join(timeout=10)
    events.extend(worker.poll_events(limit=1000))
    try:
        failures = [
            event for event in events if event.event_type is WorkerEventType.FAILED
        ]
        completed = [
            event for event in events if event.event_type is WorkerEventType.COMPLETED
        ]
        assert exitcode == 0
        assert not failures
        assert len(completed) == 1
        assert completed[0].payload["state"] == "FINALIZED"
        assert completed[0].payload["first_trigger_host_monotonic_ns"] is None
        trial_directory = Path(completed[0].payload["trial_directory"])
        assert trial_directory.is_dir()
        sync_manifest = json.loads(
            (trial_directory / "reports/sync_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert sync_manifest["status"] == "NOT_RECEIVED"
        assert sync_manifest["quality"] == "OPTIONAL"
        assert sync_manifest["formal_t0_source"] == "recording_gate_start"
        assert not (trial_directory / "reports/sync_failure.json").exists()
    finally:
        if worker.is_alive:
            worker.terminate_for_recovery()
        if not worker.is_alive:
            worker.close()
