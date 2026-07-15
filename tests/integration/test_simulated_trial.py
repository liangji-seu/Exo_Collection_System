from __future__ import annotations

import json
import time

import h5py
import pytest

import exo_collection.orchestration.simulated as simulated_module
from exo_collection.acquisition.messages import WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository
from exo_collection.domain.states import TrialState
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.orchestration.simulated import run_simulated_trial
from exo_collection.readers.binary_block import BlockBinaryReader
from exo_collection.readers.binary_block import scan_binary_file
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import verify_checksum_manifest
from exo_collection.storage.manifest import load_manifest


def test_simulated_trial_produces_complete_immutable_package(tmp_path) -> None:
    request = TrialRunRequest(data_root=tmp_path, duration_s=0.5)
    result = run_simulated_trial(request)

    manifest = load_manifest(result.manifest_path)
    assert manifest.state is TrialState.FINALIZED
    assert result.quality_grade == "A"
    assert result.pulse_event_count >= 2
    assert not list(tmp_path.rglob("*.partial"))
    assert not list(tmp_path.rglob("*.recording"))
    assert all(verify_checksum_manifest(result.trial_directory / "checksums.sha256").values())
    assert {item.modality for item in manifest.modalities} == {
        "ultrasound",
        "imu",
        "encoder",
        "sync_pulse",
    }
    for artifact in manifest.artifacts:
        assert (result.trial_directory / artifact.relative_path).stat().st_size == artifact.size_bytes

    with BlockBinaryReader(result.trial_directory / "raw/ultrasound.bin") as reader:
        blocks = list(reader.iter_blocks())
        assert sum(block.header.sample_count for block in blocks) == result.modality_counts["ultrasound"]
        assert all(block.header.host_monotonic_ns > 0 for block in blocks)
        assert all(block.header.host_utc_ns > 0 for block in blocks)

    for modality in ("imu", "encoder", "sync_pulse"):
        with h5py.File(result.trial_directory / f"raw/{modality}.h5", "r") as file:
            assert bool(file.attrs["closed_cleanly"])
            assert file["samples/data"].shape[0] == result.modality_counts[modality]
            assert file["samples/sample_index"].shape[0] == result.modality_counts[modality]
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
