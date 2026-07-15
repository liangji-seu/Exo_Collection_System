from __future__ import annotations

import time

import h5py

from exo_collection.acquisition.messages import WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository
from exo_collection.domain.states import TrialState
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.orchestration.simulated import run_simulated_trial
from exo_collection.readers.binary_block import BlockBinaryReader
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

