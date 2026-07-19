"""Windows spawn test for persistent preview -> Collector recording IPC."""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from exo_collection.acquisition.messages import WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.apps.collector.device_preview import (
    ModalityPreviewProcessHandle,
    ProfileModalityAdapterFactory,
)
from exo_collection.orchestration.models import TrialRunRequest


MODALITIES = ("ultrasound", "imu", "encoder", "sync_pulse")


def _observe_ultrasound_channels(event, observed: set[int]) -> None:
    if event.event_type is not WorkerEventType.PREVIEW:
        return
    channel_index = event.payload.get("channel_index")
    if channel_index is not None:
        observed.add(int(channel_index))
        return
    # The simulated adapter publishes one four-channel FrameBatch, whereas
    # Raw Ethernet publishes four independent channel-tagged FrameBatch items.
    observed.update(range(int(event.payload.get("channel_count") or 0)))


@pytest.mark.skipif(os.name != "nt", reason="Windows spawn contract")
def test_persistent_preview_processes_feed_collector_without_reconnect(
    tmp_path: Path,
) -> None:
    """Exercise the exact two-child-process boundary used by the desktop UI."""

    handles: dict[str, ModalityPreviewProcessHandle] = {}
    worker: CollectorWorker | None = None
    trial_uuid: str | None = None
    recording_started: set[str] = set()
    try:
        for modality in MODALITIES:
            handle = ModalityPreviewProcessHandle(
                ProfileModalityAdapterFactory(
                    profile_key="simulated",
                    modality=modality,
                ),
                device_id=f"spawn_{modality}",
                modality=modality,
                simulated=True,
                health_poll_interval_s=0.05,
                recording_queue_size=512,
            )
            handles[modality] = handle
            handle.start()

        preview_channels_before: set[int] = set()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            for modality, handle in handles.items():
                for event in handle.poll_events(limit=200):
                    if modality == "ultrasound":
                        _observe_ultrasound_channels(
                            event, preview_channels_before
                        )
            if (
                all(
                    handle.recording_endpoint is not None
                    for handle in handles.values()
                )
                and preview_channels_before == {0, 1, 2, 3}
            ):
                break
            assert all(handle.is_alive for handle in handles.values())
            time.sleep(0.01)

        endpoints = {
            modality: handle.recording_endpoint
            for modality, handle in handles.items()
        }
        assert all(endpoint is not None for endpoint in endpoints.values())

        request = TrialRunRequest(
            data_root=tmp_path,
            duration_s=None,
            sync_wait_timeout_s=None,
            enabled_modalities=frozenset(MODALITIES),
        )
        trial_uuid = str(request.trial_uuid)
        worker = CollectorWorker(
            request,
            {name: endpoint for name, endpoint in endpoints.items() if endpoint is not None},
        )
        worker.start()
        original_pids = {name: handle._process.pid for name, handle in handles.items()}
        for modality, handle in handles.items():
            handle.begin_recording(trial_uuid)
            recording_started.add(modality)

        preview_channels_during: set[int] = set()
        worker_events = []
        record_until = time.monotonic() + 1.25
        while time.monotonic() < record_until:
            worker_events.extend(worker.poll_events(limit=500))
            for modality, handle in handles.items():
                    for event in handle.poll_events(limit=200):
                        if modality == "ultrasound":
                            _observe_ultrasound_channels(
                                event, preview_channels_during
                            )
            assert all(handle.is_alive for handle in handles.values())
            time.sleep(0.01)

        for modality, handle in handles.items():
            handle.end_recording(trial_uuid)
            recording_started.discard(modality)
        worker.request_stop()

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            worker_events.extend(worker.poll_events(limit=1000))
            for handle in handles.values():
                handle.poll_events(limit=200)
            if any(
                event.event_type in {
                    WorkerEventType.COMPLETED,
                    WorkerEventType.FAILED,
                }
                for event in worker_events
            ):
                break
            time.sleep(0.02)

        exitcode = worker.join(timeout=10.0)
        worker_events.extend(worker.poll_events(limit=1000))
        failures = [
            event
            for event in worker_events
            if event.event_type is WorkerEventType.FAILED
        ]
        completed = [
            event
            for event in worker_events
            if event.event_type is WorkerEventType.COMPLETED
        ]
        assert not failures, failures[0].payload if failures else None
        assert exitcode == 0
        assert len(completed) == 1
        assert completed[0].payload["state"] == "FINALIZED"
        assert preview_channels_before == {0, 1, 2, 3}
        assert preview_channels_during == {0, 1, 2, 3}
        assert all(handle.is_alive for handle in handles.values())
        assert {
            name: handle._process.pid for name, handle in handles.items()
        } == original_pids
        assert list(tmp_path.rglob("manifest.json"))
    finally:
        if trial_uuid is not None:
            for modality in tuple(recording_started):
                try:
                    handles[modality].end_recording(trial_uuid)
                except Exception:
                    pass
        if worker is not None:
            if worker.is_alive:
                worker.request_stop()
                worker.join(timeout=2.0)
            if worker.is_alive:
                worker.terminate_for_recovery(timeout=2.0)
            if not worker.is_alive:
                worker.close()
        for handle in handles.values():
            if handle.is_alive:
                handle.request_stop()
                handle.join(timeout=2.0)
            if handle.is_alive:
                handle.terminate(timeout=2.0)
            handle.close()
