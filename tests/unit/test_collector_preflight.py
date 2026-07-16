from __future__ import annotations

from pathlib import Path
import time

import pytest

from exo_collection.apps.collector import preflight as preflight_module
from exo_collection.apps.collector.preflight import (
    CollectorPreflightReport,
    CollectorPreflightWorker,
    run_simulated_preflight,
)


def test_simulated_preflight_exercises_lifecycle_samples_sync_and_storage(
    tmp_path: Path,
) -> None:
    report = run_simulated_preflight(
        tmp_path,
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )

    assert report.ready
    assert report.writable
    assert report.disk_free_bytes > 0
    assert report.write_probe_bytes == 1024**2
    assert report.write_probe_elapsed_s > 0
    assert report.measured_write_mib_s > 0
    assert report.minimum_write_mib_s is None
    assert set(report.devices) == {
        "ultrasound",
        "imu",
        "encoder",
        "sync_pulse",
    }
    assert all(item.status == "READY" for item in report.devices.values())
    assert all(item.observed_raw_data for item in report.devices.values())
    assert report.devices["ultrasound"].channel_count == 4
    assert report.devices["sync_pulse"].observed_sync_rising_edge is True
    assert not list(tmp_path.glob(".exo-write-probe-*.tmp"))
    assert not list(tmp_path.rglob("*.recording"))
    assert not (tmp_path / "catalog.sqlite3").exists()


def test_preflight_refuses_insufficient_free_space(tmp_path: Path) -> None:
    report = run_simulated_preflight(
        tmp_path,
        minimum_free_space_gib=10**9,
        timeout_s=1.0,
    )
    assert not report.ready
    assert report.writable
    assert all(item.status == "FAILED" for item in report.devices.values())
    assert all("free space" in item.message for item in report.devices.values())


def test_preflight_can_enforce_a_configured_write_throughput_threshold(
    tmp_path: Path,
) -> None:
    report = run_simulated_preflight(
        tmp_path,
        minimum_free_space_gib=0,
        minimum_write_mib_s=10**12,
        write_probe_mib=0.01,
        timeout_s=1.0,
    )

    assert not report.ready
    assert all(item.status == "FAILED" for item in report.devices.values())
    assert all("write probe" in item.message for item in report.devices.values())


def test_preflight_turns_adapter_connect_failure_into_device_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_type = preflight_module._ADAPTERS["imu"]

    class BrokenImu(original_type):
        def connect(self, config=None) -> None:  # type: ignore[no-untyped-def]
            raise OSError("simulated connect failure")

    monkeypatch.setitem(preflight_module._ADAPTERS, "imu", BrokenImu)
    report = run_simulated_preflight(
        tmp_path,
        minimum_free_space_gib=0,
        timeout_s=0.3,
    )
    assert not report.ready
    assert report.devices["imu"].status == "FAILED"
    assert "connect failure" in report.devices["imu"].message


def test_preflight_worker_uses_spawn_and_returns_report(tmp_path: Path) -> None:
    worker = CollectorPreflightWorker(
        tmp_path,
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )
    worker.start()
    try:
        response: tuple[str, object] | None = None
        deadline = time.monotonic() + 20.0
        while response is None and time.monotonic() < deadline:
            response = worker.poll_result()
            time.sleep(0.02)
        assert response is not None
        status, payload = response
        assert status == "completed"
        assert isinstance(payload, CollectorPreflightReport)
        assert payload.data_root == tmp_path.resolve()
        assert payload.ready
        worker.join(5.0)
        assert worker.exitcode == 0
        assert not worker.is_alive
    finally:
        if worker.is_alive:
            worker.terminate(timeout=1.0)
        worker.join(0)
        worker.close()
