from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import time
from typing import Any

import numpy as np
import pytest

from exo_collection.adapters.base import (
    AdapterError,
    ModalityDescriptor,
    PreparedInfo,
    StartToken,
    StopReport,
    TrialContext,
)
from exo_collection.apps.collector import preflight as preflight_module
from exo_collection.apps.collector.preflight import (
    CollectorPreflightReport,
    CollectorPreflightWorker,
    run_device_preflight,
    run_simulated_preflight,
)
from exo_collection.configuration import adapter_registry as reg_module
from exo_collection.configuration.device_profiles import (
    ELONXI_ULTRASOUND_ADAPTER,
    SYNC_PULSE_ADAPTER,
    TEENSY_ENCODER_ADAPTER,
    XSENS_AWINDA_ADAPTER,
    HardwareDeviceProfileDocument,
)
from exo_collection.domain.events import (
    EdgeType,
    FrameBatch,
    HealthSnapshot,
    HealthStatus,
    SampleBatch,
    SyncPulseEvent,
)


# ---------------------------------------------------------------------------
#  Helper: generic fake adapter that queues one event on start
# ---------------------------------------------------------------------------

def _fake_event_for_modality(
    *,
    modality: str,
    device_id: str,
    clock_domain: str,
) -> FrameBatch | SampleBatch | SyncPulseEvent:
    """Return one plausible raw event so the preflight observer loop succeeds."""
    if modality == "ultrasound":
        return FrameBatch(
            event_type="frame_batch",
            device_id=device_id,
            modality=modality,
            clock_domain=clock_domain,
            first_frame_index=0,
            frame_count=10,
            sequence_number=0,
            frame_rate_hz=20.0,
            data=np.zeros((10, 1000, 4), dtype=np.float32),
        )
    if modality == "sync_pulse":
        return SyncPulseEvent(
            event_type="sync_pulse",
            device_id=device_id,
            modality=modality,
            clock_domain=clock_domain,
            pulse_id="pulse_0",
            source_device=device_id,
            edge_type=EdgeType.RISING,
            sample_index=0,
            amplitude=3.3,
            detection_threshold=2.5,
            confidence=1.0,
            detector_version="1.0.0",
        )
    return SampleBatch(
        event_type="sample_batch",
        device_id=device_id,
        modality=modality,
        clock_domain=clock_domain,
        first_sample_index=0,
        sample_count=10,
        sequence_number=0,
        sample_rate_hz=100.0,
        data=np.zeros((10, 3), dtype=np.float32),
    )


class FakeQueuedAdapter:
    """Minimal adapter that satisfies the preflight lifecycle.

    On ``start()`` it pushes one event into an internal queue so the
    raw-data observer sees at least one batch.

    Subclasses set the class attribute ``_modality`` (and optionally
    ``_channels``, ``_units``, ``_nominal_rate_hz``) so that
    ``build_adapters`` (which does not pass modality as a config key)
    can still produce a correct descriptor.
    """

    _modality: str = "unknown"
    _channels: tuple[str, ...] = ("ch0",)
    _units: tuple[str, ...] = ("raw",)
    _nominal_rate_hz: float = 100.0

    def __init__(self, config: dict[str, Any]) -> None:
        self._device_id: str = config.get("device_id", "fake_device")
        self._clock_domain: str = config.get("clock_domain", "fake_clock")
        self._queue_capacity: int = int(config.get("queue_capacity", 64))
        self._queue: Queue[Any] = Queue(maxsize=self._queue_capacity)
        self._started = False
        self._fault_message: str | None = None
        self._connect_error: Exception | None = None

    def set_connect_error(self, exc: Exception) -> None:
        self._connect_error = exc

    def descriptor(self) -> ModalityDescriptor:
        return ModalityDescriptor(
            device_id=self._device_id,
            modality=type(self)._modality,
            display_name=f"Fake {type(self)._modality}",
            clock_domain=self._clock_domain,
            event_kind=(
                "frame_batch"
                if type(self)._modality == "ultrasound"
                else "sample_batch"
            ),
            channels=type(self)._channels,
            units=type(self)._units,
            nominal_rate_hz=type(self)._nominal_rate_hz,
            sample_shape=(10,),
            dtype="float32",
        )

    def configuration_snapshot(self) -> dict[str, Any]:
        return {}

    def connect(self, config: Any = None) -> None:
        if self._connect_error is not None:
            raise self._connect_error

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        return PreparedInfo(
            device_id=self._device_id,
            modality=type(self)._modality,
            trial_uuid=str(trial.trial_uuid),
            clock_domain=self._clock_domain,
            nominal_rate_hz=type(self)._nominal_rate_hz,
            channels=type(self)._channels,
            units=type(self)._units,
            queue_capacity=self._queue_capacity,
        )

    def start(self, start_token: StartToken | None = None) -> None:
        self._started = True
        if type(self)._modality == "sync_pulse":
            self._queue.put(
                SampleBatch(
                    event_type="sample_batch",
                    device_id=self._device_id,
                    modality="sync_pulse",
                    clock_domain=self._clock_domain,
                    first_sample_index=0,
                    sample_count=10,
                    sequence_number=0,
                    sample_rate_hz=type(self)._nominal_rate_hz,
                    data=np.zeros((10, 1), dtype=np.float32),
                )
            )
        event = _fake_event_for_modality(
            modality=type(self)._modality,
            device_id=self._device_id,
            clock_domain=self._clock_domain,
        )
        self._queue.put(event)

    def stop(self) -> StopReport:
        self._started = False
        return StopReport(
            device_id=self._device_id,
            modality=type(self)._modality,
            batches_emitted=1,
            samples_emitted=10,
            injected_dropped_batches=0,
            raw_queue_overflows=0,
            first_data_monotonic_ns=None,
            last_data_monotonic_ns=None,
            fault=None,
        )

    def close(self) -> None:
        pass

    def health(self) -> HealthSnapshot:
        return HealthSnapshot(
            device_id=self._device_id,
            modality=type(self)._modality,
            status=HealthStatus.HEALTHY,
            ready=True,
            connected=True,
            actual_sample_rate_hz=type(self)._nominal_rate_hz,
            nominal_sample_rate_hz=type(self)._nominal_rate_hz,
            queue_capacity=self._queue_capacity,
        )

    def get_event(self, timeout: float | None = None) -> Any | None:
        try:
            return self._queue.get(timeout=timeout or 0)
        except Empty:
            return None

    poll_event = get_event

    def raise_if_faulted(self) -> None:
        if self._fault_message is not None:
            raise AdapterError(self._fault_message)


class FakeUltrasoundAdapter(FakeQueuedAdapter):
    _modality = "ultrasound"
    _channels = ("ch1", "ch2", "ch3", "ch4")
    _units = ("raw", "raw", "raw", "raw")
    _nominal_rate_hz = 20.0


class FakeImuAdapter(FakeQueuedAdapter):
    _modality = "imu"
    _channels = ("acc_x", "acc_y", "acc_z")
    _units = ("m/s^2", "m/s^2", "m/s^2")
    _nominal_rate_hz = 200.0


class FakeEncoderAdapter(FakeQueuedAdapter):
    _modality = "encoder"
    _channels = ("pos",)
    _units = ("count",)
    _nominal_rate_hz = 100.0


class FakeSyncPulseAdapter(FakeQueuedAdapter):
    _modality = "sync_pulse"
    _channels = ("sync",)
    _units = ("V",)
    _nominal_rate_hz = 1000.0


def _load_hardware_profile() -> HardwareDeviceProfileDocument:
    """Return a validated in-memory hardware profile for testing."""
    return HardwareDeviceProfileDocument.model_validate(
        {
            "profile_kind": "hardware",
            "display_name": "test hardware",
            "schema_version": "1.0.0",
            "laboratory_sync_ready": False,
            "devices": [
                {
                    "id": "ultrasound_hw",
                    "modality": "ultrasound",
                    "adapter": ELONXI_ULTRASOUND_ADAPTER,
                    "writer": "block_binary",
                    "required": True,
                    "clock_domain": "hw_ultrasound",
                    "simulated": False,
                    "parameters": {
                        "queue_capacity": 64,
                        "nominal_rate_hz": 20.0,
                        "samples_per_channel": 1000,
                    },
                },
                {
                    "id": "imu_hw",
                    "modality": "imu",
                    "adapter": XSENS_AWINDA_ADAPTER,
                    "writer": "hdf5_signal",
                    "required": True,
                    "clock_domain": "hw_imu",
                    "simulated": False,
                    "parameters": {
                        "queue_capacity": 64,
                        "sample_rate_hz": 200.0,
                    },
                },
                {
                    "id": "encoder_hw",
                    "modality": "encoder",
                    "adapter": TEENSY_ENCODER_ADAPTER,
                    "writer": "hdf5_signal",
                    "required": True,
                    "clock_domain": "hw_encoder",
                    "simulated": False,
                    "parameters": {
                        "queue_capacity": 64,
                        "nominal_rate_hz": 100.0,
                        "batch_size": 20,
                    },
                },
                {
                    "id": "sync_pulse_hw",
                    "modality": "sync_pulse",
                    "adapter": SYNC_PULSE_ADAPTER,
                    "writer": "hdf5_signal",
                    "required": True,
                    "clock_domain": "hw_sync",
                    "simulated": True,
                    "parameters": {
                        "queue_capacity": 64,
                        "sample_rate_hz": 1000,
                        "pulse_interval_s": 1.0,
                        "pulse_width_s": 0.02,
                    },
                },
            ],
        }
    )


def _make_fake_adapters_for_registry() -> dict[str, type[FakeQueuedAdapter]]:
    """Return per-modality FakeQueuedAdapter subclasses keyed by registry slug."""
    return {
        ELONXI_ULTRASOUND_ADAPTER: FakeUltrasoundAdapter,
        XSENS_AWINDA_ADAPTER: FakeImuAdapter,
        TEENSY_ENCODER_ADAPTER: FakeEncoderAdapter,
        SYNC_PULSE_ADAPTER: FakeSyncPulseAdapter,
    }


# ---------------------------------------------------------------------------
#  Existing simulated-only tests (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
#  New: run_device_preflight with simulated profile
# ---------------------------------------------------------------------------


def test_run_device_preflight_simulated_passes(
    tmp_path: Path,
) -> None:
    """run_device_preflight with simulated profile exercises full lifecycle."""
    report = run_device_preflight(
        tmp_path,
        device_profile_key="simulated",
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )

    assert report.ready
    assert report.profile_kind == "simulated"
    assert report.profile_key == "simulated"
    assert set(report.devices) == {"ultrasound", "imu", "encoder", "sync_pulse"}
    assert all(item.status == "READY" for item in report.devices.values())
    assert all(item.observed_raw_data for item in report.devices.values())


# ---------------------------------------------------------------------------
#  New: hardware profile with fake adapters via registry monkeypatch
# ---------------------------------------------------------------------------


def test_run_device_preflight_hardware_with_fake_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hardware profile preflight succeeds when adapters behave correctly."""
    fake_types = _make_fake_adapters_for_registry()
    for slug, fake_cls in fake_types.items():
        monkeypatch.setitem(reg_module.ADAPTER_REGISTRY, slug, fake_cls)
    monkeypatch.setattr(
        preflight_module,
        "load_device_profile",
        lambda key: _load_hardware_profile(),
    )

    report = run_device_preflight(
        tmp_path,
        device_profile_key="hardware",
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )

    assert report.ready
    assert report.profile_kind == "hardware"
    assert report.profile_key == "hardware"
    modalities = set(report.devices)
    assert modalities == {"ultrasound", "imu", "encoder", "sync_pulse"}
    for modality in modalities:
        item = report.devices[modality]
        assert item.status == "READY", f"{modality}: {item.message}"
        assert item.observed_raw_data
    assert report.devices["sync_pulse"].observed_sync_rising_edge is True


class _BrokenUltrasoundAdapter(FakeUltrasoundAdapter):
    def connect(self, config: Any = None) -> None:
        raise ImportError(
            "未安装 pythonnet — 请先安装硬件依赖。"
        )


class _BrokenImuAdapter(FakeImuAdapter):
    def connect(self, config: Any = None) -> None:
        raise ImportError(
            "未安装 Xsens MT SDK 的 Python xsensdeviceapi wheel"
        )


class _BrokenEncoderAdapter(FakeEncoderAdapter):
    def connect(self, config: Any = None) -> None:
        raise ImportError(
            "未安装 pyserial — 请先安装硬件依赖。"
        )


def test_hardware_preflight_missing_sdk_gives_clear_error_no_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every hardware adapter fails connect the report FAILs each
    modality clearly — no silent fallback to simulation."""

    fake_types = {
        ELONXI_ULTRASOUND_ADAPTER: _BrokenUltrasoundAdapter,
        XSENS_AWINDA_ADAPTER: _BrokenImuAdapter,
        TEENSY_ENCODER_ADAPTER: _BrokenEncoderAdapter,
        SYNC_PULSE_ADAPTER: FakeSyncPulseAdapter,  # sync is still simulated
    }
    for slug, fake_cls in fake_types.items():
        monkeypatch.setitem(reg_module.ADAPTER_REGISTRY, slug, fake_cls)
    monkeypatch.setattr(
        preflight_module,
        "load_device_profile",
        lambda key: _load_hardware_profile(),
    )

    report = run_device_preflight(
        tmp_path,
        device_profile_key="hardware",
        minimum_free_space_gib=0,
        timeout_s=0.5,
    )

    assert not report.ready
    assert report.profile_kind == "hardware"

    # Hardware modalities must fail with the ImportError message
    for mod in ("ultrasound", "imu", "encoder"):
        item = report.devices[mod]
        assert item.status == "FAILED", f"{mod}: {item.message}"
        assert "ImportError" in item.message

    # sync_pulse uses FakeSyncPulseAdapter (not broken), so it may pass
    # depending on timing.  The critical assertion is that no modality
    # silently degraded to READY on a hardware adapter that raised.


def test_hardware_preflight_report_has_correct_profile_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report carries profile_kind and profile_key for audit."""
    fake_types = _make_fake_adapters_for_registry()
    for slug, fake_cls in fake_types.items():
        monkeypatch.setitem(reg_module.ADAPTER_REGISTRY, slug, fake_cls)
    monkeypatch.setattr(
        preflight_module,
        "load_device_profile",
        lambda key: _load_hardware_profile(),
    )

    report = run_device_preflight(
        tmp_path,
        device_profile_key="hardware",
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )

    assert report.profile_kind == "hardware"
    assert report.profile_key == "hardware"


def test_hardware_preflight_sync_pulse_annotated_as_simulated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In hardware profile, sync_pulse message notes it is still simulated."""
    fake_types = _make_fake_adapters_for_registry()
    for slug, fake_cls in fake_types.items():
        monkeypatch.setitem(reg_module.ADAPTER_REGISTRY, slug, fake_cls)
    monkeypatch.setattr(
        preflight_module,
        "load_device_profile",
        lambda key: _load_hardware_profile(),
    )

    report = run_device_preflight(
        tmp_path,
        device_profile_key="hardware",
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )

    sync_item = report.devices["sync_pulse"]
    assert sync_item.status == "READY"
    assert "模拟同步" in sync_item.message
    assert "台架验证" in sync_item.message


def test_collector_preflight_worker_accepts_profile_key(
    tmp_path: Path,
) -> None:
    """Worker stores device_profile_key and passes it to the process entry."""
    worker = CollectorPreflightWorker(
        tmp_path,
        device_profile_key="hardware",
        device_overrides={"ultrasound": {"queue_capacity": 128}},
        minimum_free_space_gib=0,
        timeout_s=1.0,
    )

    # The worker's process is not started; verify kwargs were captured.
    assert worker.data_root == tmp_path.expanduser().resolve()
    assert worker._process._target == preflight_module._device_preflight_process_entry
    args = worker._process._args
    # args[1] is the kwargs dict
    kwargs_sent = args[1]
    assert kwargs_sent["device_profile_key"] == "hardware"
    assert kwargs_sent["device_overrides"] == {"ultrasound": {"queue_capacity": 128}}
    assert kwargs_sent["minimum_free_space_gib"] == 0
    assert kwargs_sent["timeout_s"] == 1.0

    worker.close()


def test_worker_defaults_to_simulated_profile_key(tmp_path: Path) -> None:
    """Backward-compatible: omitting profile key defaults to 'simulated'."""
    worker = CollectorPreflightWorker(
        tmp_path,
        minimum_free_space_gib=0,
    )

    args = worker._process._args
    kwargs_sent = args[1]
    assert kwargs_sent["device_profile_key"] == "simulated"
    assert "device_overrides" not in kwargs_sent

    worker.close()
