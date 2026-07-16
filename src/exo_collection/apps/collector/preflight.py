"""Real lifecycle and storage preflight for the built-in simulators."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, replace
import multiprocessing as mp
from multiprocessing.queues import Queue
import os
from pathlib import Path
from queue import Empty
import shutil
import time
import traceback
from typing import Any
from uuid import uuid4

from exo_collection.adapters.base import QueuedSimulatedAdapter, StartToken, TrialContext
from exo_collection.adapters.encoder.simulated import SimulatedEncoderAdapter
from exo_collection.adapters.imu.simulated import SimulatedImuAdapter
from exo_collection.adapters.sync_pulse.simulated import SimulatedSyncPulseAdapter
from exo_collection.adapters.ultrasound.simulated import SimulatedUltrasoundAdapter
from exo_collection.configuration.device_profiles import load_simulated_device_profile
from exo_collection.domain.events import EdgeType, FrameBatch, SampleBatch, SyncPulseEvent


_ADAPTERS: dict[str, type[QueuedSimulatedAdapter[Any]]] = {
    "ultrasound": SimulatedUltrasoundAdapter,
    "imu": SimulatedImuAdapter,
    "encoder": SimulatedEncoderAdapter,
    "sync_pulse": SimulatedSyncPulseAdapter,
}


@dataclass(frozen=True, slots=True)
class DevicePreflightResult:
    modality: str
    status: str
    device_id: str
    nominal_rate_hz: float
    actual_rate_hz: float | None
    channel_count: int
    queue_capacity: int
    observed_raw_data: bool
    observed_sync_rising_edge: bool | None
    message: str


@dataclass(frozen=True, slots=True)
class CollectorPreflightReport:
    devices: dict[str, DevicePreflightResult]
    data_root: Path
    writable: bool
    disk_free_bytes: int
    minimum_free_bytes: int
    write_probe_bytes: int
    write_probe_elapsed_s: float
    measured_write_mib_s: float
    minimum_write_mib_s: float | None
    elapsed_s: float

    @property
    def ready(self) -> bool:
        return (
            self.writable
            and self.disk_free_bytes >= self.minimum_free_bytes
            and (
                self.minimum_write_mib_s is None
                or self.measured_write_mib_s >= self.minimum_write_mib_s
            )
            and bool(self.devices)
            and all(item.status == "READY" for item in self.devices.values())
        )


def _simulated_preflight_process_entry(
    data_root: str,
    keyword_arguments: dict[str, Any],
    result_queue: Queue[Any],
) -> None:
    """Spawn entry point for device/storage checks.

    Device discovery, vendor SDK calls and the fsync write probe must never run
    on Qt's GUI thread.  Keeping the entry point in this module also gives a
    future real-device implementation a stable process boundary to replace.
    """

    try:
        result = run_simulated_preflight(data_root, **keyword_arguments)
        result_queue.put(("completed", result))
    except BaseException:
        result_queue.put(("failed", traceback.format_exc(limit=30)))


class CollectorPreflightWorker:
    """Own one spawned preflight process and a single-result queue."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        minimum_free_space_gib: float = 2.0,
        minimum_write_mib_s: float | None = None,
        write_probe_mib: float = 1.0,
        timeout_s: float = 1.25,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self._context = context or mp.get_context("spawn")
        self._result_queue = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_simulated_preflight_process_entry,
            args=(
                str(self.data_root),
                {
                    "minimum_free_space_gib": minimum_free_space_gib,
                    "minimum_write_mib_s": minimum_write_mib_s,
                    "write_probe_mib": write_probe_mib,
                    "timeout_s": timeout_s,
                },
                self._result_queue,
            ),
            name="collector-device-preflight",
            daemon=False,
        )
        self._closed = False
        self._exitcode: int | None = None

    @property
    def is_alive(self) -> bool:
        return False if self._closed else self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._exitcode if self._closed else self._process.exitcode

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("preflight worker is closed")
        if self._process.pid is not None:
            raise RuntimeError("preflight worker can only be started once")
        try:
            self._process.start()
        except BaseException:
            self._cleanup_after_start_failure()
            raise

    def poll_result(self) -> tuple[str, object] | None:
        if self._closed:
            return None
        try:
            status, payload = self._result_queue.get_nowait()
        except Empty:
            return None
        return str(status), payload

    def join(self, timeout: float | None = None) -> int | None:
        if self._closed:
            return self._exitcode
        self._process.join(timeout)
        self._exitcode = self._process.exitcode
        return self._exitcode

    def terminate(self, timeout: float = 1.0) -> int | None:
        """Bound shutdown so a hung SDK probe cannot keep the app alive."""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._closed:
            return self._exitcode
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
        self._exitcode = self._process.exitcode
        return self._exitcode

    def close(self) -> None:
        if self._closed:
            return
        if self._process.is_alive():
            raise RuntimeError("cannot close a running preflight worker")
        self._exitcode = self._process.exitcode
        self._result_queue.close()
        self._result_queue.join_thread()
        self._process.close()
        self._closed = True

    def _cleanup_after_start_failure(self) -> None:
        if self._process.pid is not None:
            with suppress(BaseException):
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=1.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)
        with suppress(BaseException):
            self._exitcode = self._process.exitcode
        with suppress(BaseException):
            self._result_queue.close()
        with suppress(BaseException):
            self._result_queue.join_thread()
        with suppress(BaseException):
            self._process.close()
        self._closed = True


def _probe_storage(
    data_root: Path,
    minimum_free_bytes: int,
    write_probe_bytes: int,
) -> tuple[bool, int, float, float]:
    data_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(data_root).free)
    probe = data_root / f".exo-write-probe-{uuid4().hex}.tmp"
    writable = False
    elapsed_s = 0.0
    try:
        payload = bytes(min(write_probe_bytes, 1024 * 1024))
        remaining = write_probe_bytes
        started = time.perf_counter()
        with probe.open("xb") as stream:
            while remaining:
                chunk = payload[: min(remaining, len(payload))]
                stream.write(chunk)
                remaining -= len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        elapsed_s = max(time.perf_counter() - started, 1e-9)
        writable = True
    finally:
        probe.unlink(missing_ok=True)
    measured_mib_s = (write_probe_bytes / 1024**2) / elapsed_s
    return writable, free, elapsed_s, measured_mib_s


def run_simulated_preflight(
    data_root: str | Path,
    *,
    minimum_free_space_gib: float = 2.0,
    minimum_write_mib_s: float | None = None,
    write_probe_mib: float = 1.0,
    timeout_s: float = 1.25,
) -> CollectorPreflightReport:
    """Connect, prepare and briefly sample every required simulator.

    The probe deliberately creates no Project, Catalog, Session or Trial
    package.  It exercises the same Adapter lifecycle and validates raw data
    and the analog sync rising edge before the UI enables collection.
    """

    if minimum_free_space_gib < 0:
        raise ValueError("minimum_free_space_gib must be non-negative")
    if minimum_write_mib_s is not None and minimum_write_mib_s <= 0:
        raise ValueError("minimum_write_mib_s must be positive when configured")
    if write_probe_mib <= 0:
        raise ValueError("write_probe_mib must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    started = time.perf_counter()
    root = Path(data_root).expanduser().resolve()
    minimum_free_bytes = int(minimum_free_space_gib * 1024**3)
    write_probe_bytes = max(1, int(write_probe_mib * 1024**2))
    (
        storage_ready,
        free_bytes,
        write_probe_elapsed_s,
        measured_write_mib_s,
    ) = _probe_storage(root, minimum_free_bytes, write_probe_bytes)
    profile = load_simulated_device_profile()
    adapters: dict[str, QueuedSimulatedAdapter[Any]] = {}
    prepared: dict[str, Any] = {}
    observed_raw = {modality: False for modality in _ADAPTERS}
    observed_sync_rising = False
    failures: dict[str, str] = {}

    context = TrialContext(
        trial_uuid=uuid4(),
        session_uuid=uuid4(),
        condition={"purpose": "preflight_only"},
        recording_dir=None,
    )
    try:
        for device in profile.devices:
            adapter_type = _ADAPTERS[device.modality]
            adapter = adapter_type(device.adapter_configuration())
            adapters[device.modality] = adapter
            try:
                descriptor = adapter.descriptor()
                if descriptor.modality != device.modality:
                    raise ValueError("descriptor modality differs from profile")
                if descriptor.device_id != device.device_id:
                    raise ValueError("descriptor device_id differs from profile")
                if descriptor.clock_domain != device.clock_domain:
                    raise ValueError("descriptor clock domain differs from profile")
                if not descriptor.channels or len(descriptor.channels) != len(descriptor.units):
                    raise ValueError("channel/unit declaration is invalid")
                adapter.connect()
                info = adapter.prepare(context)
                if not adapter.health().ready:
                    raise RuntimeError("adapter did not report READY after prepare")
                prepared[device.modality] = info
            except Exception as exc:
                failures[device.modality] = f"{type(exc).__name__}: {exc}"

        if not failures and len(prepared) == len(profile.devices):
            token = StartToken()
            for adapter in adapters.values():
                adapter.start(token)
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                for modality, adapter in adapters.items():
                    for _ in range(64):
                        event = adapter.get_event(timeout=0)
                        if event is None:
                            break
                        if isinstance(event, (FrameBatch, SampleBatch)):
                            observed_raw[modality] = True
                        elif (
                            modality == "sync_pulse"
                            and isinstance(event, SyncPulseEvent)
                            and event.edge_type is EdgeType.RISING
                        ):
                            observed_sync_rising = True
                if all(observed_raw.values()) and observed_sync_rising:
                    break
                time.sleep(0.002)
            for modality, adapter in adapters.items():
                if not observed_raw[modality]:
                    failures[modality] = "no raw batch observed during preflight"
                try:
                    adapter.raise_if_faulted()
                except Exception as exc:
                    failures[modality] = f"{type(exc).__name__}: {exc}"
            if not observed_sync_rising:
                failures["sync_pulse"] = "no qualified analog sync rising edge observed"
    finally:
        for adapter in adapters.values():
            try:
                adapter.close()
            except Exception as exc:
                modality = adapter.descriptor().modality
                failures.setdefault(modality, f"close failed: {type(exc).__name__}: {exc}")

    results: dict[str, DevicePreflightResult] = {}
    for device in profile.devices:
        adapter = adapters.get(device.modality)
        descriptor = adapter.descriptor() if adapter is not None else None
        info = prepared.get(device.modality)
        message = failures.get(device.modality, "Adapter lifecycle and sample probe passed")
        # health() remains an auditable snapshot after close; use the observed
        # rate captured immediately from its counters rather than claiming a
        # calibrated hardware rate.
        health = adapter.health() if adapter is not None else None
        results[device.modality] = DevicePreflightResult(
            modality=device.modality,
            status="FAILED" if device.modality in failures else "READY",
            device_id=device.device_id,
            nominal_rate_hz=(descriptor.nominal_rate_hz if descriptor else 0.0),
            actual_rate_hz=(
                float(health.actual_sample_rate_hz)
                if health is not None and health.actual_sample_rate_hz is not None
                else None
            ),
            channel_count=len(descriptor.channels) if descriptor else 0,
            queue_capacity=int(info.queue_capacity) if info is not None else 0,
            observed_raw_data=observed_raw.get(device.modality, False),
            observed_sync_rising_edge=(
                observed_sync_rising if device.modality == "sync_pulse" else None
            ),
            message=message,
        )

    throughput_ready = (
        minimum_write_mib_s is None
        or measured_write_mib_s >= minimum_write_mib_s
    )
    if not storage_ready or free_bytes < minimum_free_bytes or not throughput_ready:
        if not storage_ready:
            storage_message = "data root is not writable"
        elif free_bytes < minimum_free_bytes:
            storage_message = (
                f"free space {free_bytes} B is below minimum {minimum_free_bytes} B"
            )
        else:
            assert minimum_write_mib_s is not None
            storage_message = (
                f"write probe {measured_write_mib_s:.2f} MiB/s is below "
                f"configured minimum {minimum_write_mib_s:.2f} MiB/s"
            )
        results = {
            modality: replace(
                item,
                status="FAILED",
                message=f"{item.message}; {storage_message}",
            )
            for modality, item in results.items()
        }

    return CollectorPreflightReport(
        devices=results,
        data_root=root,
        writable=storage_ready,
        disk_free_bytes=free_bytes,
        minimum_free_bytes=minimum_free_bytes,
        write_probe_bytes=write_probe_bytes,
        write_probe_elapsed_s=write_probe_elapsed_s,
        measured_write_mib_s=measured_write_mib_s,
        minimum_write_mib_s=minimum_write_mib_s,
        elapsed_s=max(0.0, time.perf_counter() - started),
    )


__all__ = [
    "CollectorPreflightWorker",
    "CollectorPreflightReport",
    "DevicePreflightResult",
    "run_simulated_preflight",
]
