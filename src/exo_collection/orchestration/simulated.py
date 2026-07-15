"""End-to-end simulated Trial acquisition and immutable package publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Protocol
from uuid import UUID
from uuid import uuid4

import h5py
import numpy as np

from exo_collection import __version__
from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.adapters.base import QueuedSimulatedAdapter, StartToken, TrialContext
from exo_collection.adapters.encoder.simulated import SimulatedEncoderAdapter
from exo_collection.adapters.imu.simulated import SimulatedImuAdapter
from exo_collection.adapters.sync_pulse.simulated import SimulatedSyncPulseAdapter
from exo_collection.adapters.ultrasound.simulated import SimulatedUltrasoundAdapter
from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository
from exo_collection.configuration.device_profiles import (
    SimulatedDeviceProfileDocument,
    load_simulated_device_profile,
)
from exo_collection.domain.events import FrameBatch, HealthStatus, SampleBatch, SyncPulseEvent
from exo_collection.domain.models import (
    ArtifactKind,
    Condition,
    DeviceReference,
    Project,
    QualityGrade,
    Session,
    Subject,
    utc_now,
)
from exo_collection.domain.states import TrialState, TrialStateMachine
from exo_collection.readers.binary_block import scan_binary_file
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.layout import TrialLayout
from exo_collection.storage.manifest import (
    ClockAndAlignment,
    ClockDomainKind,
    ClockDomainManifest,
    ClockMapping,
    ConfigurationSnapshot,
    DeviceProvenance,
    ManifestArtifact,
    ModalityManifest,
    QualityIssue,
    QualityIssueSeverity,
    QualitySummary,
    ResidualStatistics,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
)
from exo_collection.storage.package import (
    ArtifactDraft,
    finalize_trial_package,
    publish_artifact,
    publish_json,
)
from exo_collection.timing.clock_model import fit_affine_clock
from exo_collection.writers.block_binary_process import BlockBinaryWriterProcess
from exo_collection.writers.hdf5_signal import Hdf5SignalWriter

from .models import TrialRunRequest, TrialRunResult


PublishCallback = Callable[[WorkerEvent], None]


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


class _NeverStop:
    def is_set(self) -> bool:
        return False


class _JsonlJournal:
    def __init__(self, layout: TrialLayout) -> None:
        self.layout = layout
        self.relative_path = "logs/trial.jsonl"
        self.path = layout.partial_path(self.relative_path)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._closed = False

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Trial journal is closed")
        record = {
            "timestamp_utc": utc_now().isoformat().replace("+00:00", "Z"),
            "host_monotonic_ns": time.perf_counter_ns(),
            "level": "ERROR" if "exception_type" in payload else "INFO",
            "process": "collector-core",
            "event_type": event_type,
            "session_uuid": str(self.layout.session_uuid),
            "trial_uuid": str(self.layout.trial_uuid),
            "device_id": payload.get("device_id"),
            "message": payload.get("message") or payload.get("reason"),
            "exception": (
                {
                    "type": payload.get("exception_type"),
                    "message": payload.get("message"),
                }
                if "exception_type" in payload
                else None
            ),
            **payload,
        }
        self._stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    @property
    def closed(self) -> bool:
        return self._closed

    def close_and_publish(self) -> Path:
        if not self._closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True
        return self.layout.publish_partial(self.relative_path)

    def close_incomplete(self) -> None:
        if not self._closed:
            self._stream.flush()
            self._stream.close()
            self._closed = True


def _publish(callback: PublishCallback | None, event: WorkerEvent) -> None:
    if callback is not None:
        try:
            callback(event)
        except Exception:
            # UI/control-plane telemetry is best-effort and must never change
            # the raw acquisition or package-publication result.
            pass


def _git_commit() -> str:
    explicit = os.environ.get("EXO_GIT_COMMIT", "").strip()
    if explicit:
        return explicit
    try:
        bundled = resources.files("exo_collection").joinpath("build-info.json")
        value = json.loads(bundled.read_text(encoding="utf-8")).get("git_commit", "").strip()
        if value:
            return value
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError, OSError, AttributeError):
        pass
    root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
            shell=False,
        )
        value = completed.stdout.strip()
        return value or "unknown-local-build"
    except (OSError, subprocess.SubprocessError):
        return "unknown-local-build"


def _write_session_file(layout: TrialLayout, session: Session) -> None:
    destination = layout.session_directory / "session.json"
    if destination.exists():
        existing = Session.model_validate_json(destination.read_text(encoding="utf-8"))
        if (
            existing.session_uuid != session.session_uuid
            or existing.project_uuid != session.project_uuid
            or existing.subject_uuid != session.subject_uuid
        ):
            raise ValueError(f"existing session.json has conflicting hierarchy UUIDs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"session.json.{uuid4().hex}.partial")
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(session.model_dump_json(indent=2))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


_SIMULATED_ADAPTER_TYPES: dict[str, type[QueuedSimulatedAdapter[Any]]] = {
    "ultrasound": SimulatedUltrasoundAdapter,
    "imu": SimulatedImuAdapter,
    "encoder": SimulatedEncoderAdapter,
    "sync_pulse": SimulatedSyncPulseAdapter,
}


def _make_adapters(
    request: TrialRunRequest,
    profile: SimulatedDeviceProfileDocument | None = None,
) -> dict[str, QueuedSimulatedAdapter[Any]]:
    """Build only statically registered simulators from a validated profile."""

    resolved_profile = profile or load_simulated_device_profile()
    devices = resolved_profile.by_modality()
    unknown_overrides = set(request.simulation) - set(devices)
    if unknown_overrides:
        display = ", ".join(sorted(unknown_overrides))
        raise ValueError(f"Unknown simulated modality override(s): {display}")

    adapters: dict[str, QueuedSimulatedAdapter[Any]] = {}
    for modality, device in devices.items():
        configuration = device.adapter_configuration()
        configuration.update(request.simulation.get(modality, {}))
        adapters[modality] = _SIMULATED_ADAPTER_TYPES[modality](configuration)
    return adapters


def _create_hdf5_writer(path: Path, adapter: QueuedSimulatedAdapter[Any]) -> Hdf5SignalWriter:
    descriptor = adapter.descriptor()
    return Hdf5SignalWriter(
        path,
        channels=descriptor.channels,
        units=descriptor.units,
        device_metadata={
            "device_id": descriptor.device_id,
            "clock_domain": descriptor.clock_domain,
            **dict(descriptor.metadata),
        },
        clock_model={"status": "unfitted_during_acquisition"},
        dtype=descriptor.dtype,
        sample_shape=descriptor.sample_shape,
        chunk_rows=max(32, int(descriptor.nominal_rate_hz)),
        nominal_rate_hz=descriptor.nominal_rate_hz,
        flush_every_batches=1,
    )


def _preview_event(event: FrameBatch | SampleBatch, trial_uuid: UUID) -> WorkerEvent:
    values = np.asarray(event.data)
    if isinstance(event, FrameBatch):
        flattened = values[-1].astype(np.float32, copy=False).reshape(-1)
        if flattened.size > 512:
            indices = np.linspace(0, flattened.size - 1, 512, dtype=np.int64)
            flattened = flattened[indices]
        payload: dict[str, Any] = {
            "host_monotonic_ns": event.host_monotonic_ns,
            "values": flattened.tolist(),
            "shape": [int(flattened.size)],
        }
    else:
        modality = event.modality
        if modality == "imu":
            signal = values[:, 0, 0]
            channel = "acc_x"
        elif modality == "encoder":
            signal = values[:, 0]
            channel = "left_position"
        else:
            signal = values[:, 0]
            channel = "voltage"
        rate = event.sample_rate_hz or 1.0
        x = (event.first_sample_index + np.arange(signal.size)) / rate
        payload = {
            "host_monotonic_ns": event.host_monotonic_ns,
            "x": x.astype(float).tolist(),
            "values": signal.astype(float).tolist(),
            "channel": channel,
        }
    return WorkerEvent(
        event_type=WorkerEventType.PREVIEW,
        trial_uuid=str(trial_uuid),
        modality=event.modality,
        payload=payload,
    )


def _clock_mappings(
    descriptors: dict[str, Any],
    anchors: dict[str, list[tuple[float, int]]],
) -> list[ClockMapping]:
    mappings: list[ClockMapping] = []
    for modality, values in anchors.items():
        if not values:
            continue
        # Retain a bounded, evenly spread set if a long Trial produced many batches.
        if len(values) > 2000:
            selected = np.linspace(0, len(values) - 1, 2000, dtype=np.int64)
            values = [values[int(index)] for index in selected]
        model = fit_affine_clock((item[0] for item in values), (item[1] for item in values))
        residuals = model.residuals
        mappings.append(
            ClockMapping(
                source_clock_domain=descriptors[modality].clock_domain,
                scale_a=model.scale_a,
                offset_b_ns=model.offset_b_ns,
                valid_source_start=model.source_start,
                valid_source_end=model.source_end,
                anchor_count=model.anchor_count,
                residuals=ResidualStatistics(
                    count=residuals.count,
                    mean_ns=residuals.mean_ns,
                    rms_ns=residuals.rms_ns,
                    standard_deviation_ns=residuals.standard_deviation_ns,
                    p95_absolute_ns=residuals.p95_absolute_ns,
                    max_absolute_ns=residuals.max_absolute_ns,
                ),
                algorithm_version=model.algorithm_version,
            )
        )
    return mappings


def _verify_hdf5(path: Path, expected_samples: int) -> None:
    with h5py.File(path, "r") as file:
        if not bool(file.attrs.get("closed_cleanly", False)):
            raise RuntimeError(f"HDF5 file is not marked clean: {path.name}")
        if int(file.attrs.get("sample_count", -1)) != expected_samples:
            raise RuntimeError(f"HDF5 sample count mismatch: {path.name}")
        if file["samples/data"].shape[0] != expected_samples:
            raise RuntimeError(f"HDF5 dataset length mismatch: {path.name}")


def run_simulated_trial(
    request: TrialRunRequest,
    *,
    stop_requested: StopSignal | None = None,
    publish: PublishCallback | None = None,
) -> TrialRunResult:
    """Collect four simulated modalities and atomically publish one Trial."""

    stop_signal = stop_requested or _NeverStop()
    root = request.data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    layout = TrialLayout.build(
        root,
        request.project_uuid,
        request.subject_uuid,
        request.session_uuid,
        request.trial_uuid,
    )
    machine = TrialStateMachine()
    device_profile = load_simulated_device_profile()
    profiles_by_modality = device_profile.by_modality()
    adapters = _make_adapters(request, device_profile)
    descriptors = {name: adapter.descriptor() for name, adapter in adapters.items()}
    condition = Condition(
        condition_code=request.condition_code,
        condition_name=request.condition_name,
        condition_level=request.condition_level,
        parameters=request.condition_parameters,
        repeat_index=request.repeat_index,
        protocol_version=request.protocol_version,
    )
    device_refs = [
        DeviceReference(
            device_id=descriptor.device_id,
            modality=descriptor.modality,
            required=profiles_by_modality[descriptor.modality].required,
            clock_domain=descriptor.clock_domain,
            metadata={
                "simulated": True,
                "profile_adapter": profiles_by_modality[descriptor.modality].adapter,
                "writer": profiles_by_modality[descriptor.modality].writer,
            },
        )
        for descriptor in descriptors.values()
    ]
    project = Project(
        project_uuid=request.project_uuid,
        project_name=request.project_name,
        principal_investigator=request.principal_investigator,
        protocol_version=request.protocol_version,
        data_root=str(root),
        condition_definition_version=request.protocol_version,
        default_device_config={"profile": "simulated"},
    )
    subject = Subject(
        subject_uuid=request.subject_uuid,
        project_uuid=request.project_uuid,
        subject_code=request.subject_code,
        group=request.subject_group,
        attributes={"simulated": True},
    )
    visit = Session(
        session_uuid=request.session_uuid,
        project_uuid=request.project_uuid,
        subject_uuid=request.subject_uuid,
        operator=request.operator,
        software_version=__version__,
        devices=device_refs,
    )

    catalog: Catalog | None = None
    repository: CatalogRepository | None = None
    journal: _JsonlJournal | None = None
    writers: dict[str, Any] = {}
    stop_reports: dict[str, Any] = {}
    counts = {name: 0 for name in adapters}
    pulse_event_count = 0
    anchors: dict[str, list[tuple[float, int]]] = {name: [] for name in adapters}
    last_preview_ns = {name: 0 for name in adapters}
    sample_bounds: dict[str, list[int | None]] = {
        name: [None, None] for name in adapters
    }

    def record_sample_bounds(modality: str, first_index: int, sample_count: int) -> None:
        last_index = first_index + sample_count - 1
        bounds = sample_bounds[modality]
        bounds[0] = first_index if bounds[0] is None else min(bounds[0], first_index)
        bounds[1] = last_index if bounds[1] is None else max(bounds[1], last_index)

    def transition(target: TrialState, reason: str) -> None:
        record = machine.transition(target, reason=reason)
        if journal is not None and not journal.closed:
            journal.write(
                "trial_state_transition",
                {
                    "trial_uuid": str(request.trial_uuid),
                    "from_state": record.from_state.value,
                    "to_state": record.to_state.value,
                    "reason": reason,
                },
            )
        _publish(
            publish,
            WorkerEvent(
                event_type=WorkerEventType.STATE,
                trial_uuid=str(request.trial_uuid),
                message=reason,
                payload={"state": target.value},
            ),
        )

    activity_lock = AcquisitionLock(
        root,
        request.trial_uuid,
        release_on_exception=False,
    )
    try:
        with activity_lock as activity:
            # The dataset-root lease is acquired before Catalog, Session or
            # Trial-directory side effects, so a losing Collector leaves no
            # orphan `.recording` package.
            catalog = Catalog(root / "catalog.sqlite3")
            catalog.migrate()
            repository = CatalogRepository(catalog)
            repository.register_hierarchy(project, subject, visit)
            _write_session_file(layout, visit)
            layout.create_recording()
            journal = _JsonlJournal(layout)
            transition(TrialState.PREPARING, "checking and preparing simulated devices")
            trial_context = TrialContext(
                trial_uuid=request.trial_uuid,
                session_uuid=request.session_uuid,
                condition=condition.model_dump(mode="json"),
                recording_dir=layout.recording_directory,
            )
            for adapter in adapters.values():
                adapter.connect()
                adapter.prepare(trial_context)

            writers["ultrasound"] = BlockBinaryWriterProcess(
                layout.partial_path("raw/ultrasound.bin"),
                dtype=descriptors["ultrasound"].dtype,
                sample_shape=descriptors["ultrasound"].sample_shape,
                metadata={
                    **dict(descriptors["ultrasound"].metadata),
                    "clock_domain": descriptors["ultrasound"].clock_domain,
                    "nominal_frame_rate_hz": descriptors["ultrasound"].nominal_rate_hz,
                    "unit": "a.u.",
                    "device": {"device_id": descriptors["ultrasound"].device_id, "simulated": True},
                },
            )
            for modality in ("imu", "encoder", "sync_pulse"):
                writers[modality] = _create_hdf5_writer(
                    layout.partial_path(f"raw/{modality}.h5"), adapters[modality]
                )

            transition(TrialState.READY, "all required simulated devices are ready")
            start_token = StartToken()
            started_at_utc = datetime.fromtimestamp(start_token.host_utc_ns / 1e9, timezone.utc)
            for adapter in adapters.values():
                adapter.start(start_token)
            transition(TrialState.RECORDING, "shared start token released")

            deadline_ns = start_token.host_monotonic_ns + int(request.duration_s * 1_000_000_000)
            # Emit an initial health snapshot immediately, then at 2 Hz.
            last_health_ns = start_token.host_monotonic_ns - 500_000_000
            last_heartbeat_ns = start_token.host_monotonic_ns
            while time.perf_counter_ns() < deadline_ns and not stop_signal.is_set():
                processed = False
                now_ns = time.perf_counter_ns()
                for modality, adapter in adapters.items():
                    for _ in range(32):
                        event = adapter.get_event(timeout=0)
                        if event is None:
                            break
                        processed = True
                        if isinstance(event, FrameBatch):
                            writers["ultrasound"].append(
                                event.data,
                                device_timestamp=int(event.device_timestamp or 0),
                                host_monotonic_ns=event.host_monotonic_ns,
                                host_utc_ns=event.host_utc_ns,
                                first_sample_index=event.first_frame_index,
                                sequence=event.sequence_number,
                            )
                            record_sample_bounds(
                                modality, event.first_frame_index, event.frame_count
                            )
                            counts[modality] += event.frame_count
                            if event.device_timestamp is not None:
                                anchors[modality].append(
                                    (float(event.device_timestamp), event.host_monotonic_ns)
                                )
                        elif isinstance(event, SampleBatch):
                            writers[modality].append_batch(event)
                            record_sample_bounds(
                                modality, event.first_sample_index, event.sample_count
                            )
                            counts[modality] += event.sample_count
                            if event.device_timestamp is not None:
                                anchors[modality].append(
                                    (float(event.device_timestamp), event.host_monotonic_ns)
                                )
                        elif isinstance(event, SyncPulseEvent):
                            writers["sync_pulse"].append_event(event)
                            pulse_event_count += 1
                            continue
                        else:
                            raise TypeError(f"Unsupported raw event: {type(event).__name__}")

                        if now_ns - last_preview_ns[modality] >= 66_000_000:
                            _publish(publish, _preview_event(event, request.trial_uuid))
                            last_preview_ns[modality] = now_ns

                if now_ns - last_health_ns >= 500_000_000:
                    for modality, adapter in adapters.items():
                        health = adapter.health()
                        _publish(
                            publish,
                            WorkerEvent(
                                event_type=WorkerEventType.HEALTH,
                                trial_uuid=str(request.trial_uuid),
                                modality=modality,
                                payload={
                                    "device_id": health.device_id,
                                    "status": health.status.value,
                                    "actual_sample_rate_hz": health.actual_sample_rate_hz,
                                    "queue_depth": health.queue_depth,
                                    "queue_capacity": health.queue_capacity,
                                    "sample_count": counts[modality],
                                },
                            ),
                        )
                        if health.status is HealthStatus.UNHEALTHY:
                            adapter.raise_if_faulted()
                    _publish(
                        publish,
                        WorkerEvent(
                            event_type=WorkerEventType.METRIC,
                            trial_uuid=str(request.trial_uuid),
                            payload={
                                "modality_counts": dict(counts),
                                "pulse_event_count": pulse_event_count,
                            },
                        ),
                    )
                    last_health_ns = now_ns
                if now_ns - last_heartbeat_ns >= 1_000_000_000:
                    activity.heartbeat()
                    last_heartbeat_ns = now_ns
                if not processed:
                    time.sleep(0.001)

            transition(TrialState.STOPPING, "controlled stop requested")
            for modality, adapter in adapters.items():
                stop_reports[modality] = adapter.stop()
            # A stop can race the final produced batch; drain every bounded raw queue.
            for modality, adapter in adapters.items():
                while (event := adapter.get_event(timeout=0)) is not None:
                    if isinstance(event, FrameBatch):
                        writers["ultrasound"].append(
                            event.data,
                            device_timestamp=int(event.device_timestamp or 0),
                            host_monotonic_ns=event.host_monotonic_ns,
                            host_utc_ns=event.host_utc_ns,
                            first_sample_index=event.first_frame_index,
                            sequence=event.sequence_number,
                        )
                        record_sample_bounds(
                            modality, event.first_frame_index, event.frame_count
                        )
                        counts[modality] += event.frame_count
                        if event.device_timestamp is not None:
                            anchors[modality].append((float(event.device_timestamp), event.host_monotonic_ns))
                    elif isinstance(event, SampleBatch):
                        writers[modality].append_batch(event)
                        record_sample_bounds(
                            modality, event.first_sample_index, event.sample_count
                        )
                        counts[modality] += event.sample_count
                        if event.device_timestamp is not None:
                            anchors[modality].append((float(event.device_timestamp), event.host_monotonic_ns))
                    elif isinstance(event, SyncPulseEvent):
                        writers["sync_pulse"].append_event(event)
                        pulse_event_count += 1

            transition(TrialState.FINALIZING, "all adapters acknowledged stop")
            stopped_reading_ns = time.perf_counter_ns()
            stopped_at_utc = datetime.now(timezone.utc)
            for writer in writers.values():
                writer.close()
            for adapter in adapters.values():
                adapter.close()

            # Verify source formats before any temporary name is published.
            ultrasound_scan = scan_binary_file(layout.partial_path("raw/ultrasound.bin"))
            if ultrasound_scan.error is not None or ultrasound_scan.complete_block_count == 0:
                raise RuntimeError(f"ultrasound integrity check failed: {ultrasound_scan.error}")
            ultrasound_writer = writers["ultrasound"]
            if (
                not isinstance(ultrasound_writer, BlockBinaryWriterProcess)
                or ultrasound_scan.complete_block_count != ultrasound_writer.written_count
                or sum(header.sample_count for header in ultrasound_scan.headers)
                != counts["ultrasound"]
            ):
                raise RuntimeError("ultrasound Writer count differs from the verified binary file")
            for modality in ("imu", "encoder", "sync_pulse"):
                _verify_hdf5(layout.partial_path(f"raw/{modality}.h5"), counts[modality])

            issues: list[QualityIssue] = []
            if pulse_event_count == 0:
                issues.append(
                    QualityIssue(
                        code="NO_SYNC_PULSE",
                        severity=QualityIssueSeverity.WARNING,
                        message="No complete simulated synchronization pulse was detected",
                        modality="sync_pulse",
                        metric="pulse_event_count",
                        observed_value=0,
                        threshold=1,
                    )
                )
            for modality, report in stop_reports.items():
                if report.injected_dropped_batches or report.raw_queue_overflows or report.fault:
                    issues.append(
                        QualityIssue(
                            code="DEVICE_DISCONTINUITY",
                            severity=QualityIssueSeverity.ERROR,
                            message=f"{modality} reported a dropped batch, overflow, or fault",
                            modality=modality,
                            observed_value=report.injected_dropped_batches + report.raw_queue_overflows,
                            threshold=0,
                        )
                    )
            grade = (
                QualityGrade.C
                if any(issue.severity is QualityIssueSeverity.ERROR for issue in issues)
                else QualityGrade.B
                if issues
                else QualityGrade.A
            )
            finalized_at_utc = datetime.now(timezone.utc)
            finalize_monotonic_ns = time.perf_counter_ns()

            statistics = {
                "schema_version": "1.0.0",
                "trial_uuid": str(request.trial_uuid),
                "duration_s": (stopped_reading_ns - start_token.host_monotonic_ns) / 1e9,
                "modality_counts": counts,
                "pulse_event_count": pulse_event_count,
                "ultrasound_block_count": ultrasound_scan.complete_block_count,
                "stop_reports": {name: asdict(report) for name, report in stop_reports.items()},
            }
            quality_report = {
                "schema_version": "1.0.0",
                "trial_uuid": str(request.trial_uuid),
                "computed_grade": grade.value,
                "required_artifacts_complete": True,
                "integrity_checks_passed": True,
                "issues": [issue.model_dump(mode="json") for issue in issues],
                "algorithm_version": "milestone-basic-quality-1.0.0",
            }
            configuration_document = {
                "schema_version": "1.0.0",
                "config_version": request.config_version,
                "protocol": condition.model_dump(mode="json"),
                "simulated_devices": {
                    modality: {
                        "profile": profiles_by_modality[modality].model_dump(
                            mode="json", by_alias=True
                        ),
                        "resolved_configuration": adapter.configuration_snapshot(),
                    }
                    for modality, adapter in adapters.items()
                },
            }
            configuration_bytes = json.dumps(
                configuration_document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            publish_json(layout, "derived/statistics.json", statistics)
            publish_json(
                layout,
                "derived/configuration_snapshot.json",
                configuration_document,
            )
            publish_json(layout, "reports/quality_report.json", quality_report)
            assert journal is not None
            journal.write(
                "trial_publication_intent",
                {
                    "trial_uuid": str(request.trial_uuid),
                    "from_state": TrialState.FINALIZING.value,
                    "target_state": TrialState.FINALIZED.value,
                    "reason": "artifacts verified; preparing atomic directory publication",
                },
            )
            activity.heartbeat()
            journal.close_and_publish()

            draft_by_key: dict[str, ArtifactDraft] = {
                "ultrasound": ArtifactDraft(
                    request.trial_uuid,
                    "ultrasound",
                    ArtifactKind.RAW,
                    "application/x-exo-ultrasound-blocks",
                    "raw/ultrasound.bin",
                    created_at_utc=started_at_utc,
                ),
                "ultrasound_meta": ArtifactDraft(
                    request.trial_uuid,
                    "ultrasound",
                    ArtifactKind.RAW,
                    "application/json",
                    "raw/ultrasound.meta.json",
                    created_at_utc=started_at_utc,
                ),
                "ultrasound_index": ArtifactDraft(
                    request.trial_uuid,
                    "ultrasound",
                    ArtifactKind.DERIVED,
                    "application/x-exo-ultrasound-index",
                    "raw/ultrasound.idx",
                    created_at_utc=started_at_utc,
                ),
                "imu": ArtifactDraft(
                    request.trial_uuid,
                    "imu",
                    ArtifactKind.RAW,
                    "application/x-hdf5",
                    "raw/imu.h5",
                    created_at_utc=started_at_utc,
                ),
                "encoder": ArtifactDraft(
                    request.trial_uuid,
                    "encoder",
                    ArtifactKind.RAW,
                    "application/x-hdf5",
                    "raw/encoder.h5",
                    created_at_utc=started_at_utc,
                ),
                "sync_pulse": ArtifactDraft(
                    request.trial_uuid,
                    "sync_pulse",
                    ArtifactKind.RAW,
                    "application/x-hdf5",
                    "raw/sync_pulse.h5",
                    created_at_utc=started_at_utc,
                    metadata={"contains_raw_waveform": True, "contains_detected_events": True},
                ),
                "statistics": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.DERIVED,
                    "application/json",
                    "derived/statistics.json",
                    created_at_utc=finalized_at_utc,
                ),
                "configuration": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.DERIVED,
                    "application/json",
                    "derived/configuration_snapshot.json",
                    created_at_utc=finalized_at_utc,
                ),
                "quality": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.REPORT,
                    "application/json",
                    "reports/quality_report.json",
                    created_at_utc=finalized_at_utc,
                ),
                "journal": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.LOG,
                    "application/x-ndjson",
                    "logs/trial.jsonl",
                    created_at_utc=started_at_utc,
                ),
            }
            artifacts: list[ManifestArtifact] = []
            for key, draft in draft_by_key.items():
                artifacts.append(publish_artifact(layout, draft, finalized_at_utc))
            artifact_map = {key: artifact for key, artifact in zip(draft_by_key, artifacts, strict=True)}

            mappings = _clock_mappings(descriptors, anchors)
            clock_domains = [
                ClockDomainManifest(
                    clock_domain=descriptor.clock_domain,
                    kind=ClockDomainKind.DEVICE_TICK,
                    unit="tick" if modality == "encoder" else "ns",
                    device_id=descriptor.device_id,
                    nominal_rate_hz=descriptor.nominal_rate_hz,
                    description="Simulated device clock; mapped to host monotonic time from batch anchors",
                )
                for modality, descriptor in descriptors.items()
            ]
            modalities: list[ModalityManifest] = []
            for modality, descriptor in descriptors.items():
                keys = (
                    ("ultrasound", "ultrasound_meta", "ultrasound_index")
                    if modality == "ultrasound"
                    else (modality,)
                )
                kwargs: dict[str, Any] = {
                    "modality": modality,
                    "required": profiles_by_modality[modality].required,
                    "adapter_type": f"{adapters[modality].__class__.__module__}.{adapters[modality].__class__.__name__}",
                    "writer_type": profiles_by_modality[modality].writer,
                    "clock_domain": descriptor.clock_domain,
                    "device_ids": [descriptor.device_id],
                    "artifact_uuids": [artifact_map[key].artifact_uuid for key in keys],
                    "channels": list(descriptor.channels),
                    "units": list(descriptor.units),
                    "first_sample_index": sample_bounds[modality][0],
                    "last_sample_index": sample_bounds[modality][1],
                    "sequence_gap_count": (
                        ultrasound_scan.sequence_gap_count
                        if modality == "ultrasound"
                        else stop_reports[modality].injected_dropped_batches
                    ),
                    "nominal_sample_rate_hz": descriptor.nominal_rate_hz,
                    "metadata": {"simulated": True, **dict(descriptor.metadata)},
                }
                if modality == "ultrasound":
                    kwargs["frame_count"] = counts[modality]
                    kwargs["metadata"]["source_sequence_gap_ranges"] = [
                        list(item) for item in ultrasound_scan.sequence_gap_ranges
                    ]
                else:
                    kwargs["sample_count"] = counts[modality]
                modalities.append(ModalityManifest(**kwargs))

            config_hash = hashlib.sha256(configuration_bytes).hexdigest()
            manifest = TrialManifest(
                project_uuid=request.project_uuid,
                subject_uuid=request.subject_uuid,
                session_uuid=request.session_uuid,
                trial_uuid=request.trial_uuid,
                state=TrialState.FINALIZED,
                condition=condition,
                timing=TrialTiming(
                    started_at_utc=started_at_utc,
                    stopped_at_utc=stopped_at_utc,
                    finalized_at_utc=finalized_at_utc,
                    start_host_monotonic_ns=start_token.host_monotonic_ns,
                    stop_host_monotonic_ns=stopped_reading_ns,
                    finalize_host_monotonic_ns=finalize_monotonic_ns,
                ),
                software=SoftwareProvenance(
                    application="Exo Collector",
                    application_version=__version__,
                    core_version=__version__,
                    git_commit=_git_commit(),
                    python_version=platform.python_version(),
                ),
                configuration=ConfigurationSnapshot(
                    config_version=request.config_version,
                    protocol_version=request.protocol_version,
                    condition_definition_version=request.protocol_version,
                    content_sha256=config_hash,
                    snapshot_relative_path="derived/configuration_snapshot.json",
                ),
                devices=[
                    DeviceProvenance(
                        device_id=descriptor.device_id,
                        modality=modality,
                        adapter_type=f"{adapters[modality].__class__.__module__}.{adapters[modality].__class__.__name__}",
                        manufacturer="simulator",
                        model="deterministic built-in simulator",
                        metadata={
                            "simulated": True,
                            "profile_required": profiles_by_modality[modality].required,
                            "profile_clock_domain": profiles_by_modality[modality].clock_domain,
                            "writer": profiles_by_modality[modality].writer,
                        },
                    )
                    for modality, descriptor in descriptors.items()
                ],
                modalities=modalities,
                artifacts=artifacts,
                clock_and_alignment=ClockAndAlignment(
                    clock_domains=clock_domains,
                    mappings=mappings,
                    raw_sync_pulse_artifact_uuids=[artifact_map["sync_pulse"].artifact_uuid],
                    sync_event_artifact_uuids=[artifact_map["sync_pulse"].artifact_uuid],
                ),
                quality=QualitySummary(
                    computed_grade=grade,
                    required_artifacts_complete=True,
                    integrity_checks_passed=True,
                    algorithm_version="milestone-basic-quality-1.0.0",
                    assessed_at_utc=finalized_at_utc,
                    issues=issues,
                    metric_count=len(counts) + 2,
                    report_artifact_uuid=artifact_map["quality"].artifact_uuid,
                ),
            )
            activity.heartbeat()
            final_directory = finalize_trial_package(layout, manifest)
            final_manifest_path = final_directory / "manifest.json"
            transition(TrialState.FINALIZED, "Trial directory atomically published")
            assert repository is not None
            try:
                repository.index_manifest(manifest, final_manifest_path)
            except Exception as catalog_error:
                # Catalog is a rebuildable index. A transient SQLite failure
                # cannot invalidate an already atomically published Trial.
                _publish(
                    publish,
                    WorkerEvent(
                        event_type=WorkerEventType.ALERT,
                        trial_uuid=str(request.trial_uuid),
                        message="Trial finalized; Catalog indexing deferred",
                        payload={
                            "catalog_index_deferred": True,
                            "exception_type": type(catalog_error).__name__,
                            "message": str(catalog_error),
                        },
                    ),
                )
            return TrialRunResult(
                trial_uuid=request.trial_uuid,
                state=TrialState.FINALIZED.value,
                trial_directory=final_directory,
                manifest_path=final_manifest_path,
                duration_s=(stopped_reading_ns - start_token.host_monotonic_ns) / 1e9,
                modality_counts=counts,
                pulse_event_count=pulse_event_count,
                quality_grade=grade.value,
            )
    except BaseException as exc:
        failure_target = {
            TrialState.PREPARING: TrialState.FAILED,
            TrialState.READY: TrialState.FAILED,
            TrialState.RECORDING: TrialState.ABORTED,
            TrialState.STOPPING: TrialState.RECOVERABLE,
            TrialState.FINALIZING: TrialState.RECOVERABLE,
        }.get(machine.state)
        failure_payload = {
            "trial_uuid": str(request.trial_uuid),
            "state": machine.state.value,
            "recovery_state": failure_target.value if failure_target is not None else None,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        if journal is not None and not journal.closed:
            try:
                journal.write("trial_failure", failure_payload)
            except BaseException:
                pass
        elif layout.recording_directory.is_dir():
            # The normal journal is intentionally immutable once published.
            # A finalization failure therefore gets a separate recovery report.
            try:
                publish_json(
                    layout,
                    f"reports/finalization-failure-{uuid4()}.json",
                    {
                        "schema_version": "1.0.0",
                        **failure_payload,
                    },
                )
            except BaseException:
                pass

        if failure_target is not None and machine.can_transition_to(failure_target):
            try:
                transition(
                    failure_target,
                    f"{type(exc).__name__} during {machine.state.value}",
                )
            except BaseException:
                pass
        for writer in writers.values():
            try:
                if isinstance(writer, BlockBinaryWriterProcess):
                    writer.abort()
                elif isinstance(writer, Hdf5SignalWriter):
                    writer.close(clean=False)
                else:
                    writer.close()
            except BaseException:
                pass
        for adapter in adapters.values():
            try:
                adapter.close()
            except BaseException:
                pass
        if journal is not None:
            try:
                journal.close_incomplete()
            except BaseException:
                pass
        raise
    finally:
        if catalog is not None:
            try:
                catalog.close()
            except BaseException:
                pass
        activity_lock.release()


__all__ = ["run_simulated_trial"]
