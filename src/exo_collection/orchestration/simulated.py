"""End-to-end simulated Trial acquisition and immutable package publication."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
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
from exo_collection.adapters.base import (
    AdapterError,
    QueuedSimulatedAdapter,
    StartToken,
    TrialContext,
)
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
from exo_collection.domain.events import (
    EdgeType,
    FrameBatch,
    HealthStatus,
    SampleBatch,
    SyncPulseEvent,
)
from exo_collection.domain.models import (
    ArtifactKind,
    Condition,
    DeviceReference,
    Project,
    Session,
    Subject,
    utc_now,
)
from exo_collection.domain.states import TrialState, TrialStateMachine
from exo_collection.quality import (
    ClockMappingEvidence,
    SignalEvidence,
    SyncEdgeEvidence,
    TrialQualityEvidence,
    UltrasoundEvidence,
    check_disk_space,
    evaluate_trial_quality,
    load_quality_rules,
    load_storage_policy,
    scan_hdf5_signal_evidence,
)
from exo_collection.readers.binary_block import scan_binary_file
from exo_collection.reporting.preview_png import (
    BoundedPreviewHistory,
    SIGNAL_PREVIEW_PATH,
    US_PREVIEW_PATH,
    publish_quality_preview_pngs,
)
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file
from exo_collection.storage.layout import TrialLayout
from exo_collection.storage.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ClockAndAlignment,
    ClockDomainKind,
    ClockDomainManifest,
    ClockMapping,
    ConfigurationSnapshot,
    DeviceProvenance,
    ManifestArtifact,
    ModalityManifest,
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
from exo_collection.writers.hdf5_signal import HDF5_SIGNAL_VERSION

from .models import TrialRunRequest, TrialRunResult


PublishCallback = Callable[[WorkerEvent], None]


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


class _NeverStop:
    def is_set(self) -> bool:
        return False


class MissingSyncTriggerError(RuntimeError):
    """A Trial that acquired pre-trigger raw data but never received its t0."""

    def __init__(
        self,
        reason: str,
        recording_directory: Path,
        *,
        primary_exception: BaseException | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.recording_directory = recording_directory
        self.primary_exception_type = (
            None if primary_exception is None else type(primary_exception).__name__
        )
        self.primary_exception_message = (
            None if primary_exception is None else str(primary_exception)
        )

    def worker_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "failure_code": "MISSING_SYNC_TRIGGER",
            "state": TrialState.RECOVERABLE.value,
            "recovery_state": TrialState.RECOVERABLE.value,
            "sync_status": "MISSING_TRIGGER",
            "recording_directory": str(self.recording_directory),
        }
        if self.primary_exception_type is not None:
            payload["primary_exception_type"] = self.primary_exception_type
            payload["primary_exception_message"] = self.primary_exception_message
        return payload


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
    def with_worktree_state(commit: str, dirty: bool | None) -> str:
        commit = commit.strip()
        if not commit:
            return "unknown-local-build"
        if dirty is True:
            return commit if commit.endswith("+dirty") else f"{commit}+dirty"
        if dirty is None and len(commit) == 40 and all(
            character in "0123456789abcdefABCDEF" for character in commit
        ):
            return f"{commit}+provenance-unavailable"
        return commit

    explicit = os.environ.get("EXO_GIT_COMMIT", "").strip()
    if explicit:
        return explicit
    try:
        bundled = resources.files("exo_collection").joinpath("build-info.json")
        build_info = json.loads(bundled.read_text(encoding="utf-8"))
        value = str(build_info.get("git_commit", "")).strip()
        dirty_value = build_info.get("git_worktree_dirty")
        dirty = dirty_value if isinstance(dirty_value, bool) else None
        if value:
            return with_worktree_state(value, dirty)
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
        if not value:
            return "unknown-local-build"
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                shell=False,
            )
            dirty: bool | None = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            dirty = None
        return with_worktree_state(value, dirty)
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


def _publish_csv(
    layout: TrialLayout,
    relative_path: str,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> Path:
    """Publish a deterministic UTF-8 CSV through the Trial partial-file path."""

    partial = layout.partial_path(relative_path)
    with partial.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    return layout.publish_partial(relative_path)


def _publish_text(layout: TrialLayout, relative_path: str, content: str) -> Path:
    """Publish a small immutable UTF-8 compatibility report atomically."""

    partial = layout.partial_path(relative_path)
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        if not content.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return layout.publish_partial(relative_path)


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


def _create_hdf5_writer(
    path: Path,
    adapter: QueuedSimulatedAdapter[Any],
    *,
    trial_metadata: Mapping[str, Any],
) -> Hdf5SignalWriter:
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
        trial_metadata=trial_metadata,
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
        source_frame = values[-1]
        frame = source_frame.astype(np.float32, copy=False)

        def downsample(signal: np.ndarray) -> np.ndarray:
            flattened = signal.reshape(-1)
            if flattened.size <= 512:
                return flattened
            indices = np.linspace(0, flattened.size - 1, 512, dtype=np.int64)
            return flattened[indices]

        is_multichannel_a_line = frame.ndim == 2 and 1 <= frame.shape[0] <= 16
        channels = (
            [downsample(frame[index]) for index in range(frame.shape[0])]
            if is_multichannel_a_line
            else [downsample(frame)]
        )
        source_channels = (
            [source_frame[index].reshape(-1) for index in range(source_frame.shape[0])]
            if is_multichannel_a_line
            else [source_frame.reshape(-1)]
        )

        def format_metrics(signal: np.ndarray) -> dict[str, Any]:
            count = max(1, int(signal.size))
            if np.issubdtype(signal.dtype, np.floating):
                finite = np.isfinite(signal)
                nonfinite_fraction = 1.0 - float(np.count_nonzero(finite)) / count
                finite_signal = signal[finite]
            else:
                nonfinite_fraction = 0.0
                finite_signal = signal
            zero_fraction = (
                float(np.count_nonzero(finite_signal == 0)) / count
                if finite_signal.size
                else 0.0
            )
            full_scale_fraction: float | None = None
            full_scale_value: int | float | None = None
            if np.issubdtype(signal.dtype, np.integer):
                full_scale_value = int(np.iinfo(signal.dtype).max)
                full_scale_fraction = (
                    float(np.count_nonzero(signal == full_scale_value)) / count
                )
            return {
                "dtype": str(signal.dtype),
                "zero_fraction": zero_fraction,
                "nonfinite_fraction": nonfinite_fraction,
                "full_scale_fraction": full_scale_fraction,
                "full_scale_value": full_scale_value,
                "all_zero": bool(signal.size and np.all(signal == 0)),
            }

        payload: dict[str, Any] = {
            "host_monotonic_ns": event.host_monotonic_ns,
            "values": channels[0].tolist(),
            "channels": [channel.tolist() for channel in channels],
            "channel_count": len(channels),
            "shape": [int(value) for value in frame.shape],
            "preview_sample_count": int(channels[0].size),
            "geometry": "a_line" if is_multichannel_a_line else "frame",
            "format_metrics": [format_metrics(channel) for channel in source_channels],
        }
    else:
        if event.modality == "imu":
            labels = ("imu_trunk", "imu_left", "imu_right")
            channels = [
                values[:, device_index, 0].astype(float).tolist()
                for device_index in range(min(values.shape[1], len(labels)))
            ]
            payload = {
                "host_monotonic_ns": event.host_monotonic_ns,
                "values": channels[0] if channels else [],
                "channels": channels,
                "labels": list(labels[: len(channels)]),
                "channel": "acc_x",
                "channel_count": len(channels),
            }
        elif event.modality == "encoder":
            labels = ("left_position", "right_position")
            channels = [
                values[:, 0].astype(float).tolist(),
                values[:, 3].astype(float).tolist(),
            ]
            payload = {
                "host_monotonic_ns": event.host_monotonic_ns,
                "values": channels[0],
                "channels": channels,
                "labels": list(labels),
                "channel": "position",
                "channel_count": len(channels),
            }
        else:
            signal = values[:, 0]
            rate = event.sample_rate_hz or 1.0
            x = (event.first_sample_index + np.arange(signal.size)) / rate
            payload = {
                "host_monotonic_ns": event.host_monotonic_ns,
                "x": x.astype(float).tolist(),
                "values": signal.astype(float).tolist(),
                "channel": "voltage",
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


def _ultrasound_formal_frame_count(
    headers: Sequence[Any],
    *,
    frame_rate_hz: float,
    formal_start_ns: int,
    formal_stop_ns: int,
) -> int:
    """Count formal frames from block headers without rereading large payloads."""

    period_ns = 1_000_000_000 / frame_rate_hz
    count = 0
    for header in headers:
        frame_times = header.host_monotonic_ns + np.rint(
            np.arange(header.sample_count, dtype=np.float64) * period_ns
        ).astype(np.int64)
        count += int(
            np.count_nonzero(
                (frame_times >= formal_start_ns) & (frame_times <= formal_stop_ns)
            )
        )
    return count


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
    quality_rules = load_quality_rules()
    storage_policy = load_storage_policy()
    # A failed disk-space preflight occurs before Catalog, Session, acquisition
    # lock, or Trial-directory side effects.  It cannot create a misleading
    # recoverable package that never started sampling.
    disk_space_evidence = check_disk_space(root, storage_policy)
    layout = TrialLayout.build(
        root,
        request.project_uuid,
        request.subject_uuid,
        request.session_uuid,
        request.trial_uuid,
        project_partition=request.project_code,
        subject_code=request.subject_code,
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
        project_code=request.project_code,
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
    sync_edge_events: list[SyncPulseEvent] = []
    trigger_count = 0
    first_trigger_host_monotonic_ns: int | None = None
    first_trigger_utc_ns: int | None = None
    accepted_trigger: SyncPulseEvent | None = None
    anchors: dict[str, list[tuple[float, int]]] = {name: [] for name in adapters}
    last_preview_ns = {name: 0 for name in adapters}
    preview_history = BoundedPreviewHistory()
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
            hdf5_trial_metadata = {
                # This immutable metadata is written when the raw file is
                # created.  No post-finalization patching is required or
                # permitted; only facts already known before sampling belong
                # here.  Trigger/finalization results remain in the Manifest.
                "schema_version": "1.1.0",
                "project_uuid": str(request.project_uuid),
                "subject_uuid": str(request.subject_uuid),
                "session_uuid": str(request.session_uuid),
                "trial_uuid": str(request.trial_uuid),
                "project_code": request.project_code,
                "project_name": request.project_name,
                "subject_code": request.subject_code,
                "condition": condition.model_dump(mode="json"),
                "experiment_metadata": request.experiment_metadata.model_dump(
                    mode="json"
                ),
                "versions": {
                    "application": "Exo Collector",
                    "application_version": __version__,
                    "core_version": __version__,
                    "hdf5_signal_format_version": HDF5_SIGNAL_VERSION,
                    "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                    "protocol_version": request.protocol_version,
                    "configuration_version": request.config_version,
                    "condition_definition_version": request.protocol_version,
                },
                "clock_policy": {
                    "primary_timeline": "host_monotonic_ns",
                    "host_monotonic_api": "time.perf_counter_ns",
                    "utc_role": "audit_only_not_interval_measurement",
                    "formal_t0": "first_qualified_sync_rising_edge",
                    "pretrigger_raw_data_preserved": True,
                    "per_sample_host_monotonic_timestamps": True,
                    "device_clock_mapping": (
                        "post_acquisition_affine_mapping_to_host_monotonic"
                    ),
                },
            }
            for modality in ("imu", "encoder", "sync_pulse"):
                writers[modality] = _create_hdf5_writer(
                    layout.partial_path(f"raw/{modality}.h5"),
                    adapters[modality],
                    trial_metadata=hdf5_trial_metadata,
                )

            transition(TrialState.READY, "all required simulated devices are ready")
            start_token = StartToken()
            for adapter in adapters.values():
                adapter.start(start_token)
            transition(
                TrialState.WAITING_SYNC,
                "devices are sampling pre-trigger data; waiting for a qualified sync rising edge",
            )

            started_at_utc: datetime | None = None
            recording_deadline_ns: int | None = None
            formal_stop_monotonic_ns: int | None = None
            sync_wait_deadline_ns = (
                None
                if request.sync_wait_timeout_s is None
                else start_token.host_monotonic_ns
                + int(request.sync_wait_timeout_s * 1_000_000_000)
            )

            def sync_payload(status: str, quality: str) -> dict[str, Any]:
                trigger_time = (
                    None
                    if started_at_utc is None
                    else started_at_utc.isoformat().replace("+00:00", "Z")
                )
                return {
                    "status": status,
                    "quality": quality,
                    "trigger_count": trigger_count,
                    "first_trigger_host_monotonic_ns": first_trigger_host_monotonic_ns,
                    "trigger_time_utc": trigger_time,
                }

            _publish(
                publish,
                WorkerEvent(
                    event_type=WorkerEventType.SYNC,
                    trial_uuid=str(request.trial_uuid),
                    modality="sync_pulse",
                    message="Waiting for synchronization trigger",
                    payload=sync_payload("WAITING_SYNC", "WAITING"),
                ),
            )

            def accept_raw_event(modality: str, event: Any, now_ns: int) -> None:
                nonlocal pulse_event_count
                nonlocal trigger_count
                nonlocal first_trigger_host_monotonic_ns
                nonlocal first_trigger_utc_ns
                nonlocal accepted_trigger
                nonlocal started_at_utc
                nonlocal recording_deadline_ns

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
                    sync_edge_events.append(event)
                    if event.edge_type is EdgeType.RISING:
                        acceptance_boundaries: list[int] = []
                        if formal_stop_monotonic_ns is not None:
                            acceptance_boundaries.append(formal_stop_monotonic_ns)
                        if first_trigger_host_monotonic_ns is None:
                            if sync_wait_deadline_ns is not None:
                                acceptance_boundaries.append(sync_wait_deadline_ns)
                        elif recording_deadline_ns is not None:
                            acceptance_boundaries.append(recording_deadline_ns)
                        acceptance_boundary_ns = (
                            min(acceptance_boundaries)
                            if acceptance_boundaries
                            else None
                        )
                        is_qualified = (
                            acceptance_boundary_ns is None
                            or event.host_monotonic_ns <= acceptance_boundary_ns
                        )
                        if is_qualified:
                            trigger_count += 1
                        if is_qualified and first_trigger_host_monotonic_ns is None:
                            accepted_trigger = event
                            first_trigger_host_monotonic_ns = event.host_monotonic_ns
                            first_trigger_utc_ns = start_token.host_utc_ns + (
                                event.host_monotonic_ns - start_token.host_monotonic_ns
                            )
                            started_at_utc = datetime.fromtimestamp(
                                first_trigger_utc_ns / 1e9, timezone.utc
                            )
                            recording_deadline_ns = (
                                None
                                if request.duration_s is None
                                else first_trigger_host_monotonic_ns
                                + int(request.duration_s * 1_000_000_000)
                            )
                            transition(
                                TrialState.RECORDING,
                                "qualified sync rising edge received; formal Trial t0 established",
                            )
                            trigger_payload = sync_payload("TRIGGERED", "PASS")
                            trigger_payload.update(
                                {
                                    "pulse_id": event.pulse_id,
                                    "source_device": event.source_device,
                                    "confidence": event.confidence,
                                }
                            )
                            _publish(
                                publish,
                                WorkerEvent(
                                    event_type=WorkerEventType.SYNC,
                                    trial_uuid=str(request.trial_uuid),
                                    modality="sync_pulse",
                                    message="Synchronization trigger accepted",
                                    payload=trigger_payload,
                                ),
                            )
                    return
                else:
                    raise TypeError(f"Unsupported raw event: {type(event).__name__}")

                if now_ns - last_preview_ns[modality] >= 66_000_000:
                    # Retain the same low-rate, spatially downsampled view used
                    # for UI telemetry.  Expensive PNG rendering is deferred
                    # until every adapter and Writer has stopped.
                    preview_history.capture(event)
                    _publish(publish, _preview_event(event, request.trial_uuid))
                    last_preview_ns[modality] = now_ns

            # Emit an initial health snapshot immediately, then at 2 Hz.
            last_health_ns = start_token.host_monotonic_ns - 500_000_000
            last_heartbeat_ns = start_token.host_monotonic_ns
            stop_reason = "controlled stop requested"
            while True:
                processed = False
                now_ns = time.perf_counter_ns()
                if stop_signal.is_set():
                    stop_reason = "manual stop requested"
                    formal_stop_monotonic_ns = max(
                        now_ns,
                        first_trigger_host_monotonic_ns or now_ns,
                    )
                    break
                if recording_deadline_ns is not None and now_ns >= recording_deadline_ns:
                    stop_reason = "configured post-trigger duration elapsed"
                    formal_stop_monotonic_ns = recording_deadline_ns
                    break
                if (
                    first_trigger_host_monotonic_ns is None
                    and sync_wait_deadline_ns is not None
                    and now_ns >= sync_wait_deadline_ns
                ):
                    stop_reason = "synchronization wait timeout elapsed"
                    formal_stop_monotonic_ns = sync_wait_deadline_ns
                    break
                for modality, adapter in adapters.items():
                    for _ in range(32):
                        event = adapter.get_event(timeout=0)
                        if event is None:
                            break
                        processed = True
                        accept_raw_event(modality, event, now_ns)

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
                                    "dropped_packets": health.dropped_packets,
                                    "queue_depth": health.queue_depth,
                                    "queue_capacity": health.queue_capacity,
                                    "last_data_host_monotonic_ns": (
                                        health.last_data_host_monotonic_ns
                                    ),
                                    "sampled_at_utc": health.sampled_at_utc.isoformat().replace(
                                        "+00:00", "Z"
                                    ),
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
                                **sync_payload(
                                    "TRIGGERED"
                                    if first_trigger_host_monotonic_ns is not None
                                    else "WAITING_SYNC",
                                    "PASS"
                                    if first_trigger_host_monotonic_ns is not None
                                    else "WAITING",
                                ),
                            },
                        ),
                    )
                    last_health_ns = now_ns
                if now_ns - last_heartbeat_ns >= 1_000_000_000:
                    activity.heartbeat()
                    last_heartbeat_ns = now_ns
                if not processed:
                    time.sleep(0.001)

            stopping_transition_recorded = False
            assert formal_stop_monotonic_ns is not None
            if first_trigger_host_monotonic_ns is not None:
                transition(TrialState.STOPPING, stop_reason)
                stopping_transition_recorded = True
            for modality, adapter in adapters.items():
                stop_reports[modality] = adapter.stop()
            # A stop can race the final produced batch; drain every bounded raw queue.
            for modality, adapter in adapters.items():
                while (event := adapter.get_event(timeout=0)) is not None:
                    accept_raw_event(modality, event, time.perf_counter_ns())
            final_health = {
                modality: adapter.health() for modality, adapter in adapters.items()
            }
            final_adapter_states = {
                modality: adapter.state.value for modality, adapter in adapters.items()
            }
            fatal_adapter_faults = {
                modality: report.fault
                for modality, report in stop_reports.items()
                if report.fault is not None
            }
            if fatal_adapter_faults:
                # A device may fault after the last 2 Hz health poll but just
                # before a short Trial stops. Preserve every batch already
                # accepted by the orchestrator through clean Writer barriers,
                # then fail the package instead of publishing it as if the
                # acquisition had completed normally.
                for writer in writers.values():
                    writer.close()
                for adapter in adapters.values():
                    adapter.close()
                detail = "; ".join(
                    f"{modality}: {fault}"
                    for modality, fault in fatal_adapter_faults.items()
                )
                raise AdapterError(f"fatal adapter fault after stop: {detail}")

            if first_trigger_host_monotonic_ns is None:
                failure_reason = (
                    "No qualified synchronization rising edge was received before "
                    + stop_reason
                )
                failure_document = {
                    "schema_version": "1.0.0",
                    "trial_uuid": str(request.trial_uuid),
                    "state": TrialState.RECOVERABLE.value,
                    "failure_code": "MISSING_SYNC_TRIGGER",
                    "reason": failure_reason,
                    "sync": sync_payload("MISSING_TRIGGER", "FAIL"),
                    "modality_counts": dict(counts),
                    "pulse_event_count": pulse_event_count,
                    "stop_reports": {
                        name: asdict(report) for name, report in stop_reports.items()
                    },
                }
                _publish(
                    publish,
                    WorkerEvent(
                        event_type=WorkerEventType.SYNC,
                        trial_uuid=str(request.trial_uuid),
                        modality="sync_pulse",
                        message=failure_reason,
                        payload=sync_payload("MISSING_TRIGGER", "FAIL"),
                    ),
                )
                assert journal is not None
                journal.write("sync_trigger_missing", failure_document)
                publish_json(layout, "reports/sync_failure.json", failure_document)
                transition(TrialState.RECOVERABLE, failure_reason)
                for writer in writers.values():
                    writer.close()
                for adapter in adapters.values():
                    adapter.close()
                journal.close_incomplete()
                raise MissingSyncTriggerError(
                    failure_reason,
                    layout.recording_directory,
                )

            if not stopping_transition_recorded:
                transition(
                    TrialState.STOPPING,
                    f"{stop_reason}; sync trigger was accepted during final queue drain",
                )

            transition(TrialState.FINALIZING, "all adapters acknowledged stop")
            stopped_reading_ns = formal_stop_monotonic_ns
            stopped_utc_ns = start_token.host_utc_ns + (
                stopped_reading_ns - start_token.host_monotonic_ns
            )
            stopped_at_utc = datetime.fromtimestamp(
                stopped_utc_ns / 1e9,
                timezone.utc,
            )
            assert first_trigger_host_monotonic_ns is not None
            assert first_trigger_utc_ns is not None
            assert accepted_trigger is not None
            assert started_at_utc is not None
            formal_duration_s = max(
                0.0,
                (stopped_reading_ns - first_trigger_host_monotonic_ns) / 1e9,
            )
            pretrigger_duration_s = max(
                0.0,
                (
                    first_trigger_host_monotonic_ns
                    - start_token.host_monotonic_ns
                )
                / 1e9,
            )
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

            preview_bundle = publish_quality_preview_pngs(
                layout,
                preview_history,
                formal_t0_host_monotonic_ns=first_trigger_host_monotonic_ns,
            )
            mappings = _clock_mappings(descriptors, anchors)
            formal_item_counts: dict[str, int] = {
                "ultrasound": _ultrasound_formal_frame_count(
                    ultrasound_scan.headers,
                    frame_rate_hz=descriptors["ultrasound"].nominal_rate_hz,
                    formal_start_ns=first_trigger_host_monotonic_ns,
                    formal_stop_ns=stopped_reading_ns,
                )
            }
            signal_evidence: dict[str, SignalEvidence] = {}
            hdf5_signal_evidence: dict[str, SignalEvidence] = {}
            for modality in ("imu", "encoder", "sync_pulse"):
                scanned_signal = scan_hdf5_signal_evidence(
                    layout.partial_path(f"raw/{modality}.h5"),
                    formal_start_ns=first_trigger_host_monotonic_ns,
                    formal_stop_ns=stopped_reading_ns,
                )
                hdf5_signal_evidence[modality] = scanned_signal
                formal_item_counts[modality] = scanned_signal.formal_sample_count
                if modality in {"imu", "encoder"}:
                    signal_evidence[modality] = scanned_signal

            sequence_gap_counts = {
                modality: (
                    ultrasound_scan.sequence_gap_count
                    if modality == "ultrasound"
                    else hdf5_signal_evidence[modality].sequence_gap_count
                )
                for modality in adapters
            }
            dropped_batch_counts = {
                modality: (
                    stop_reports[modality].injected_dropped_batches
                    + stop_reports[modality].raw_queue_overflows
                )
                for modality in adapters
            }
            quality_sync_edges = tuple(
                SyncEdgeEvidence(
                    pulse_id=event.pulse_id,
                    edge_type=event.edge_type.value,
                    host_monotonic_ns=event.host_monotonic_ns,
                    pulse_width_ns=event.pulse_width_ns,
                )
                for event in sync_edge_events
                if (
                    event.host_monotonic_ns >= first_trigger_host_monotonic_ns
                    and event.host_monotonic_ns <= stopped_reading_ns
                )
            )
            modality_by_clock_domain = {
                descriptor.clock_domain: modality
                for modality, descriptor in descriptors.items()
            }
            mapping_evidence = tuple(
                ClockMappingEvidence(
                    modality=modality_by_clock_domain[mapping.source_clock_domain],
                    anchor_count=mapping.anchor_count,
                    rms_residual_ns=mapping.residuals.rms_ns,
                )
                for mapping in mappings
                if mapping.source_clock_domain in modality_by_clock_domain
            )
            ultrasound_soft_metrics = preview_bundle.soft_metrics["ultrasound"]
            quality_evidence = TrialQualityEvidence(
                formal_duration_s=formal_duration_s,
                formal_item_counts=formal_item_counts,
                sequence_gap_counts=sequence_gap_counts,
                dropped_batch_counts=dropped_batch_counts,
                sync_edges=quality_sync_edges,
                first_trigger_host_monotonic_ns=first_trigger_host_monotonic_ns,
                clock_mappings=mapping_evidence,
                ultrasound=UltrasoundEvidence(
                    formal_frame_count=formal_item_counts["ultrasound"],
                    zero_fraction=ultrasound_soft_metrics.get("zero_fraction"),
                    saturation_fraction=ultrasound_soft_metrics.get(
                        "saturation_fraction"
                    ),
                ),
                signals=signal_evidence,
                disk_space=disk_space_evidence,
            )
            quality_evaluation = evaluate_trial_quality(
                quality_evidence,
                quality_rules,
            )
            issues = list(quality_evaluation.issues)
            grade = quality_evaluation.grade
            quality_configuration_document = {
                "schema_version": "1.0.0",
                "algorithm_version": quality_evaluation.algorithm_version,
                "quality_rules": quality_rules.model_dump(mode="json"),
                "storage_policy": storage_policy.model_dump(mode="json"),
            }
            quality_configuration_path = publish_json(
                layout,
                "derived/quality_rules_snapshot.json",
                quality_configuration_document,
            )
            quality_configuration_hash = sha256_file(quality_configuration_path)
            finalize_monotonic_ns = max(
                time.perf_counter_ns(),
                stopped_reading_ns,
            )
            finalize_utc_ns = start_token.host_utc_ns + (
                finalize_monotonic_ns - start_token.host_monotonic_ns
            )
            finalized_at_utc = datetime.fromtimestamp(
                finalize_utc_ns / 1e9,
                timezone.utc,
            )

            statistics = {
                "schema_version": "1.0.0",
                "trial_uuid": str(request.trial_uuid),
                "duration_s": formal_duration_s,
                "pretrigger_duration_s": pretrigger_duration_s,
                "modality_counts": counts,
                "formal_item_counts": formal_item_counts,
                "sequence_gap_counts": sequence_gap_counts,
                "dropped_batch_counts": dropped_batch_counts,
                "pulse_event_count": pulse_event_count,
                "trigger_count": trigger_count,
                "first_trigger_host_monotonic_ns": first_trigger_host_monotonic_ns,
                "first_trigger_utc_ns": first_trigger_utc_ns,
                "ultrasound_block_count": ultrasound_scan.complete_block_count,
                "stop_reports": {name: asdict(report) for name, report in stop_reports.items()},
                "soft_quality_metrics": preview_bundle.soft_metrics,
                "quality_assessment": {
                    "algorithm_version": quality_evaluation.algorithm_version,
                    "rules_snapshot_relative_path": "derived/quality_rules_snapshot.json",
                    "rules_snapshot_sha256": quality_configuration_hash,
                    "computed_grade": grade.value,
                    "rule_count": len(quality_evaluation.results),
                    "unassessed_rule_count": quality_evaluation.unassessed_count,
                    "rule_results": [
                        result.model_dump(mode="json")
                        for result in quality_evaluation.results
                    ],
                },
            }
            quality_report = {
                "schema_version": "1.0.0",
                "trial_uuid": str(request.trial_uuid),
                "computed_grade": grade.value,
                "required_artifacts_complete": True,
                "integrity_checks_passed": True,
                "issues": [issue.model_dump(mode="json") for issue in issues],
                "algorithm_version": quality_evaluation.algorithm_version,
                "rules_snapshot_relative_path": "derived/quality_rules_snapshot.json",
                "rules_snapshot_sha256": quality_configuration_hash,
                "rule_count": len(quality_evaluation.results),
                "unassessed_rule_count": quality_evaluation.unassessed_count,
                "rule_results": [
                    result.model_dump(mode="json")
                    for result in quality_evaluation.results
                ],
                "evidence": quality_evidence.model_dump(mode="json"),
                "soft_metrics": preview_bundle.soft_metrics,
            }
            device_status_fieldnames = (
                "schema_version",
                "trial_uuid",
                "modality",
                "item_kind",
                "device_id",
                "required",
                "health_status",
                "device_status",
                "adapter_state",
                "nominal_sample_rate_hz",
                "actual_sample_rate_hz",
                "persisted_item_count",
                "batches_emitted",
                "emitted_item_count",
                "injected_dropped_batches",
                "dropped_item_count",
                "sequence_gap_count",
                "raw_queue_overflows",
                "queue_depth",
                "queue_capacity",
                "last_data_host_monotonic_ns",
                "sampled_at_utc",
                "fault",
            )
            device_status_rows: list[dict[str, object]] = []
            for modality in adapters:
                health = final_health[modality]
                report = stop_reports[modality]
                device_status_rows.append(
                    {
                        "schema_version": "1.0.0",
                        "trial_uuid": str(request.trial_uuid),
                        "modality": modality,
                        "item_kind": "frame" if modality == "ultrasound" else "sample",
                        "device_id": report.device_id,
                        "required": str(profiles_by_modality[modality].required).lower(),
                        "health_status": health.status.value,
                        "device_status": health.device_status.value,
                        "adapter_state": final_adapter_states[modality],
                        "nominal_sample_rate_hz": health.nominal_sample_rate_hz,
                        "actual_sample_rate_hz": (
                            None
                            if health.actual_sample_rate_hz is None
                            else round(health.actual_sample_rate_hz, 6)
                        ),
                        "persisted_item_count": counts[modality],
                        "batches_emitted": report.batches_emitted,
                        "emitted_item_count": report.samples_emitted,
                        "injected_dropped_batches": report.injected_dropped_batches,
                        "dropped_item_count": health.dropped_packets,
                        "sequence_gap_count": (
                            ultrasound_scan.sequence_gap_count
                            if modality == "ultrasound"
                            else report.injected_dropped_batches
                        ),
                        "raw_queue_overflows": report.raw_queue_overflows,
                        "queue_depth": health.queue_depth,
                        "queue_capacity": health.queue_capacity,
                        "last_data_host_monotonic_ns": health.last_data_host_monotonic_ns,
                        "sampled_at_utc": health.sampled_at_utc.isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "fault": report.fault,
                    }
                )
            sync_check_fieldnames = (
                "schema_version",
                "trial_uuid",
                "status",
                "quality",
                "trigger_count",
                "pulse_event_count",
                "first_trigger_host_monotonic_ns",
                "first_trigger_utc_ns",
                "first_trigger_time_utc",
                "pulse_id",
                "source_device",
                "confidence",
                "detection_threshold",
                "pretrigger_duration_s",
                "formal_duration_s",
            )
            sync_check_rows: list[dict[str, object]] = [
                {
                    "schema_version": "1.0.0",
                    "trial_uuid": str(request.trial_uuid),
                    "status": "TRIGGERED",
                    "quality": "PASS",
                    "trigger_count": trigger_count,
                    "pulse_event_count": pulse_event_count,
                    "first_trigger_host_monotonic_ns": first_trigger_host_monotonic_ns,
                    "first_trigger_utc_ns": first_trigger_utc_ns,
                    "first_trigger_time_utc": started_at_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "pulse_id": accepted_trigger.pulse_id,
                    "source_device": accepted_trigger.source_device,
                    "confidence": round(accepted_trigger.confidence, 6),
                    "detection_threshold": accepted_trigger.detection_threshold,
                    "pretrigger_duration_s": round(pretrigger_duration_s, 9),
                    "formal_duration_s": round(formal_duration_s, 9),
                }
            ]
            pulse_audit: dict[str, dict[str, Any]] = {}
            previous_rising_ns: int | None = None
            edge_audit: list[dict[str, Any]] = []
            for edge in sorted(sync_edge_events, key=lambda item: item.host_monotonic_ns):
                mapped_utc_ns = start_token.host_utc_ns + (
                    edge.host_monotonic_ns - start_token.host_monotonic_ns
                )
                interval_ns = None
                if edge.edge_type is EdgeType.RISING:
                    interval_ns = (
                        None
                        if previous_rising_ns is None
                        else edge.host_monotonic_ns - previous_rising_ns
                    )
                    previous_rising_ns = edge.host_monotonic_ns
                edge_audit.append(
                    {
                        "event_uuid": str(edge.event_uuid),
                        "pulse_id": edge.pulse_id,
                        "edge_type": edge.edge_type.value,
                        "sample_index": edge.sample_index,
                        "host_monotonic_ns": edge.host_monotonic_ns,
                        "mapped_host_utc_ns": mapped_utc_ns,
                        "mapped_host_time_utc": datetime.fromtimestamp(
                            mapped_utc_ns / 1e9, timezone.utc
                        ).isoformat().replace("+00:00", "Z"),
                        "raw_event_host_utc_ns": edge.host_utc_ns,
                        "amplitude": edge.amplitude,
                        "pulse_width_ns": edge.pulse_width_ns,
                        "interval_since_previous_rising_ns": interval_ns,
                        "detection_threshold": edge.detection_threshold,
                        "confidence": edge.confidence,
                        "detector_version": edge.detector_version,
                        "accepted_as_formal_t0": edge.event_uuid
                        == accepted_trigger.event_uuid,
                        "within_formal_window": (
                            first_trigger_host_monotonic_ns
                            <= edge.host_monotonic_ns
                            <= stopped_reading_ns
                        ),
                    }
                )
                pulse = pulse_audit.setdefault(
                    edge.pulse_id,
                    {
                        "pulse_id": edge.pulse_id,
                        "source_device": edge.source_device,
                        "rising_host_monotonic_ns": None,
                        "falling_host_monotonic_ns": None,
                        "pulse_width_ns": None,
                        "interval_since_previous_rising_ns": None,
                        "complete": False,
                    },
                )
                if edge.edge_type is EdgeType.RISING:
                    pulse["rising_host_monotonic_ns"] = edge.host_monotonic_ns
                    pulse["interval_since_previous_rising_ns"] = interval_ns
                else:
                    pulse["falling_host_monotonic_ns"] = edge.host_monotonic_ns
                    pulse["pulse_width_ns"] = edge.pulse_width_ns
                    pulse["complete"] = True
            sync_rule_results = [
                result.model_dump(mode="json")
                for result in quality_evaluation.results
                if result.scope in {"sync", "clock"}
            ]
            sync_manifest_document = {
                "schema_version": "1.0.0",
                "trial_uuid": str(request.trial_uuid),
                "status": "TRIGGERED",
                "first_trigger_host_monotonic_ns": first_trigger_host_monotonic_ns,
                "first_trigger_utc_ns": first_trigger_utc_ns,
                "formal_stop_host_monotonic_ns": stopped_reading_ns,
                "trigger_count": trigger_count,
                "edge_count": len(edge_audit),
                "complete_pulse_count": sum(
                    bool(pulse["complete"]) for pulse in pulse_audit.values()
                ),
                "edges": edge_audit,
                "pulses": list(pulse_audit.values()),
                "clock_mappings": [
                    mapping.model_dump(mode="json") for mapping in mappings
                ],
                "quality_algorithm_version": quality_evaluation.algorithm_version,
                "quality_rule_results": sync_rule_results,
            }
            warning_count = sum(
                issue.severity is QualityIssueSeverity.WARNING for issue in issues
            )
            error_count = sum(
                issue.severity is QualityIssueSeverity.ERROR for issue in issues
            )
            warning_lines = [
                "Exo Collection Trial Quality Warnings",
                f"trial_uuid: {request.trial_uuid}",
                f"quality_grade: {grade.value}",
                f"warning_count: {warning_count}",
                f"error_count: {error_count}",
                "",
            ]
            if issues:
                warning_lines.extend(
                    f"[{issue.severity.value}] {issue.code} "
                    f"({issue.modality or 'trial'}): {issue.message}"
                    for issue in issues
                )
            else:
                warning_lines.append("No warnings or errors were detected.")
            warnings_document = "\n".join(warning_lines)
            configuration_document = {
                "schema_version": "1.0.0",
                "config_version": request.config_version,
                "project": {
                    "project_code": request.project_code,
                    "project_name": request.project_name,
                    "subject_code": request.subject_code,
                },
                "experiment_metadata": request.experiment_metadata.model_dump(
                    mode="json"
                ),
                "protocol": condition.model_dump(mode="json"),
                "quality_assessment_configuration": {
                    "algorithm_version": quality_evaluation.algorithm_version,
                    "rules_snapshot_relative_path": "derived/quality_rules_snapshot.json",
                    "rules_snapshot_sha256": quality_configuration_hash,
                    "quality_rules": quality_rules.model_dump(mode="json"),
                    "storage_policy": storage_policy.model_dump(mode="json"),
                },
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
            publish_json(layout, "derived/statistics.json", statistics)
            configuration_path = publish_json(
                layout,
                "derived/configuration_snapshot.json",
                configuration_document,
            )
            config_hash = sha256_file(configuration_path)
            publish_json(layout, "reports/quality_report.json", quality_report)
            publish_json(layout, "reports/sync_manifest.json", sync_manifest_document)
            _publish_csv(
                layout,
                "reports/device_status.csv",
                device_status_fieldnames,
                device_status_rows,
            )
            _publish_csv(
                layout,
                "reports/sync_check.csv",
                sync_check_fieldnames,
                sync_check_rows,
            )
            _publish_text(layout, "reports/warnings.txt", warnings_document)
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

            ultrasound_artifact_uuid = uuid4()
            imu_artifact_uuid = uuid4()
            encoder_artifact_uuid = uuid4()
            sync_pulse_artifact_uuid = uuid4()
            statistics_artifact_uuid = uuid4()
            quality_artifact_uuid = uuid4()
            quality_rules_artifact_uuid = uuid4()
            sync_manifest_artifact_uuid = uuid4()
            raw_artifact_uuids = (
                ultrasound_artifact_uuid,
                imu_artifact_uuid,
                encoder_artifact_uuid,
                sync_pulse_artifact_uuid,
            )
            draft_by_key: dict[str, ArtifactDraft] = {
                "ultrasound": ArtifactDraft(
                    request.trial_uuid,
                    "ultrasound",
                    ArtifactKind.RAW,
                    "application/x-exo-ultrasound-blocks",
                    "raw/ultrasound.bin",
                    artifact_uuid=ultrasound_artifact_uuid,
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
                    artifact_uuid=imu_artifact_uuid,
                    created_at_utc=started_at_utc,
                ),
                "encoder": ArtifactDraft(
                    request.trial_uuid,
                    "encoder",
                    ArtifactKind.RAW,
                    "application/x-hdf5",
                    "raw/encoder.h5",
                    artifact_uuid=encoder_artifact_uuid,
                    created_at_utc=started_at_utc,
                ),
                "sync_pulse": ArtifactDraft(
                    request.trial_uuid,
                    "sync_pulse",
                    ArtifactKind.RAW,
                    "application/x-hdf5",
                    "raw/sync_pulse.h5",
                    artifact_uuid=sync_pulse_artifact_uuid,
                    created_at_utc=started_at_utc,
                    metadata={"contains_raw_waveform": True, "contains_detected_events": True},
                ),
                "statistics": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.DERIVED,
                    "application/json",
                    "derived/statistics.json",
                    artifact_uuid=statistics_artifact_uuid,
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=raw_artifact_uuids,
                ),
                "quality_rules": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.DERIVED,
                    "application/json",
                    "derived/quality_rules_snapshot.json",
                    artifact_uuid=quality_rules_artifact_uuid,
                    created_at_utc=finalized_at_utc,
                    metadata={
                        "format_version": "1.0.0",
                        "algorithm_version": quality_evaluation.algorithm_version,
                        "content_sha256": quality_configuration_hash,
                    },
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
                    artifact_uuid=quality_artifact_uuid,
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(
                        *raw_artifact_uuids,
                        quality_rules_artifact_uuid,
                    ),
                ),
                "device_status": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.REPORT,
                    "text/csv; charset=utf-8",
                    "reports/device_status.csv",
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(statistics_artifact_uuid,),
                    metadata={
                        "format_version": "1.0.0",
                        "row_count": len(device_status_rows),
                    },
                ),
                "sync_check": ArtifactDraft(
                    request.trial_uuid,
                    "sync_pulse",
                    ArtifactKind.REPORT,
                    "text/csv; charset=utf-8",
                    "reports/sync_check.csv",
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(
                        sync_pulse_artifact_uuid,
                        statistics_artifact_uuid,
                    ),
                    metadata={
                        "format_version": "1.0.0",
                        "quality": "PASS",
                        "trigger_count": trigger_count,
                    },
                ),
                "sync_manifest": ArtifactDraft(
                    request.trial_uuid,
                    "sync_pulse",
                    ArtifactKind.REPORT,
                    "application/json",
                    "reports/sync_manifest.json",
                    artifact_uuid=sync_manifest_artifact_uuid,
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(
                        *raw_artifact_uuids,
                        quality_rules_artifact_uuid,
                    ),
                    metadata={
                        "format_version": "1.0.0",
                        "edge_count": len(edge_audit),
                        "complete_pulse_count": sync_manifest_document[
                            "complete_pulse_count"
                        ],
                        "clock_mapping_count": len(mappings),
                    },
                ),
                "warnings": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.REPORT,
                    "text/plain; charset=utf-8",
                    "reports/warnings.txt",
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(quality_artifact_uuid,),
                    metadata={
                        "format_version": "1.0.0",
                        "issue_count": len(issues),
                    },
                ),
                "us_quality_preview": ArtifactDraft(
                    request.trial_uuid,
                    "ultrasound",
                    ArtifactKind.REPORT,
                    "image/png",
                    US_PREVIEW_PATH,
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(ultrasound_artifact_uuid,),
                    metadata=preview_bundle.ultrasound_metadata,
                ),
                "imu_encoder_preview": ArtifactDraft(
                    request.trial_uuid,
                    "trial",
                    ArtifactKind.REPORT,
                    "image/png",
                    SIGNAL_PREVIEW_PATH,
                    created_at_utc=finalized_at_utc,
                    source_artifact_uuids=(
                        imu_artifact_uuid,
                        encoder_artifact_uuid,
                    ),
                    metadata=preview_bundle.signal_metadata,
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

            manifest = TrialManifest(
                project_uuid=request.project_uuid,
                project_code=request.project_code,
                project_name=request.project_name,
                subject_uuid=request.subject_uuid,
                subject_code=request.subject_code,
                session_uuid=request.session_uuid,
                trial_uuid=request.trial_uuid,
                state=TrialState.FINALIZED,
                condition=condition,
                timing=TrialTiming(
                    started_at_utc=started_at_utc,
                    stopped_at_utc=stopped_at_utc,
                    finalized_at_utc=finalized_at_utc,
                    start_host_monotonic_ns=first_trigger_host_monotonic_ns,
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
                    sync_event_artifact_uuids=[
                        artifact_map["sync_pulse"].artifact_uuid,
                        artifact_map["sync_manifest"].artifact_uuid,
                    ],
                ),
                quality=QualitySummary(
                    computed_grade=grade,
                    required_artifacts_complete=True,
                    integrity_checks_passed=True,
                    algorithm_version=quality_evaluation.algorithm_version,
                    assessed_at_utc=finalized_at_utc,
                    issues=issues,
                    metric_count=len(quality_evaluation.results),
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
                duration_s=formal_duration_s,
                modality_counts=counts,
                pulse_event_count=pulse_event_count,
                trigger_count=trigger_count,
                first_trigger_host_monotonic_ns=first_trigger_host_monotonic_ns,
                quality_grade=grade.value,
            )
    except BaseException as exc:
        missing_sync_during_failure = (
            machine.state is TrialState.WAITING_SYNC
            and first_trigger_host_monotonic_ns is None
            and not isinstance(exc, MissingSyncTriggerError)
        )
        failure_target = {
            TrialState.PREPARING: TrialState.FAILED,
            TrialState.READY: TrialState.FAILED,
            TrialState.WAITING_SYNC: TrialState.RECOVERABLE,
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
        if stop_reports:
            failure_payload["stop_reports"] = {
                name: asdict(report) for name, report in stop_reports.items()
            }
        missing_sync_reason: str | None = None
        if missing_sync_during_failure:
            missing_sync_reason = (
                "No qualified synchronization rising edge was established before "
                f"{type(exc).__name__}: {exc}"
            )
            missing_sync_payload = {
                "status": "MISSING_TRIGGER",
                "quality": "FAIL",
                "trigger_count": trigger_count,
                "first_trigger_host_monotonic_ns": None,
                "trigger_time_utc": None,
            }
            failure_payload.update(
                {
                    "failure_code": "MISSING_SYNC_TRIGGER",
                    "sync_status": "MISSING_TRIGGER",
                    "sync": missing_sync_payload,
                }
            )
            sync_failure_document = {
                "schema_version": "1.0.0",
                **failure_payload,
                "failure_state": machine.state.value,
                "state": TrialState.RECOVERABLE.value,
                "recovery_state": TrialState.RECOVERABLE.value,
                "reason": missing_sync_reason,
                "modality_counts": dict(counts),
                "pulse_event_count": pulse_event_count,
                "stop_reports": {
                    name: asdict(report) for name, report in stop_reports.items()
                },
            }
            _publish(
                publish,
                WorkerEvent(
                    event_type=WorkerEventType.SYNC,
                    trial_uuid=str(request.trial_uuid),
                    modality="sync_pulse",
                    message=missing_sync_reason,
                    payload=missing_sync_payload,
                ),
            )
            if journal is not None and not journal.closed:
                try:
                    journal.write("sync_trigger_missing", sync_failure_document)
                except BaseException:
                    pass
            if layout.recording_directory.is_dir():
                try:
                    sync_failure_path = layout.path("reports/sync_failure.json")
                    if not sync_failure_path.exists():
                        publish_json(
                            layout,
                            "reports/sync_failure.json",
                            sync_failure_document,
                        )
                except BaseException:
                    pass

        if isinstance(exc, MissingSyncTriggerError):
            # The controlled missing-trigger path already wrote its dedicated
            # audit report.  Do not mislabel it as a finalization failure.
            pass
        elif journal is not None and not journal.closed:
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
        if missing_sync_during_failure and missing_sync_reason is not None:
            raise MissingSyncTriggerError(
                missing_sync_reason,
                layout.recording_directory,
                primary_exception=exc,
            ) from exc
        raise
    finally:
        if catalog is not None:
            try:
                catalog.close()
            except BaseException:
                pass
        activity_lock.release()


__all__ = ["MissingSyncTriggerError", "run_simulated_trial"]
