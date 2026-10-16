"""Enable-bounded online recordings using the collector's raw storage formats.

Producers only enqueue references with ``put_nowait``. The single writer owns
all files. Raw batches retain their original indices, dtype and timestamps;
batches crossing an enable edge are preserved whole and readers must apply the
explicit half-open [start, stop) interval in ``online_session.json``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from queue import Queue, Empty, Full
from uuid import UUID, uuid4

import numpy as np

from exo_collection.acquisition.recording_stream import RecordingBoundary, RecordingBoundaryKind
from exo_collection.domain.events import SampleBatch, FrameBatch
from exo_collection.domain.models import ArtifactKind, Condition, utc_now
from exo_collection.domain.states import TrialState
from exo_collection.storage.layout import TrialLayout
from exo_collection.storage.manifest import (TrialManifest, TrialTiming, SoftwareProvenance,
    ConfigurationSnapshot, ModalityManifest, QualitySummary, AbnormalTermination, DeviceProvenance)
from exo_collection.storage.package import (ArtifactDraft, publish_artifact, publish_json,
    finalize_trial_package)
from exo_collection.writers.hdf5_signal import Hdf5SignalWriter
from exo_collection.writers.binary_block import BlockBinaryWriter


def _json(value):
    if dataclasses.is_dataclass(value):
        return {key: _json(item) for key, item in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _json(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, (Path, datetime, UUID)):
        return str(value)
    return value


class _OnlineLayout(TrialLayout):
    @property
    def subject_directory(self):
        return self.dataset_root

    @property
    def _trial_leaf_name(self):
        return self.started_at_utc.astimezone().strftime("%Y%m%d_%H%M%S_%f")


class OnlineRecording:
    """Nonblocking ingress shared by motor, inference, and raw queue consumers.

    A stopped session remains eligible for delayed raw batches until every
    connected raw source passes its stop timestamp. A finite timeout closes an
    unverifiable tail as ABORTED, never as a successful lossless recording.
    """
    def __init__(self, root, *, queue_capacity=2048, tail_timeout_s=1.0, now=None):
        self.root = Path(root)
        self._queue = Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()
        self._sessions = {}
        self._failed_starts = []
        self._active = None
        self._context = {}
        self._streams = {}
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._last_start = None
        self._tail_timeout_s = tail_timeout_s
        self._closing = False
        self.error = None
        self.last_path = None
        self.completed = []
        self._thread = threading.Thread(target=self._run, name="online-recording", daemon=True)
        self._thread.start()

    @property
    def active(self):
        with self._lock:
            return self._active is not None

    def configure(self, metadata=None, streams=None):
        with self._lock:
            if metadata is not None:
                self._context = dict(metadata)
            if streams is not None:
                self._streams = dict(streams)
                if self._active is not None:
                    self._sessions[self._active]["streams"].update(streams)

    def _put(self, item):
        try:
            self._queue.put_nowait(item)
            return True
        except Full:
            self.error = "online recording queue overflow; recording incomplete"
            for session in self._sessions.values():
                if self.error not in session["errors"]:
                    session["errors"].append(self.error)
            return False

    def motor_event(self, kind, payload):
        """Called under the motor lock; no I/O, waits or Qt callbacks here."""
        at = int(payload.get("host_monotonic_ns", time.perf_counter_ns())) if isinstance(payload, dict) else int(payload.host_monotonic_ns)
        with self._lock:
            if self._closing:
                return
            if kind == "enabled":
                if self._active is not None:
                    return
                start = self._now()
                if self._last_start is not None and start <= self._last_start:
                    start = self._last_start + timedelta(microseconds=1)
                self._last_start = start
                token = str(uuid4())
                session = {"token": token, "start_ns": at, "stop_ns": None,
                    "started_utc": start, "metadata": dict(self._context),
                    "streams": dict(self._streams), "errors": [], "watermarks": {},
                    "reason": None}
                self._sessions[token] = session
                self._active = token
                if not self._put(("start", token, session)):
                    self._active = None
                    self._sessions.pop(token)
                    self._failed_starts.append((session, self.error))
                    return
            token = self._active
            if token is not None:
                self._put(("event", token, (kind, at, payload)))
                if kind == "disabled":
                    session = self._sessions[token]
                    session.update(stop_ns=at, reason=payload.get("reason", "disabled"),
                                   deadline=time.monotonic() + self._tail_timeout_s)
                    self._active = None
                    self._put(("stop", token, None))

    def prediction(self, payload):
        with self._lock:
            if self._active is not None:
                self._put(("event", self._active,
                    ("prediction", int(payload["host_monotonic_ns"]), payload)))

    def raw(self, modality, item, descriptor=None):
        # Invoked before Hub converts/flat-flattens the event. Never consume a
        # queue twice or reconstruct raw samples from the bounded feature ring.
        with self._lock:
            if isinstance(item, RecordingBoundary):
                if item.kind in (RecordingBoundaryKind.FAULT, RecordingBoundaryKind.END):
                    self.source_failed(modality, f"raw stream {item.kind}: {item.message}")
                return
            event = getattr(item, "event", item)
            at = getattr(event, "host_monotonic_ns", None)
            if at is None:
                return
            count = getattr(event, "sample_count", getattr(event, "frame_count", 1))
            rate = getattr(event, "sample_rate_hz", None) or getattr(event, "frame_rate_hz", None)
            end = int(at) + (int(round((count - 1) * 1e9 / rate)) if rate else 0)
            for token, session in list(self._sessions.items()):
                if modality not in session["streams"]:
                    continue
                stop = session["stop_ns"]
                if end >= session["start_ns"] and (stop is None or int(at) < stop):
                    self._put(("raw", token, (modality, event, descriptor or {})))
                session["watermarks"][modality] = max(int(at), session["watermarks"].get(modality, 0))

    def source_failed(self, modality, reason):
        with self._lock:
            for session in self._sessions.values():
                if modality in session["streams"]:
                    session["errors"].append(f"{modality}: {reason}")

    def close(self, timeout=3.0):
        # Caller must disarm motors first. Flushing never delays physical stop.
        with self._lock:
            if self._active is not None:
                self.motor_event("disabled", {"host_monotonic_ns": time.perf_counter_ns(), "reason": "recorder_closed"})
            self._closing = True
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.error = "online recorder shutdown timeout; pending recording incomplete"
            return False
        return True

    def _run(self):
        opened = {}
        while True:
            try:
                kind, token, payload = self._queue.get(timeout=.02)
            except Empty:
                kind = None
            if kind is not None:
                try:
                    if kind == "start":
                        opened[token] = _SessionWriter(self.root, payload)
                        self.last_path = opened[token].layout.recording_directory
                    elif token in opened:
                        if kind == "raw":
                            opened[token].raw(*payload)
                        elif kind == "event":
                            opened[token].event(*payload)
                except Exception as exc:
                    self.error = f"online recording failed: {exc}"
                    with self._lock:
                        if token in self._sessions:
                            self._sessions[token]["errors"].append(self.error)
                        if kind == "start":
                            self._active = None if self._active == token else self._active
                            session = self._sessions.pop(token, payload)
                            self._failed_starts.append((session, self.error))
                    if token in opened:
                        opened[token].failed = True
                finally:
                    self._queue.task_done()
            # Checking queue-empty and sealing the session share the ingress
            # lock: no late raw batch can sneak behind finalization.
            ready = []
            with self._lock:
                failures, self._failed_starts = self._failed_starts, []
                if self._queue.empty():
                    for token, session in list(self._sessions.items()):
                        stop = session["stop_ns"]
                        if stop is None:
                            continue
                        passed = all(session["watermarks"].get(m, 0) >= stop for m in session["streams"])
                        if not passed and time.monotonic() < session["deadline"] and not self._closing:
                            continue
                        if not passed:
                            session["errors"].append("raw tail not verified before flush timeout")
                        self._sessions.pop(token)
                        writer = opened.pop(token, None)
                        # File work happens outside ingress lock below.
                        if writer is not None:
                            ready.append((writer, session))
                    finished = self._closing and not self._sessions and not opened and self._queue.empty()
                else:
                    finished = False
            for session, error in failures:
                self._failure_evidence(session, error)
            for writer, session in ready:
                try:
                    path = writer.finish(session)
                    self.last_path = path
                    self.completed.append(path)
                    if session["errors"]:
                        self.error = "; ".join(dict.fromkeys(session["errors"]))
                except Exception as exc:
                    self.error = f"online recording finalization failed: {exc}"
                    writer.abort(self.error)
            if finished:
                return

    def _failure_evidence(self, session, error):
        """Best effort evidence even when the start command/constructor failed."""
        try:
            layout = _OnlineLayout(dataset_root=self.root, project_uuid=uuid4(),
                subject_uuid=uuid4(), session_uuid=uuid4(), trial_uuid=uuid4(),
                started_at_utc=session["started_utc"])
            layout.exo_directory.mkdir(parents=True, exist_ok=True)
            (layout.exo_directory / "recording_failure.json").write_text(json.dumps({
                "complete": False, "error": error, "start_host_monotonic_ns": session["start_ns"],
                "start_reason": "motor_enabled_feedback_confirmed"}), encoding="utf-8")
            self.last_path = layout.recording_directory
        except Exception as exc:
            self.error = f"{error}; failure evidence could not be written: {exc}"


class _SessionWriter:
    def __init__(self, root, session):
        uid = uuid4()
        self.layout = _OnlineLayout(dataset_root=root, project_uuid=uid, subject_uuid=uid,
            session_uuid=uid, trial_uuid=uuid4(), started_at_utc=session["started_utc"])
        self.layout.create_recording()
        self.session = session
        self.writers = {}
        self.failed = False
        self.counts = {}
        self.paths = {}
        self.first_raw_times = {}
        self._motor_feedback_count = 0
        self.events = self.layout.partial_path("raw/online_events.jsonl").open("x", encoding="utf-8")
        publish_json(self.layout, ".exo/configuration_snapshot.json", _json(session["metadata"]))

    def event(self, kind, at, payload):
        if not self.failed:
            self.events.write(json.dumps({"kind": kind, "host_monotonic_ns": at,
                "payload": _json(payload)}, ensure_ascii=False) + "\n")
            if kind == "feedback":
                # Persist the service's every CRC-verified frame in the same
                # six-channel format as collector encoder data. Never derive
                # this stream from the GUI's lower-rate feedback snapshots.
                from exo_collection.adapters.encoder.simulated import ENCODER_CHANNELS
                fields = ("left_position_rad", "left_velocity_rad_s", "left_torque_nm",
                          "right_position_rad", "right_velocity_rad_s", "right_torque_nm")
                get = payload.get if isinstance(payload, dict) else lambda name, default=None: getattr(payload, name, default)
                data = np.asarray([[get(field) for field in fields]], dtype=np.float32)
                anchor = self.session["started_utc"]
                utc_anchor_ns = int(anchor.timestamp()) * 1_000_000_000 + anchor.microsecond * 1000
                batch = SampleBatch(device_id="motor_feedback", modality="encoder",
                    clock_domain="motor_device_time", host_monotonic_ns=at,
                    host_utc_ns=utc_anchor_ns + at - self.session["start_ns"],
                    first_sample_index=self._motor_feedback_count, sample_count=1,
                    sequence_number=get("seq"), device_timestamp=get("device_time"),
                    sample_rate_hz=200, data=data)
                self._motor_feedback_count += 1
                self.raw("encoder", batch, {"channels": list(ENCODER_CHANNELS),
                    "units": ["rad", "rad/s", "N*m", "rad", "rad/s", "N*m"],
                    "nominal_rate_hz": 200, "metadata": {"simulated": False,
                        "source": "motor_feedback", "host_time_semantics": "actual per-frame receive time",
                        "host_utc_semantics": "derived from session UTC/monotonic start anchor; not writer wall time",
                        "device_time_semantics": "original status frame device tick; no fitted mapping"}})

    def raw(self, modality, event, descriptor):
        if self.failed:
            return
        if not isinstance(event, (SampleBatch, FrameBatch)):
            self.event("raw_event", int(event.host_monotonic_ns), event)
            return
        self.first_raw_times.setdefault(modality, int(event.host_monotonic_ns))
        self.event("raw_batch_header", int(event.host_monotonic_ns),
                   event.model_dump(exclude={"data"}, mode="python"))
        # A modality can contain independently indexed physical streams.
        key = (modality, event.device_id)
        if key not in self.writers:
            safe = re.sub(r"[^\w.-]", "_", event.device_id)
            suffix = "" if not any(k[0] == modality for k in self.writers) else f"__{safe}"
            binary = isinstance(event, FrameBatch) or modality == "emg"
            relative = f"raw/{modality}{suffix}.{'bin' if binary else 'h5'}"
            self.paths[key] = relative
            data = np.asarray(event.data)
            metadata = {**descriptor.get("metadata", {}), **descriptor, "device_id": event.device_id,
                "device": {"device_id": event.device_id,
                    "simulated": descriptor.get("metadata", {}).get("simulated")},
                "clock_domain": event.clock_domain, "original_source": True,
                "nominal_sample_rate_hz": getattr(event, "sample_rate_hz", None),
                "nominal_frame_rate_hz": getattr(event, "frame_rate_hz", None)}
            if binary:
                writer = BlockBinaryWriter(self.layout.path(relative), dtype=data.dtype,
                    sample_shape=data.shape[1:], metadata=_json(metadata))
            else:
                channels = descriptor.get("channels") or [f"ch_{n}" for n in range(data.shape[-1])]
                writer = Hdf5SignalWriter(self.layout.partial_path(relative), channels=channels,
                    units=descriptor.get("units") or ["unknown"] * len(channels),
                    device_metadata=_json(metadata), dtype=data.dtype, sample_shape=data.shape[1:],
                    nominal_rate_hz=event.sample_rate_hz, flush_every_batches=16)
            self.writers[key] = writer
            self.counts[key] = 0
        writer = self.writers[key]
        if isinstance(writer, Hdf5SignalWriter):
            writer.append_batch(event)
        else:
            from exo_collection.adapters.ultrasound.raw_ethernet import encode_raw_ethernet_flags
            writer.append(event.data, first_sample_index=getattr(event, "first_sample_index", getattr(event, "first_frame_index", 0)),
                sequence=event.sequence_number,
                device_timestamp=int(event.device_timestamp) if event.device_timestamp is not None else None,
                host_monotonic_ns=event.host_monotonic_ns, host_utc_ns=event.host_utc_ns,
                flags=encode_raw_ethernet_flags(event.channel, event.tail_flags) if isinstance(event, FrameBatch) else 0)
        self.counts[key] += int(np.asarray(event.data).shape[0])

    def finish(self, session):
        for writer in self.writers.values():
            writer.close()
        self.events.close()
        if self.failed:
            raise RuntimeError("writer failed; original partial evidence retained")
        self.layout.publish_partial("raw/online_events.jsonl")
        complete = not session["errors"] and not self.failed
        document = {"schema_version": "1.0.0", "common_clock": "time.perf_counter_ns",
            "start_host_monotonic_ns": session["start_ns"], "stop_host_monotonic_ns": session["stop_ns"],
            "interval": "[start, stop)", "cross_boundary_batches": "preserved; filter samples by interval",
            "start_reason": "motor_enabled_feedback_confirmed", "stop_reason": session["reason"],
            "complete": complete, "errors": list(dict.fromkeys(session["errors"])),
            "first_raw_host_monotonic_ns": self.first_raw_times,
            "initial_raw_delay_ns": {m: max(0, at - session["start_ns"]) for m, at in self.first_raw_times.items()},
            "streams": _json(session["streams"]), "raw_watermarks": session["watermarks"],
            "metadata": _json(session["metadata"])}
        publish_json(self.layout, ".exo/online_session.json", document)
        now = utc_now()
        artifacts = []
        # Include binary companion files and all audit metadata in checksums.
        for file in sorted(self.layout.recording_directory.rglob("*")):
            if file.is_file():
                relative = file.relative_to(self.layout.recording_directory).as_posix()
                canonical = relative.removesuffix(".partial")
                modality = next((key[0] for key, path in self.paths.items() if canonical == path), "online_runtime")
                artifacts.append(publish_artifact(self.layout, ArtifactDraft(trial_uuid=self.layout.trial_uuid,
                    modality=modality, kind=ArtifactKind.RAW if canonical.startswith("raw/") else ArtifactKind.LOG,
                    media_type="application/octet-stream", relative_path=canonical, created_at_utc=now), now))
        config = self.layout.exo_path("configuration_snapshot.json").read_bytes()
        modalities = []
        for modality in dict.fromkeys(key[0] for key in self.writers):
            modalities.append(ModalityManifest(modality=modality, adapter_type="online_raw_ingress",
                writer_type="collector_canonical", clock_domain="host_monotonic",
                device_ids=[key[1] for key in self.writers if key[0] == modality],
                artifact_uuids=[a.artifact_uuid for a in artifacts if a.modality == modality],
                metadata={"boundary_filter_required": True}))
        manifest = TrialManifest(schema_version="1.0.0", project_uuid=self.layout.project_uuid,
            subject_uuid=self.layout.subject_uuid, session_uuid=self.layout.session_uuid,
            trial_uuid=self.layout.trial_uuid, state=TrialState.FINALIZED if complete else TrialState.ABORTED,
            condition=Condition(condition_code="online", condition_name="Online motor enable session",
                repeat_index=1, protocol_version="online-1"),
            timing=TrialTiming(started_at_utc=session["started_utc"], stopped_at_utc=now, finalized_at_utc=now,
                start_host_monotonic_ns=session["start_ns"], stop_host_monotonic_ns=session["stop_ns"],
                finalize_host_monotonic_ns=max(time.perf_counter_ns(), session["stop_ns"])),
            software=SoftwareProvenance(application="Exo Online Runtime", application_version="1.0",
                core_version="1.0", git_commit="not_available"),
            configuration=ConfigurationSnapshot(config_version="1", protocol_version="online-1",
                condition_definition_version="1", content_sha256=hashlib.sha256(config).hexdigest(),
                snapshot_relative_path=".exo/configuration_snapshot.json"),
            modalities=modalities, artifacts=artifacts,
            devices=list({device: DeviceProvenance(device_id=device, modality=modality,
                adapter_type="online_raw_ingress") for modality, device in self.writers}.values()),
            quality=QualitySummary(required_artifacts_complete=complete, integrity_checks_passed=True),
            abnormal_termination=AbnormalTermination(occurred=not complete,
                reason="; ".join(session["errors"]) or ("writer failed" if not complete else None)))
        return finalize_trial_package(self.layout, manifest)

    def abort(self, reason):
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass
        try:
            self.events.close()
            (self.layout.exo_directory / "recording_failure.json").write_text(
                json.dumps({"complete": False, "error": reason}), encoding="utf-8")
        except Exception:
            pass


def read_online_recording(path):
    """Load a closed session with sample-accurate enable interval filtering.

    Returns raw arrays in their original dtype, per-sample host timestamps,
    original indices/sequence and device-clock anchors, plus the complete audit
    event log. ``timeline`` joins raw rows and runtime events on the common host
    clock; it deliberately makes no nearest-neighbour interpolation.
    """
    import h5py
    from exo_collection.readers.binary_block import BlockBinaryReader
    from exo_collection.storage.manifest import load_manifest

    root = Path(path)
    manifest = load_manifest(root / ".exo/manifest.json")
    session = json.loads((root / ".exo/online_session.json").read_text(encoding="utf-8"))
    start, stop = session["start_host_monotonic_ns"], session["stop_host_monotonic_ns"]
    streams = {}
    timeline = []
    for artifact in manifest.artifacts:
        relative = artifact.relative_path
        if relative.endswith(".h5"):
            with h5py.File(root / relative, "r") as file:
                samples = file["samples"]
                times = samples["host_monotonic_ns"][:]
                mask = (times >= start) & (times < stop)
                stream = {"data": samples["data"][:][mask], "host_monotonic_ns": times[mask],
                    "sample_index": samples["sample_index"][:][mask],
                    "source_sequence": samples["source_sequence"][:][mask],
                    "device_time": samples["device_time"][:][mask]}
        elif relative.endswith(".bin"):
            parts, times, indices, sequences, device, flags = [], [], [], [], [], []
            with BlockBinaryReader(root / relative) as reader:
                metadata = reader.metadata
                rate = metadata.get("nominal_sample_rate_hz") or metadata.get("nominal_frame_rate_hz")
                for block in reader.iter_blocks():
                    header = block.header
                    if not rate and len(block.data) > 1:
                        raise ValueError("binary batch lacks nominal rate; cannot infer sample times")
                    stamps = header.host_monotonic_ns + np.rint(np.arange(len(block.data)) * (1e9 / rate if rate else 0)).astype(np.uint64)
                    mask = (stamps >= start) & (stamps < stop)
                    parts.append(block.data[mask])
                    times.append(stamps[mask])
                    indices.append((header.first_sample_index + np.arange(len(block.data), dtype=np.uint64))[mask])
                    sequences.append(np.full(int(mask.sum()), header.sequence, dtype=np.uint64))
                    device.append(np.full(int(mask.sum()), header.device_timestamp, dtype=np.int64))
                    flags.append(np.full(int(mask.sum()), header.flags, dtype=np.uint32))
                stream = {"data": np.concatenate(parts) if parts else np.empty((0, *reader.sample_shape), dtype=reader.dtype),
                    "host_monotonic_ns": np.concatenate(times) if times else np.empty(0, dtype=np.uint64),
                    "sample_index": np.concatenate(indices) if indices else np.empty(0, dtype=np.uint64),
                    "source_sequence": np.concatenate(sequences) if sequences else np.empty(0, dtype=np.uint64),
                    "flags": np.concatenate(flags) if flags else np.empty(0, dtype=np.uint32),
                    "device_time": np.concatenate(device) if device else np.empty(0, dtype=np.int64),
                    "device_time_semantics": "original block anchor; no assumed device tick rate"}
                if metadata.get("protocol") == "raw_ethernet_uint8":
                    from exo_collection.adapters.ultrasound.raw_ethernet import decode_raw_ethernet_flags
                    decoded = [decode_raw_ethernet_flags(int(flag)) for flag in stream["flags"]]
                    stream["channel"] = np.asarray([entry.channel for entry in decoded], dtype=np.uint8)
                    stream["has_trailer"] = np.asarray([entry.has_trailer for entry in decoded], dtype=bool)
        else:
            continue
        streams[relative] = stream
        timeline.extend({"kind": "raw_sample", "stream": relative, "row": index,
            "host_monotonic_ns": int(at)} for index, at in enumerate(stream["host_monotonic_ns"]))
    events = [json.loads(line) for line in (root / "raw/online_events.jsonl").read_text(encoding="utf-8").splitlines()]
    timeline.extend(event for event in events if start <= event["host_monotonic_ns"] < stop
                    and event["kind"] != "raw_batch_header")
    timeline.sort(key=lambda entry: entry["host_monotonic_ns"])
    return {"session": session, "manifest": manifest, "streams": streams, "events": events, "timeline": timeline}
