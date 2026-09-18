"""Modality subscription hub: the ROS2-topic analog for the online runtime.

Each connected preview worker exposes a ``RecordingStreamEndpoint`` whose queue
carries ordered ``RecordingBoundary`` markers and ``RecordedRawEvent`` items.
``SubscriptionHub`` drains those queues (non-blocking, loss-tolerant, callable
from the inference thread), skips the boundary markers, and appends raw samples
into a per-modality ring buffer aligned on ``host_monotonic_ns``.

The ring buffer normalizes every modality into a flat time series so a
``FeatureExtractor`` can resample any combination of modalities onto one
control grid.  The channel naming used for explicit ``FeatureInput.channels``
selection is derived by ``flatten_channel_names`` (documented below).
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable
from queue import Empty
from time import perf_counter_ns
from typing import Any

import numpy as np

from exo_collection.acquisition.recording_stream import RecordedRawEvent, RecordingBoundary
from exo_collection.domain.events import (
    FrameBatch,
    GaitwayPacketEvent,
    SampleBatch,
    SyncPulseEvent,
)

RawCallback = Callable[[Any], None]

_NS_PER_S = 1_000_000_000


def flatten_channel_names(
    channels: tuple[str, ...] | list[str] | None,
    sample_shape: tuple[int, ...] | list[int] | None,
) -> tuple[str, ...]:
    """Expand a descriptor's ``channels`` to one name per flattened scalar.

    The descriptor stores ``channels`` at a coarser granularity than the raw
    data for a few modalities (IMU: 12 axis names × N devices; mocap: 3 axes ×
    M markers; ultrasound frame: one amplitude × S samples).  The runtime
    flattens each sample/frame to ``prod(sample_shape)`` scalars, so explicit
    channel selection needs the same flat naming:

    - already flat → identity;
    - 2-D shape with ``channels`` matching the *columns* → ``name[row]``;
    - 2-D shape with ``channels`` matching the *rows* → ``name[col]``;
    - 1-D shape with a single name → ``name[i]``;
    - anything else → ``ch_{i}``.

    This is only a *default*; users whose training pipeline names channels
    differently subclass ``FeatureExtractor`` instead of relying on it.
    """
    base = tuple(str(c) for c in (channels or ()))
    if sample_shape is None:
        return base or ()
    shape = tuple(int(v) for v in sample_shape)
    if not shape:
        return base
    total = int(np.prod(shape, dtype=np.int64))

    if len(base) == total:
        return base
    if len(shape) == 2 and shape[1] == len(base):
        return tuple(f"{name}[{row}]" for row in range(shape[0]) for name in base)
    if len(shape) == 2 and shape[0] == len(base):
        return tuple(f"{name}[{col}]" for name in base for col in range(shape[1]))
    if len(shape) == 1 and len(base) == 1:
        return tuple(f"{base[0]}[{i}]" for i in range(shape[0]))
    return tuple(f"ch_{i}" for i in range(total))


class _Stream:
    """Internal per-modality accumulator."""

    __slots__ = (
        "queue",
        "channels",
        "sample_shape",
        "rate_hz",
        "device_id",
        "descriptor",
        "configuration_snapshot",
        "samples",
        "times",
        "callbacks",
        "boundary",
        "received_items",
    )

    def __init__(
        self,
        queue: Any,
        *,
        channels: tuple[str, ...],
        sample_shape: tuple[int, ...],
        rate_hz: float,
        device_id: str,
        descriptor: dict[str, Any] | None,
        configuration_snapshot: dict[str, Any] | None,
    ) -> None:
        self.queue = queue
        self.channels = channels
        self.sample_shape = sample_shape
        self.rate_hz = rate_hz
        self.device_id = device_id
        self.descriptor = descriptor or {}
        self.configuration_snapshot = configuration_snapshot or {}
        self.samples: deque[np.ndarray] = deque()
        self.times: deque[int] = deque()
        self.callbacks: list[RawCallback] = []
        self.boundary: str | None = None
        self.received_items = 0


class SubscriptionHub:
    """Drains recording-stream endpoints into aligned per-modality ring buffers."""

    def __init__(self, *, buffer_capacity: int = 262_144) -> None:
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.RLock()
        self._buffer_capacity = max(1, int(buffer_capacity))

    # -- registration ------------------------------------------------------

    def register_endpoint(self, modality: str, endpoint: Any) -> None:
        """Register a ``RecordingStreamEndpoint`` (or duck-typed equivalent)."""
        descriptor = dict(getattr(endpoint, "descriptor", None) or {})
        snapshot = dict(getattr(endpoint, "configuration_snapshot", None) or {})
        channels = tuple(descriptor.get("channels") or ())
        shape_raw = descriptor.get("sample_shape")
        sample_shape = tuple(int(v) for v in shape_raw) if shape_raw else ()
        rate_hz = float(descriptor.get("nominal_rate_hz") or 0.0)
        self.register_stream(
            modality,
            endpoint.queue,
            channels=channels,
            sample_shape=sample_shape,
            rate_hz=rate_hz,
            device_id=str(getattr(endpoint, "device_id", "") or descriptor.get("device_id") or ""),
            descriptor=descriptor,
            configuration_snapshot=snapshot,
        )

    def register_stream(
        self,
        modality: str,
        queue: Any,
        *,
        channels: tuple[str, ...] | list[str] = (),
        sample_shape: tuple[int, ...] | list[int] = (),
        rate_hz: float = 0.0,
        device_id: str = "",
        descriptor: dict[str, Any] | None = None,
        configuration_snapshot: dict[str, Any] | None = None,
    ) -> None:
        modality = str(modality).strip()
        if not modality:
            raise ValueError("modality must be non-empty")
        stream = _Stream(
            queue,
            channels=tuple(str(c) for c in channels),
            sample_shape=tuple(int(v) for v in sample_shape),
            rate_hz=float(rate_hz),
            device_id=str(device_id),
            descriptor=descriptor,
            configuration_snapshot=configuration_snapshot,
        )
        with self._lock:
            self._streams[modality] = stream

    def unregister(self, modality: str) -> None:
        with self._lock:
            self._streams.pop(modality, None)

    def subscribe(self, modality: str, callback: RawCallback) -> None:
        with self._lock:
            stream = self._streams.get(modality)
            if stream is None:
                raise KeyError(f"unknown modality: {modality}")
            stream.callbacks.append(callback)

    # -- introspection -----------------------------------------------------

    @property
    def modalities(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._streams)

    def descriptor(self, modality: str) -> dict[str, Any]:
        with self._lock:
            stream = self._require(modality)
            return dict(stream.descriptor)

    def channels(self, modality: str) -> tuple[str, ...]:
        with self._lock:
            stream = self._require(modality)
            return stream.channels

    def flat_channels(self, modality: str) -> tuple[str, ...]:
        with self._lock:
            stream = self._require(modality)
            return flatten_channel_names(stream.channels, stream.sample_shape)

    def rate_hz(self, modality: str) -> float:
        with self._lock:
            return self._require(modality).rate_hz

    def boundary(self, modality: str) -> str | None:
        with self._lock:
            return self._streams.get(modality).boundary if modality in self._streams else None

    def count(self, modality: str) -> int:
        with self._lock:
            return len(self._streams[modality].samples) if modality in self._streams else 0

    # -- draining ----------------------------------------------------------

    def drain(self, max_per_stream: int | None = None) -> int:
        """Non-blockingly pull every pending item from every registered queue.

        Returns the total number of raw data items appended (boundary markers are
        consumed but not counted as data).  Safe to call from the inference thread
        or a dedicated pump thread.
        """
        consumed = 0
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            budget = max_per_stream
            while budget is None or budget > 0:
                try:
                    item = stream.queue.get_nowait()
                except Empty:
                    break
                except (OSError, ValueError):
                    break
                if budget is not None:
                    budget -= 1
                if self._ingest(stream, item):
                    consumed += 1
        return consumed

    def _ingest(self, stream: _Stream, item: Any) -> bool:
        """Route one queue item; return True if it added data samples."""
        if isinstance(item, RecordingBoundary):
            stream.boundary = str(item.kind)
            return False
        if isinstance(item, RecordedRawEvent):
            return self._ingest_raw(stream, item.event)
        return self._ingest_raw(stream, item)

    def _ingest_raw(self, stream: _Stream, event: Any) -> bool:
        if isinstance(event, SampleBatch):
            return self._append_sample_batch(stream, event)
        if isinstance(event, FrameBatch):
            return self._append_frame_batch(stream, event)
        if isinstance(event, (SyncPulseEvent, GaitwayPacketEvent)):
            # Pulse/packet events are not channel data; forward to callbacks only.
            for callback in list(stream.callbacks):
                _safe_callback(callback, event)
            return False
        return False

    def _append_sample_batch(self, stream: _Stream, event: SampleBatch) -> bool:
        data = np.asarray(event.data, dtype=np.float32)
        if data.ndim < 2 or data.shape[0] < 1:
            return False
        rate = float(event.sample_rate_hz or stream.rate_hz or 1.0)
        dt_ns = _NS_PER_S / rate if rate > 0 else 0.0
        base_ns = int(event.host_monotonic_ns)
        for index in range(data.shape[0]):
            row = np.ascontiguousarray(data[index].reshape(-1).astype(np.float32))
            t_ns = base_ns + int(round(index * dt_ns))
            self._push(stream, t_ns, row)
        for callback in list(stream.callbacks):
            _safe_callback(callback, event)
        return True

    def _append_frame_batch(self, stream: _Stream, event: FrameBatch) -> bool:
        data = np.asarray(event.data, dtype=np.float32)
        if data.ndim < 1 or data.shape[0] < 1:
            return False
        rate = float(event.frame_rate_hz or stream.rate_hz or 1.0)
        dt_ns = _NS_PER_S / rate if rate > 0 else 0.0
        base_ns = int(event.host_monotonic_ns)
        for index in range(data.shape[0]):
            frame = np.ascontiguousarray(data[index].reshape(-1).astype(np.float32))
            t_ns = base_ns + int(round(index * dt_ns))
            self._push(stream, t_ns, frame)
        for callback in list(stream.callbacks):
            _safe_callback(callback, event)
        return True

    def _push(self, stream: _Stream, t_ns: int, row: np.ndarray) -> None:
        stream.samples.append(row)
        stream.times.append(t_ns)
        stream.received_items += 1
        while len(stream.samples) > self._buffer_capacity:
            stream.samples.popleft()
            stream.times.popleft()

    # -- windowed access ---------------------------------------------------

    def window(self, modality: str, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times[n], values[n, L])`` over the trailing ``seconds``.

        An empty buffer returns zero-length arrays.  ``values`` has one row per
        sample/frame, flattened to ``prod(sample_shape)`` scalars.
        """
        stream = self._require(modality)
        with self._lock:
            samples = list(stream.samples)
            times = list(stream.times)
        if not samples:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        latest = times[-1]
        cutoff = latest - int(seconds * _NS_PER_S)
        # Fast path: everything is within the window.
        if times[0] >= cutoff:
            return np.asarray(times, dtype=np.int64), np.asarray(samples, dtype=np.float32)
        kept = [(t, v) for t, v in zip(times, samples) if t >= cutoff]
        if not kept:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        return (
            np.asarray([t for t, _ in kept], dtype=np.int64),
            np.asarray([v for _, v in kept], dtype=np.float32),
        )

    def latest(self, modality: str) -> tuple[int, np.ndarray] | None:
        with self._lock:
            stream = self._streams.get(modality)
            if stream is None or not stream.samples:
                return None
            return stream.times[-1], stream.samples[-1]

    def clear(self, modality: str) -> None:
        with self._lock:
            stream = self._streams.get(modality)
            if stream is not None:
                stream.samples.clear()
                stream.times.clear()

    # -- helpers -----------------------------------------------------------

    def _require(self, modality: str) -> _Stream:
        stream = self._streams.get(modality)
        if stream is None:
            raise KeyError(f"unknown modality: {modality}")
        return stream


def _safe_callback(callback: RawCallback, event: Any) -> None:
    try:
        callback(event)
    except Exception:
        # A subscriber must never break the data pump.
        return


__all__ = [
    "RawCallback",
    "SubscriptionHub",
    "flatten_channel_names",
]
