"""Single-writer, chunked HDF5 storage for structured signal modalities."""

from __future__ import annotations

import json
from pathlib import Path
from threading import get_ident
from time import time_ns
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from exo_collection.domain.events import SampleBatch


HDF5_SIGNAL_FORMAT = "exo-hdf5-signal"
HDF5_SIGNAL_VERSION = "1.0.0"


class Hdf5SignalWriterError(RuntimeError):
    """Raised for invalid append operations or writer lifecycle misuse."""


class Hdf5SignalWriter:
    """Append-only HDF5 writer following the architecture's canonical layout.

    The writer owns one file handle and enforces use from its creating thread,
    matching the one-writer-process rule.  All sample datasets are extended in
    lockstep.  ``close`` marks the file clean only after a successful flush.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        channels: Sequence[str],
        units: Sequence[str],
        device_metadata: Mapping[str, Any] | str,
        trial_metadata: Mapping[str, Any] | None = None,
        clock_model: Mapping[str, Any] | None = None,
        dtype: str | np.dtype[Any] = np.float32,
        sample_shape: Sequence[int] | None = None,
        chunk_rows: int = 1024,
        nominal_rate_hz: float | None = None,
        device_time_step: int | float | None = None,
        compression: str | None = None,
        compression_opts: Any = None,
        flush_every_batches: int = 1,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.channels = tuple(str(value) for value in channels)
        self.units = tuple(str(value) for value in units)
        if not self.channels or any(not value.strip() for value in self.channels):
            raise ValueError("channels must be non-empty strings")
        if len(self.channels) != len(self.units):
            raise ValueError("channels and units must have the same length")
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in "biufc":
            raise ValueError("signal dtype must be boolean, integer, or floating point")
        self.sample_shape = tuple(
            int(value) for value in (sample_shape or (len(self.channels),))
        )
        if not self.sample_shape or any(value <= 0 for value in self.sample_shape):
            raise ValueError("sample_shape dimensions must be positive")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if nominal_rate_hz is not None and (
            nominal_rate_hz <= 0 or not np.isfinite(nominal_rate_hz)
        ):
            raise ValueError("nominal_rate_hz must be positive and finite")
        if device_time_step is not None and not np.isfinite(device_time_step):
            raise ValueError("device_time_step must be finite")
        if flush_every_batches <= 0:
            raise ValueError("flush_every_batches must be positive")
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_thread_id = get_ident()
        self._closed = False
        self._rows = 0
        self._append_count = 0
        self._last_sample_index: int | None = None
        self._flush_every_batches = flush_every_batches
        self._nominal_rate_hz = nominal_rate_hz
        self._device_time_step = device_time_step
        self._device_metadata = (
            {"device_id": device_metadata}
            if isinstance(device_metadata, str)
            else dict(device_metadata)
        )
        self._trial_metadata = dict(trial_metadata or {})
        self._clock_model = dict(clock_model or {})

        self._file = h5py.File(self.path, "w", libver="latest")
        self._file.attrs.update(
            {
                "format_name": HDF5_SIGNAL_FORMAT,
                "format_version": HDF5_SIGNAL_VERSION,
                "created_utc_ns": np.uint64(time_ns()),
                "closed_cleanly": False,
                "sample_count": np.uint64(0),
                "dtype": self.dtype.str,
            }
        )
        if nominal_rate_hz is not None:
            self._file.attrs["nominal_rate_hz"] = float(nominal_rate_hz)

        sample_group = self._file.create_group("samples")
        chunks = (chunk_rows, *self.sample_shape)
        self._data = sample_group.create_dataset(
            "data",
            shape=(0, *self.sample_shape),
            maxshape=(None, *self.sample_shape),
            chunks=chunks,
            dtype=self.dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        vector_chunk = (chunk_rows,)
        self._sample_index = sample_group.create_dataset(
            "sample_index",
            shape=(0,),
            maxshape=(None,),
            chunks=vector_chunk,
            dtype=np.uint64,
        )
        self._device_time = sample_group.create_dataset(
            "device_time",
            shape=(0,),
            maxshape=(None,),
            chunks=vector_chunk,
            dtype=np.float64,
        )
        self._host_monotonic_ns = sample_group.create_dataset(
            "host_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            chunks=vector_chunk,
            dtype=np.uint64,
        )
        self._host_utc_ns = sample_group.create_dataset(
            "host_utc_ns",
            shape=(0,),
            maxshape=(None,),
            chunks=vector_chunk,
            dtype=np.uint64,
        )
        self._source_sequence = sample_group.create_dataset(
            "source_sequence",
            shape=(0,),
            maxshape=(None,),
            chunks=vector_chunk,
            dtype=np.uint64,
        )
        self._source_sequence.attrs["unknown_value"] = np.uint64(np.iinfo(np.uint64).max)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        event_group = self._file.create_group("events")
        discontinuity_dtype = np.dtype(
            [
                ("sample_index", np.uint64),
                ("host_monotonic_ns", np.uint64),
                ("kind", string_dtype),
                ("details_json", string_dtype),
            ]
        )
        self._discontinuities = event_group.create_dataset(
            "discontinuities",
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, min(chunk_rows, 256)),),
            dtype=discontinuity_dtype,
        )
        self._event_records = event_group.create_dataset(
            "records",
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, min(chunk_rows, 256)),),
            dtype=string_dtype,
        )

        metadata_group = self._file.create_group("metadata")
        metadata_group.create_dataset(
            "channels", data=np.asarray(self.channels, dtype=object), dtype=string_dtype
        )
        metadata_group.create_dataset(
            "units", data=np.asarray(self.units, dtype=object), dtype=string_dtype
        )
        metadata_group.create_dataset(
            "device", data=self._json(self._device_metadata), dtype=string_dtype
        )
        metadata_group.create_dataset(
            "trial", data=self._json(self._trial_metadata), dtype=string_dtype
        )
        metadata_group.create_dataset(
            "clock_model", data=self._json(self._clock_model), dtype=string_dtype
        )
        metadata_group.attrs["sample_shape"] = np.asarray(self.sample_shape, dtype=np.uint64)
        metadata_group.attrs["channel_axis"] = len(self.sample_shape) - 1
        self._file.flush()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def sample_count(self) -> int:
        return self._rows

    def append(
        self,
        data: np.ndarray | Sequence[Any],
        *,
        sample_index: int | Sequence[int] | np.ndarray | None = None,
        device_time: int | float | Sequence[int | float] | np.ndarray | None = None,
        host_monotonic_ns: int | Sequence[int] | np.ndarray,
        host_utc_ns: int | Sequence[int] | np.ndarray | None = None,
        source_sequence: int | None = None,
        sample_rate_hz: float | None = None,
        device_time_step: int | float | None = None,
        events: Iterable[Any] | None = None,
    ) -> int:
        """Append one contiguous batch and return the new total row count."""

        self._check_open_and_owner()
        array = self._normalise_data(data)
        count = int(array.shape[0])
        indices = self._normalise_indices(sample_index, count)
        if self._last_sample_index is not None and int(indices[0]) <= self._last_sample_index:
            raise Hdf5SignalWriterError("sample_index must increase across appends")
        host_times = self._normalise_host_times(
            host_monotonic_ns, count=count, sample_rate_hz=sample_rate_hz
        )
        if host_utc_ns is None:
            utc_times = np.zeros(count, dtype=np.uint64)
        else:
            utc_times = self._normalise_host_times(
                host_utc_ns, count=count, sample_rate_hz=sample_rate_hz
            )
        if source_sequence is None:
            source_sequences = np.full(count, np.iinfo(np.uint64).max, dtype=np.uint64)
        else:
            if source_sequence < 0:
                raise ValueError("source_sequence must be non-negative")
            source_sequences = np.full(count, source_sequence, dtype=np.uint64)
        device_times = self._normalise_device_times(
            device_time,
            count=count,
            sample_rate_hz=sample_rate_hz,
            device_time_step=device_time_step,
        )

        if self._last_sample_index is not None and int(indices[0]) != self._last_sample_index + 1:
            self.append_discontinuity(
                sample_index=int(indices[0]),
                host_monotonic_ns=int(host_times[0]),
                kind="sample_index_gap",
                details={
                    "expected": self._last_sample_index + 1,
                    "actual": int(indices[0]),
                    "missing_count": int(indices[0]) - self._last_sample_index - 1,
                },
            )

        old_size = self._rows
        new_size = old_size + count
        datasets = (
            self._data,
            self._sample_index,
            self._device_time,
            self._host_monotonic_ns,
            self._host_utc_ns,
            self._source_sequence,
        )
        try:
            self._data.resize((new_size, *self.sample_shape))
            self._sample_index.resize((new_size,))
            self._device_time.resize((new_size,))
            self._host_monotonic_ns.resize((new_size,))
            self._host_utc_ns.resize((new_size,))
            self._source_sequence.resize((new_size,))
            target = slice(old_size, new_size)
            self._data[target] = array
            self._sample_index[target] = indices
            self._device_time[target] = device_times
            self._host_monotonic_ns[target] = host_times
            self._host_utc_ns[target] = utc_times
            self._source_sequence[target] = source_sequences
        except BaseException:
            self._data.resize((old_size, *self.sample_shape))
            for dataset in datasets[1:]:
                dataset.resize((old_size,))
            raise

        self._rows = new_size
        self._last_sample_index = int(indices[-1])
        self._append_count += 1
        self._file.attrs["sample_count"] = np.uint64(self._rows)
        if events:
            for event in events:
                self.append_event(event)
        if self._append_count % self._flush_every_batches == 0:
            self.flush()
        return self._rows

    def append_batch(
        self,
        batch: SampleBatch,
        *,
        device_time: Sequence[int | float] | np.ndarray | None = None,
        host_monotonic_ns: Sequence[int] | np.ndarray | None = None,
        device_time_step: int | float | None = None,
        events: Iterable[Any] | None = None,
    ) -> int:
        """Append a domain ``SampleBatch`` and reconstruct regular timestamps."""

        data = np.asarray(batch.data)
        if data.shape[0] != batch.sample_count:
            raise Hdf5SignalWriterError("batch.sample_count does not match data")
        host_value: int | Sequence[int] | np.ndarray = (
            batch.host_monotonic_ns if host_monotonic_ns is None else host_monotonic_ns
        )
        device_value: int | float | Sequence[int | float] | np.ndarray | None = (
            batch.device_timestamp if device_time is None else device_time
        )
        return self.append(
            data,
            sample_index=batch.first_sample_index,
            device_time=device_value,
            host_monotonic_ns=host_value,
            host_utc_ns=batch.host_utc_ns,
            source_sequence=batch.sequence_number,
            sample_rate_hz=batch.sample_rate_hz or self._nominal_rate_hz,
            device_time_step=device_time_step,
            events=events,
        )

    def append_discontinuity(
        self,
        *,
        sample_index: int,
        host_monotonic_ns: int,
        kind: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._check_open_and_owner()
        if sample_index < 0 or host_monotonic_ns < 0 or not kind.strip():
            raise ValueError("invalid discontinuity fields")
        index = len(self._discontinuities)
        self._discontinuities.resize((index + 1,))
        self._discontinuities[index] = (
            np.uint64(sample_index),
            np.uint64(host_monotonic_ns),
            kind,
            self._json(dict(details or {})),
        )

    def append_event(self, event: Any) -> None:
        """Persist an optional structured event as canonical JSON."""

        self._check_open_and_owner()
        if hasattr(event, "model_dump"):
            payload = event.model_dump(mode="json")
        elif isinstance(event, Mapping):
            payload = dict(event)
        else:
            raise TypeError("event must be a Pydantic model or mapping")
        index = len(self._event_records)
        self._event_records.resize((index + 1,))
        self._event_records[index] = self._json(payload)

    def flush(self) -> None:
        self._check_open_and_owner()
        self._file.flush()

    def close(self, *, clean: bool = True) -> None:
        if self._closed:
            return
        self._check_owner()
        try:
            self._file.attrs["sample_count"] = np.uint64(self._rows)
            self._file.attrs["closed_utc_ns"] = np.uint64(time_ns())
            self._file.attrs["closed_cleanly"] = bool(clean)
            self._file.flush()
        finally:
            self._file.close()
            self._closed = True

    def abort(self) -> None:
        self.close(clean=False)

    def __enter__(self) -> Hdf5SignalWriter:
        self._check_open_and_owner()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close(clean=exc_type is None)
        return False

    def _normalise_data(self, data: np.ndarray | Sequence[Any]) -> np.ndarray:
        array = np.asarray(data, dtype=self.dtype)
        if self.sample_shape == (1,) and array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.shape == self.sample_shape:
            array = array.reshape((1, *self.sample_shape))
        expected_ndim = len(self.sample_shape) + 1
        if array.ndim != expected_ndim or tuple(array.shape[1:]) != self.sample_shape:
            raise Hdf5SignalWriterError(
                f"data shape must be (n, {', '.join(map(str, self.sample_shape))}); "
                f"got {array.shape}"
            )
        if array.shape[0] <= 0:
            raise Hdf5SignalWriterError("cannot append an empty data batch")
        return np.ascontiguousarray(array)

    def _normalise_indices(
        self, value: int | Sequence[int] | np.ndarray | None, count: int
    ) -> np.ndarray:
        if value is None:
            start = self._last_sample_index + 1 if self._last_sample_index is not None else 0
            result = start + np.arange(count, dtype=np.uint64)
        elif np.isscalar(value):
            if int(value) < 0:
                raise ValueError("sample_index must be non-negative")
            result = int(value) + np.arange(count, dtype=np.uint64)
        else:
            raw = np.asarray(value)
            if raw.shape != (count,) or np.any(raw < 0):
                raise ValueError("sample_index array must be non-negative and match data")
            result = raw.astype(np.uint64)
        if np.any(result[1:] <= result[:-1]):
            raise ValueError("sample_index values must be strictly increasing")
        return result

    def _normalise_host_times(
        self,
        value: int | Sequence[int] | np.ndarray,
        *,
        count: int,
        sample_rate_hz: float | None,
    ) -> np.ndarray:
        if np.isscalar(value):
            start = int(value)
            if start < 0:
                raise ValueError("host_monotonic_ns must be non-negative")
            if count > 1:
                if sample_rate_hz is None or sample_rate_hz <= 0:
                    raise ValueError("sample_rate_hz is required to expand a host timestamp")
                step = 1_000_000_000 / sample_rate_hz
                result = start + np.rint(np.arange(count) * step).astype(np.uint64)
            else:
                result = np.asarray([start], dtype=np.uint64)
        else:
            raw = np.asarray(value)
            if raw.shape != (count,) or np.any(raw < 0):
                raise ValueError("host timestamp array must be non-negative and match data")
            result = raw.astype(np.uint64)
        if np.any(result[1:] <= result[:-1]):
            raise ValueError("host timestamps must be strictly increasing")
        return result

    def _normalise_device_times(
        self,
        value: int | float | Sequence[int | float] | np.ndarray | None,
        *,
        count: int,
        sample_rate_hz: float | None,
        device_time_step: int | float | None,
    ) -> np.ndarray:
        if value is None:
            return np.full(count, np.nan, dtype=np.float64)
        if np.isscalar(value):
            step = device_time_step
            if step is None:
                step = self._device_time_step
            if step is None:
                if sample_rate_hz is None or sample_rate_hz <= 0:
                    step = 1.0
                elif "hardware_tick_hz" in self._device_metadata:
                    step = float(self._device_metadata["hardware_tick_hz"]) / sample_rate_hz
                else:
                    step = 1_000_000_000 / sample_rate_hz
            result = float(value) + np.arange(count, dtype=np.float64) * float(step)
        else:
            result = np.asarray(value, dtype=np.float64)
            if result.shape != (count,):
                raise ValueError("device_time array must match data")
        if not np.all(np.isfinite(result)):
            raise ValueError("device_time must be finite when provided")
        return result

    def _check_open_and_owner(self) -> None:
        self._check_owner()
        if self._closed:
            raise Hdf5SignalWriterError("writer is closed")

    def _check_owner(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise Hdf5SignalWriterError("HDF5 writer may only be used by its owner thread")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "HDF5_SIGNAL_FORMAT",
    "HDF5_SIGNAL_VERSION",
    "Hdf5SignalWriter",
    "Hdf5SignalWriterError",
]
