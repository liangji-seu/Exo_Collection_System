"""Append-only block binary writer for high-throughput ultrasound data.

The on-disk block header is deliberately small, fixed length, and independent
of Python or NumPy object layouts.  All integer fields are explicitly encoded
little-endian with no native alignment::

    <8sHHQQQQqQQII

    magic, format_version, header_size, sequence, first_sample_index,
    sample_count, payload_nbytes, device_timestamp, host_monotonic_ns,
    host_utc_ns, flags, payload_crc32

The payload immediately follows the header and is a contiguous C-order NumPy
array.  Its dtype and per-sample shape live in the companion metadata JSON.
The companion index is a rebuildable derivative; the binary file is the source
of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import time
from types import TracebackType
from typing import Any, BinaryIO, Mapping, Sequence
import zlib

import numpy as np
from numpy.typing import ArrayLike, NDArray


BLOCK_MAGIC = b"EXOUSBLK"
"""Eight-byte magic at the start of every ultrasound block."""

FORMAT_VERSION = 1
METADATA_SCHEMA_VERSION = "1.0.0"

# Explicit little-endian, standard sizes, and no implicit alignment.
BLOCK_HEADER_STRUCT = struct.Struct("<8sHHQQQQqQQII")
BLOCK_HEADER_SIZE = BLOCK_HEADER_STRUCT.size

INDEX_MAGIC = b"EXOIDX01"
INDEX_FORMAT_VERSION = 1
INDEX_HEADER_STRUCT = struct.Struct("<8sHHI")
INDEX_ENTRY_STRUCT = struct.Struct("<QQQQ")
INDEX_HEADER_SIZE = INDEX_HEADER_STRUCT.size
INDEX_ENTRY_SIZE = INDEX_ENTRY_STRUCT.size

DEVICE_TIMESTAMP_UNKNOWN = -(1 << 63)
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


class BinaryBlockError(Exception):
    """Base class for binary block and companion-file failures."""

    def __init__(self, message: str, *, offset: int | None = None) -> None:
        super().__init__(message)
        self.offset = offset


class BlockFormatError(BinaryBlockError):
    """A block header violates the published format."""


class UnsupportedFormatVersionError(BlockFormatError):
    """A block uses a format version this implementation cannot read."""


class TruncatedBlockError(BinaryBlockError):
    """The file ends partway through a header or payload."""

    def __init__(
        self,
        message: str,
        *,
        offset: int,
        expected_nbytes: int,
        actual_nbytes: int,
        section: str,
    ) -> None:
        super().__init__(message, offset=offset)
        self.expected_nbytes = expected_nbytes
        self.actual_nbytes = actual_nbytes
        self.section = section


class CRCMismatchError(BinaryBlockError):
    """A payload CRC32 differs from the CRC stored in its header."""

    def __init__(
        self,
        *,
        offset: int,
        sequence: int,
        expected_crc32: int,
        actual_crc32: int,
    ) -> None:
        super().__init__(
            (
                f"CRC32 mismatch for block {sequence} at offset {offset}: "
                f"expected 0x{expected_crc32:08x}, got 0x{actual_crc32:08x}"
            ),
            offset=offset,
        )
        self.sequence = sequence
        self.expected_crc32 = expected_crc32
        self.actual_crc32 = actual_crc32


# Descriptive compatibility aliases for callers that prefer payload-specific
# terminology.
PayloadChecksumError = CRCMismatchError
CRCError = CRCMismatchError


class SequenceDiscontinuityError(BinaryBlockError):
    """On-disk block sequences repeat or move backwards."""

    def __init__(self, *, offset: int, expected: int, actual: int) -> None:
        super().__init__(
            f"block sequence discontinuity at offset {offset}: "
            f"expected {expected}, got {actual}",
            offset=offset,
        )
        self.expected = expected
        self.actual = actual


class SampleIndexDiscontinuityError(BinaryBlockError):
    """A block overlaps or moves behind an already stored sample range."""

    def __init__(self, *, offset: int, minimum: int, actual: int) -> None:
        super().__init__(
            f"sample index discontinuity at offset {offset}: "
            f"expected at least {minimum}, got {actual}",
            offset=offset,
        )
        self.minimum = minimum
        self.actual = actual


class IndexFormatError(BinaryBlockError):
    """The rebuildable index is missing, malformed, or stale."""


class MetadataFormatError(BinaryBlockError):
    """The ultrasound metadata JSON is missing or inconsistent."""


def _require_int_range(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _normalise_sample_shape(sample_shape: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for position, dimension in enumerate(sample_shape):
        value = _require_int_range(
            f"sample_shape[{position}]", dimension, 1, UINT64_MAX
        )
        result.append(value)
    return tuple(result)


def normalise_storage_dtype(dtype: np.dtype[Any] | type[Any] | str) -> np.dtype[Any]:
    """Return a primitive, explicitly little-endian NumPy storage dtype."""

    result = np.dtype(dtype)
    if result.hasobject or result.fields is not None or result.subdtype is not None:
        raise TypeError("ultrasound dtype must be a primitive fixed-width dtype")
    if result.kind not in "biufc":
        raise TypeError(f"unsupported ultrasound dtype: {result}")
    if result.itemsize <= 0:
        raise TypeError("ultrasound dtype must have a fixed non-zero item size")
    return result.newbyteorder("<")


def companion_paths(data_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Derive ``.meta.json`` and ``.idx`` paths from a binary data path.

    ``ultrasound.bin.partial`` maps to ``ultrasound.meta.json.partial`` and
    ``ultrasound.idx.partial`` so all three temporary artifacts can be renamed
    together during Trial finalization.
    """

    path = Path(data_path)
    partial = path.name.endswith(".partial")
    core_name = path.name[: -len(".partial")] if partial else path.name
    core = Path(core_name)
    stem = core.name[: -len(core.suffix)] if core.suffix else core.name
    temporary_suffix = ".partial" if partial else ""
    return (
        path.with_name(f"{stem}.meta.json{temporary_suffix}"),
        path.with_name(f"{stem}.idx{temporary_suffix}"),
    )


@dataclass(frozen=True, slots=True)
class BlockHeader:
    """Decoded fixed-length ultrasound block header."""

    sequence: int
    first_sample_index: int
    sample_count: int
    payload_nbytes: int
    device_timestamp: int
    host_monotonic_ns: int
    host_utc_ns: int
    flags: int
    payload_crc32: int
    format_version: int = FORMAT_VERSION
    header_size: int = BLOCK_HEADER_SIZE
    block_magic: bytes = BLOCK_MAGIC

    def validate(self, *, offset: int | None = None) -> None:
        if self.block_magic != BLOCK_MAGIC:
            raise BlockFormatError(
                f"invalid block magic {self.block_magic!r}", offset=offset
            )
        if self.format_version != FORMAT_VERSION:
            raise UnsupportedFormatVersionError(
                f"unsupported block format version {self.format_version}",
                offset=offset,
            )
        if self.header_size != BLOCK_HEADER_SIZE:
            raise BlockFormatError(
                f"invalid header_size {self.header_size}; expected {BLOCK_HEADER_SIZE}",
                offset=offset,
            )
        _require_int_range("sequence", self.sequence, 0, UINT64_MAX)
        _require_int_range(
            "first_sample_index", self.first_sample_index, 0, UINT64_MAX
        )
        _require_int_range("sample_count", self.sample_count, 1, UINT64_MAX)
        _require_int_range("payload_nbytes", self.payload_nbytes, 1, UINT64_MAX)
        _require_int_range(
            "device_timestamp", self.device_timestamp, INT64_MIN, INT64_MAX
        )
        _require_int_range(
            "host_monotonic_ns", self.host_monotonic_ns, 0, UINT64_MAX
        )
        _require_int_range("host_utc_ns", self.host_utc_ns, 0, UINT64_MAX)
        _require_int_range("flags", self.flags, 0, UINT32_MAX)
        _require_int_range("payload_crc32", self.payload_crc32, 0, UINT32_MAX)

    def pack(self) -> bytes:
        self.validate()
        return BLOCK_HEADER_STRUCT.pack(
            self.block_magic,
            self.format_version,
            self.header_size,
            self.sequence,
            self.first_sample_index,
            self.sample_count,
            self.payload_nbytes,
            self.device_timestamp,
            self.host_monotonic_ns,
            self.host_utc_ns,
            self.flags,
            self.payload_crc32,
        )

    @classmethod
    def unpack(cls, raw: bytes, *, offset: int | None = None) -> "BlockHeader":
        if len(raw) != BLOCK_HEADER_SIZE:
            raise TruncatedBlockError(
                "incomplete ultrasound block header",
                offset=0 if offset is None else offset,
                expected_nbytes=BLOCK_HEADER_SIZE,
                actual_nbytes=len(raw),
                section="header",
            )
        (
            block_magic,
            format_version,
            header_size,
            sequence,
            first_sample_index,
            sample_count,
            payload_nbytes,
            device_timestamp,
            host_monotonic_ns,
            host_utc_ns,
            flags,
            payload_crc32,
        ) = BLOCK_HEADER_STRUCT.unpack(raw)
        header = cls(
            sequence=sequence,
            first_sample_index=first_sample_index,
            sample_count=sample_count,
            payload_nbytes=payload_nbytes,
            device_timestamp=device_timestamp,
            host_monotonic_ns=host_monotonic_ns,
            host_utc_ns=host_utc_ns,
            flags=flags,
            payload_crc32=payload_crc32,
            format_version=format_version,
            header_size=header_size,
            block_magic=block_magic,
        )
        header.validate(offset=offset)
        return header


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One rebuildable index entry."""

    sequence: int
    file_offset: int
    first_sample_index: int
    host_monotonic_ns: int

    def pack(self) -> bytes:
        return INDEX_ENTRY_STRUCT.pack(
            _require_int_range("sequence", self.sequence, 0, UINT64_MAX),
            _require_int_range("file_offset", self.file_offset, 0, UINT64_MAX),
            _require_int_range(
                "first_sample_index", self.first_sample_index, 0, UINT64_MAX
            ),
            _require_int_range(
                "host_monotonic_ns", self.host_monotonic_ns, 0, UINT64_MAX
            ),
        )

    @classmethod
    def unpack(cls, raw: bytes, *, offset: int | None = None) -> "IndexEntry":
        if len(raw) != INDEX_ENTRY_SIZE:
            raise IndexFormatError("incomplete index entry", offset=offset)
        return cls(*INDEX_ENTRY_STRUCT.unpack(raw))


@dataclass(frozen=True, slots=True)
class BlockWriteResult:
    """Location and header of a successfully appended data block."""

    header: BlockHeader
    index_entry: IndexEntry

    @property
    def sequence(self) -> int:
        return self.header.sequence

    @property
    def file_offset(self) -> int:
        return self.index_entry.file_offset


def _metadata_document(
    *,
    dtype: np.dtype[Any],
    sample_shape: tuple[int, ...],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supplied = dict(metadata or {})
    compression = supplied.get("compression", "none")
    if compression != "none":
        raise NotImplementedError(
            "only uncompressed ultrasound payloads are supported in format version 1"
        )
    document = supplied
    # Invariants are written last so arbitrary caller metadata cannot make the
    # file describe a layout different from the bytes actually stored.
    document.update(
        {
            "schema_version": METADATA_SCHEMA_VERSION,
            "format_version": FORMAT_VERSION,
            "endianness": "little",
            "byte_order": "little",
            "dtype": dtype.name,
            "numpy_dtype": dtype.str,
            "sample_shape": list(sample_shape),
            "compression": "none",
            "clock_domain": str(supplied.get("clock_domain", "unspecified")),
        }
    )
    return document


def write_ultrasound_metadata(
    path: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write metadata through a same-directory temporary file and rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                dict(document),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not overwrite and destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_ultrasound_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate fields required to reconstruct payload arrays."""

    metadata_path = Path(path)
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataFormatError(
            f"cannot read ultrasound metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise MetadataFormatError("ultrasound metadata root must be a JSON object")
    if document.get("format_version") != FORMAT_VERSION:
        raise MetadataFormatError(
            f"unsupported metadata format_version {document.get('format_version')!r}"
        )
    if document.get("endianness", document.get("byte_order")) != "little":
        raise MetadataFormatError("ultrasound metadata must declare little endianness")
    if document.get("compression", "none") != "none":
        raise MetadataFormatError("compressed ultrasound payloads are not supported")
    dtype_value = document.get("numpy_dtype", document.get("dtype"))
    try:
        normalise_storage_dtype(dtype_value)
    except (TypeError, ValueError) as exc:
        raise MetadataFormatError(f"invalid ultrasound dtype: {dtype_value!r}") from exc
    shape_value = document.get("sample_shape")
    if not isinstance(shape_value, list):
        raise MetadataFormatError("sample_shape must be a JSON array")
    try:
        _normalise_sample_shape(shape_value)
    except (TypeError, ValueError) as exc:
        raise MetadataFormatError("invalid sample_shape") from exc
    clock_domain = document.get("clock_domain")
    if not isinstance(clock_domain, str) or not clock_domain:
        raise MetadataFormatError("clock_domain must be a non-empty string")
    return document


def index_file_header() -> bytes:
    """Return the fixed index-file header."""

    return INDEX_HEADER_STRUCT.pack(
        INDEX_MAGIC, INDEX_FORMAT_VERSION, INDEX_ENTRY_SIZE, 0
    )


class BlockBinaryWriter:
    """Append C-order NumPy sample batches to an ultrasound block file.

    Parameters
    ----------
    mode:
        ``"x"`` (default) creates all artifacts exclusively, ``"w"`` is an
        explicit overwrite for disposable/test paths, and ``"a"`` resumes a
        clean existing temporary recording without modifying earlier bytes.
    fsync_on_append:
        If true, each block and its index entry are forced to stable storage.
        Regardless of this setting both streams are flushed after every block,
        and the data stream is flushed before the derived index entry.
    """

    def __init__(
        self,
        data_path: str | os.PathLike[str],
        *,
        dtype: np.dtype[Any] | type[Any] | str | None = None,
        sample_shape: Sequence[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        meta_path: str | os.PathLike[str] | None = None,
        index_path: str | os.PathLike[str] | None = None,
        mode: str = "x",
        fsync_on_append: bool = False,
    ) -> None:
        if mode not in {"x", "w", "a"}:
            raise ValueError("mode must be 'x', 'w', or 'a'")
        self.data_path = Path(data_path)
        derived_meta, derived_index = companion_paths(self.data_path)
        self.meta_path = Path(meta_path) if meta_path is not None else derived_meta
        self.index_path = (
            Path(index_path) if index_path is not None else derived_index
        )
        self.fsync_on_append = bool(fsync_on_append)
        self._data_stream: BinaryIO | None = None
        self._index_stream: BinaryIO | None = None
        self._closed = False
        self._next_sequence = 0
        self._next_sample_index = 0

        supplied_metadata = dict(metadata or {})
        if dtype is None:
            dtype = supplied_metadata.get("numpy_dtype", supplied_metadata.get("dtype"))
        if sample_shape is None:
            sample_shape = supplied_metadata.get("sample_shape")

        existing = mode == "a" and self.data_path.exists()
        if existing:
            self._open_existing(dtype=dtype, sample_shape=sample_shape)
        else:
            if dtype is None:
                raise TypeError("dtype is required when creating an ultrasound file")
            if sample_shape is None:
                raise TypeError(
                    "sample_shape is required when creating an ultrasound file"
                )
            self.dtype = normalise_storage_dtype(dtype)
            self.sample_shape = _normalise_sample_shape(sample_shape)
            self.metadata = _metadata_document(
                dtype=self.dtype,
                sample_shape=self.sample_shape,
                metadata=supplied_metadata,
            )
            self._create_new("x" if mode == "a" else mode)

    def _create_new(self, mode: str) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        overwrite = mode == "w"
        if mode == "x":
            conflicts = [
                path
                for path in (self.data_path, self.meta_path, self.index_path)
                if path.exists()
            ]
            if conflicts:
                raise FileExistsError(conflicts[0])
        write_ultrasound_metadata(
            self.meta_path, self.metadata, overwrite=overwrite
        )
        data_mode = "wb" if overwrite else "xb"
        try:
            self._data_stream = self.data_path.open(data_mode)
            self._index_stream = self.index_path.open(data_mode)
            self._index_stream.write(index_file_header())
            self._index_stream.flush()
            if self.fsync_on_append:
                os.fsync(self._index_stream.fileno())
        except Exception:
            self._close_streams_without_flush()
            raise

    def _open_existing(
        self,
        *,
        dtype: np.dtype[Any] | type[Any] | str | None,
        sample_shape: Sequence[int] | None,
    ) -> None:
        self.metadata = load_ultrasound_metadata(self.meta_path)
        stored_dtype = normalise_storage_dtype(
            self.metadata.get("numpy_dtype", self.metadata["dtype"])
        )
        stored_shape = _normalise_sample_shape(self.metadata["sample_shape"])
        if dtype is not None and normalise_storage_dtype(dtype) != stored_dtype:
            raise MetadataFormatError("requested dtype differs from existing metadata")
        if sample_shape is not None and _normalise_sample_shape(sample_shape) != stored_shape:
            raise MetadataFormatError(
                "requested sample_shape differs from existing metadata"
            )
        self.dtype = stored_dtype
        self.sample_shape = stored_shape

        # Local import avoids a writer/reader import cycle at module load time.
        from exo_collection.readers.binary_block import (
            load_index,
            rebuild_index,
            scan_binary_file,
            validate_index,
        )

        scan = scan_binary_file(self.data_path, validate_crc=True)
        if scan.error is not None:
            raise scan.error
        try:
            entries = load_index(self.index_path)
            validate_index(self.data_path, entries)
        except (FileNotFoundError, IndexFormatError):
            entries = rebuild_index(
                self.data_path, self.index_path, validate_crc=True
            )
        if entries:
            last_header = scan.headers[-1]
            self._next_sequence = last_header.sequence + 1
            self._next_sample_index = (
                last_header.first_sample_index + last_header.sample_count
            )
        self._data_stream = self.data_path.open("ab")
        self._index_stream = self.index_path.open("ab")

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def next_sample_index(self) -> int:
        return self._next_sample_index

    def append(
        self,
        samples: ArrayLike,
        *,
        device_timestamp: int | None = None,
        host_monotonic_ns: int | None = None,
        host_utc_ns: int | None = None,
        first_sample_index: int | None = None,
        sequence: int | None = None,
        flags: int = 0,
    ) -> BlockWriteResult:
        """Append one sample/frame batch and its rebuildable index entry."""

        self._ensure_open()
        array = np.asarray(samples, dtype=self.dtype, order="C")
        expected_ndim = len(self.sample_shape) + 1
        if array.ndim != expected_ndim or tuple(array.shape[1:]) != self.sample_shape:
            raise ValueError(
                f"samples must have shape (count, {', '.join(map(str, self.sample_shape))})"
                if self.sample_shape
                else "samples must be a one-dimensional scalar-sample array"
            )
        sample_count = _require_int_range(
            "sample_count", array.shape[0], 1, UINT64_MAX
        )
        contiguous: NDArray[Any] = np.ascontiguousarray(array, dtype=self.dtype)
        payload = memoryview(contiguous).cast("B")
        payload_nbytes = _require_int_range(
            "payload_nbytes", payload.nbytes, 1, UINT64_MAX
        )
        expected_nbytes = (
            sample_count * math.prod(self.sample_shape) * self.dtype.itemsize
        )
        if payload_nbytes != expected_nbytes:
            raise ValueError("payload byte count is inconsistent with dtype and shape")

        chosen_sequence = self._next_sequence if sequence is None else sequence
        chosen_sequence = _require_int_range(
            "sequence", chosen_sequence, 0, UINT64_MAX
        )
        if chosen_sequence < self._next_sequence:
            raise ValueError(
                f"sequence must be at least {self._next_sequence}; got {chosen_sequence}"
            )
        chosen_first_index = (
            self._next_sample_index
            if first_sample_index is None
            else _require_int_range(
                "first_sample_index", first_sample_index, 0, UINT64_MAX
            )
        )
        if first_sample_index is None:
            chosen_first_index = self._next_sample_index
        if chosen_first_index < self._next_sample_index:
            raise ValueError(
                f"first_sample_index must be at least {self._next_sample_index}; "
                f"got {chosen_first_index}"
            )
        device_time = (
            DEVICE_TIMESTAMP_UNKNOWN
            if device_timestamp is None
            else _require_int_range(
                "device_timestamp", device_timestamp, INT64_MIN, INT64_MAX
            )
        )
        monotonic_time = (
            time.perf_counter_ns()
            if host_monotonic_ns is None
            else _require_int_range(
                "host_monotonic_ns", host_monotonic_ns, 0, UINT64_MAX
            )
        )
        utc_time = (
            time.time_ns()
            if host_utc_ns is None
            else _require_int_range("host_utc_ns", host_utc_ns, 0, UINT64_MAX)
        )
        flags_value = _require_int_range("flags", flags, 0, UINT32_MAX)
        checksum = zlib.crc32(payload) & UINT32_MAX
        header = BlockHeader(
            sequence=chosen_sequence,
            first_sample_index=chosen_first_index,
            sample_count=sample_count,
            payload_nbytes=payload_nbytes,
            device_timestamp=device_time,
            host_monotonic_ns=monotonic_time,
            host_utc_ns=utc_time,
            flags=flags_value,
            payload_crc32=checksum,
        )

        assert self._data_stream is not None
        assert self._index_stream is not None
        file_offset = self._data_stream.tell()
        self._data_stream.write(header.pack())
        self._data_stream.write(payload)
        self._data_stream.flush()
        if self.fsync_on_append:
            os.fsync(self._data_stream.fileno())

        entry = IndexEntry(
            sequence=chosen_sequence,
            file_offset=file_offset,
            first_sample_index=chosen_first_index,
            host_monotonic_ns=monotonic_time,
        )
        # Update state after the source-of-truth block is flushed.  If the
        # derivative index write fails, reopening can safely rebuild it.
        self._next_sequence = chosen_sequence + 1
        self._next_sample_index = chosen_first_index + sample_count
        self._index_stream.write(entry.pack())
        self._index_stream.flush()
        if self.fsync_on_append:
            os.fsync(self._index_stream.fileno())
        return BlockWriteResult(header=header, index_entry=entry)

    # ``write`` is intentionally an alias: writer-worker code can use the
    # generic Writer verb without losing the more explicit append API.
    write = append

    def flush(self, *, fsync: bool | None = None) -> None:
        self._ensure_open()
        assert self._data_stream is not None
        assert self._index_stream is not None
        self._data_stream.flush()
        self._index_stream.flush()
        should_sync = self.fsync_on_append if fsync is None else fsync
        if should_sync:
            os.fsync(self._data_stream.fileno())
            os.fsync(self._index_stream.fileno())

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            if self._data_stream is not None and self._index_stream is not None:
                self.flush(fsync=True)
        except BaseException as exc:  # close both handles even after disk errors
            error = exc
        finally:
            self._close_streams_without_flush()
            self._closed = True
        if error is not None:
            raise error

    def _close_streams_without_flush(self) -> None:
        for stream in (self._index_stream, self._data_stream):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._index_stream = None
        self._data_stream = None

    def _ensure_open(self) -> None:
        if self._closed or self._data_stream is None or self._index_stream is None:
            raise ValueError("I/O operation on closed BlockBinaryWriter")

    def __enter__(self) -> "BlockBinaryWriter":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "BLOCK_HEADER_SIZE",
    "BLOCK_HEADER_STRUCT",
    "BLOCK_MAGIC",
    "CRCError",
    "CRCMismatchError",
    "DEVICE_TIMESTAMP_UNKNOWN",
    "FORMAT_VERSION",
    "INDEX_ENTRY_SIZE",
    "INDEX_ENTRY_STRUCT",
    "INDEX_FORMAT_VERSION",
    "INDEX_HEADER_SIZE",
    "INDEX_HEADER_STRUCT",
    "INDEX_MAGIC",
    "BinaryBlockError",
    "BlockBinaryWriter",
    "BlockFormatError",
    "BlockHeader",
    "BlockWriteResult",
    "IndexEntry",
    "IndexFormatError",
    "MetadataFormatError",
    "PayloadChecksumError",
    "SampleIndexDiscontinuityError",
    "SequenceDiscontinuityError",
    "TruncatedBlockError",
    "UnsupportedFormatVersionError",
    "companion_paths",
    "index_file_header",
    "load_ultrasound_metadata",
    "normalise_storage_dtype",
    "write_ultrasound_metadata",
]
