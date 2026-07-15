"""Strict reader, scanner, and rebuildable index for ultrasound block files."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import os
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Iterator, Sequence
import zlib

import numpy as np
from numpy.typing import NDArray

from exo_collection.writers.binary_block import (
    BLOCK_HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    INDEX_FORMAT_VERSION,
    INDEX_HEADER_SIZE,
    INDEX_HEADER_STRUCT,
    INDEX_MAGIC,
    UINT32_MAX,
    BinaryBlockError,
    BlockFormatError,
    BlockHeader,
    CRCMismatchError,
    IndexEntry,
    IndexFormatError,
    SampleIndexDiscontinuityError,
    SequenceDiscontinuityError,
    TruncatedBlockError,
    companion_paths,
    index_file_header,
    load_ultrasound_metadata,
    normalise_storage_dtype,
)


CRC_READ_CHUNK_NBYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BinaryFileScan:
    """Result of a forward-only scan that stops at the first invalid byte.

    ``error`` is never suppressed by higher-level strict APIs.  Keeping the
    partial result here lets recovery report the exact valid prefix without
    searching for a later magic value and accidentally skipping corruption.
    """

    data_path: Path
    file_size: int
    valid_bytes: int
    entries: tuple[IndexEntry, ...]
    headers: tuple[BlockHeader, ...]
    error: BinaryBlockError | None = None

    @property
    def complete_block_count(self) -> int:
        return len(self.entries)

    @property
    def complete_blocks(self) -> int:
        return len(self.entries)

    @property
    def is_clean(self) -> bool:
        return self.error is None and self.valid_bytes == self.file_size

    @property
    def sequence_gap_ranges(self) -> tuple[tuple[int, int], ...]:
        gaps: list[tuple[int, int]] = []
        expected = 0
        for header in self.headers:
            if header.sequence > expected:
                gaps.append((expected, header.sequence - 1))
            expected = header.sequence + 1
        return tuple(gaps)

    @property
    def sequence_gap_count(self) -> int:
        return sum(end - start + 1 for start, end in self.sequence_gap_ranges)


@dataclass(frozen=True, slots=True)
class BlockRecord:
    """A decoded header plus its NumPy payload."""

    header: BlockHeader
    data: NDArray[np.generic]
    file_offset: int

    @property
    def samples(self) -> NDArray[np.generic]:
        """Alias used by generic modality consumers."""

        return self.data


def _truncated_header(offset: int, actual_nbytes: int) -> TruncatedBlockError:
    return TruncatedBlockError(
        f"file ends inside block header at offset {offset}",
        offset=offset,
        expected_nbytes=BLOCK_HEADER_SIZE,
        actual_nbytes=actual_nbytes,
        section="header",
    )


def _truncated_payload(
    *, offset: int, expected_nbytes: int, actual_nbytes: int, sequence: int
) -> TruncatedBlockError:
    return TruncatedBlockError(
        f"file ends inside payload for block {sequence} at offset {offset}",
        offset=offset,
        expected_nbytes=expected_nbytes,
        actual_nbytes=actual_nbytes,
        section="payload",
    )


def scan_binary_file(
    data_path: str | os.PathLike[str],
    *,
    validate_crc: bool = True,
) -> BinaryFileScan:
    """Scan blocks from byte zero and stop at the first structural error.

    The scanner deliberately never searches ahead for another magic marker.
    Consequently a corrupt middle block cannot be silently omitted while later
    blocks appear valid.
    """

    path = Path(data_path)
    file_size = path.stat().st_size
    entries: list[IndexEntry] = []
    headers: list[BlockHeader] = []
    valid_bytes = 0
    error: BinaryBlockError | None = None
    previous_sequence: int | None = None
    minimum_sample_index = 0

    with path.open("rb") as stream:
        while valid_bytes < file_size:
            block_offset = valid_bytes
            remaining = file_size - block_offset
            if remaining < BLOCK_HEADER_SIZE:
                error = _truncated_header(block_offset, remaining)
                break
            stream.seek(block_offset)
            raw_header = stream.read(BLOCK_HEADER_SIZE)
            if len(raw_header) != BLOCK_HEADER_SIZE:
                error = _truncated_header(block_offset, len(raw_header))
                break
            try:
                header = BlockHeader.unpack(raw_header, offset=block_offset)
            except BinaryBlockError as exc:
                error = exc
                break
            if previous_sequence is not None and header.sequence <= previous_sequence:
                error = SequenceDiscontinuityError(
                    offset=block_offset,
                    expected=previous_sequence + 1,
                    actual=header.sequence,
                )
                break
            if header.first_sample_index < minimum_sample_index:
                error = SampleIndexDiscontinuityError(
                    offset=block_offset,
                    minimum=minimum_sample_index,
                    actual=header.first_sample_index,
                )
                break

            payload_offset = block_offset + BLOCK_HEADER_SIZE
            available_payload = file_size - payload_offset
            if available_payload < header.payload_nbytes:
                error = _truncated_payload(
                    offset=payload_offset,
                    expected_nbytes=header.payload_nbytes,
                    actual_nbytes=available_payload,
                    sequence=header.sequence,
                )
                break

            actual_crc = 0
            bytes_left = header.payload_nbytes
            if validate_crc:
                while bytes_left:
                    chunk = stream.read(min(bytes_left, CRC_READ_CHUNK_NBYTES))
                    if not chunk:
                        error = _truncated_payload(
                            offset=payload_offset,
                            expected_nbytes=header.payload_nbytes,
                            actual_nbytes=header.payload_nbytes - bytes_left,
                            sequence=header.sequence,
                        )
                        break
                    actual_crc = zlib.crc32(chunk, actual_crc)
                    bytes_left -= len(chunk)
                if error is not None:
                    break
                actual_crc &= UINT32_MAX
                if actual_crc != header.payload_crc32:
                    error = CRCMismatchError(
                        offset=block_offset,
                        sequence=header.sequence,
                        expected_crc32=header.payload_crc32,
                        actual_crc32=actual_crc,
                    )
                    break

            entries.append(
                IndexEntry(
                    sequence=header.sequence,
                    file_offset=block_offset,
                    first_sample_index=header.first_sample_index,
                    host_monotonic_ns=header.host_monotonic_ns,
                )
            )
            headers.append(header)
            valid_bytes = payload_offset + header.payload_nbytes
            previous_sequence = header.sequence
            minimum_sample_index = header.first_sample_index + header.sample_count

    return BinaryFileScan(
        data_path=path,
        file_size=file_size,
        valid_bytes=valid_bytes,
        entries=tuple(entries),
        headers=tuple(headers),
        error=error,
    )


def load_index(index_path: str | os.PathLike[str]) -> tuple[IndexEntry, ...]:
    """Load the fixed-record companion index and validate its own framing."""

    path = Path(index_path)
    with path.open("rb") as stream:
        raw_header = stream.read(INDEX_HEADER_SIZE)
        if len(raw_header) != INDEX_HEADER_SIZE:
            raise IndexFormatError("index header is truncated", offset=0)
        magic, version, entry_size, reserved = INDEX_HEADER_STRUCT.unpack(raw_header)
        if magic != INDEX_MAGIC:
            raise IndexFormatError(f"invalid index magic {magic!r}", offset=0)
        if version != INDEX_FORMAT_VERSION:
            raise IndexFormatError(
                f"unsupported index format version {version}", offset=0
            )
        if entry_size != INDEX_ENTRY_SIZE:
            raise IndexFormatError(
                f"invalid index entry size {entry_size}; expected {INDEX_ENTRY_SIZE}",
                offset=0,
            )
        if reserved != 0:
            raise IndexFormatError("index reserved field must be zero", offset=0)
        raw_entries = stream.read()
    if len(raw_entries) % INDEX_ENTRY_SIZE:
        trailing = len(raw_entries) % INDEX_ENTRY_SIZE
        raise IndexFormatError(
            f"index has a {trailing}-byte partial trailing entry",
            offset=INDEX_HEADER_SIZE + len(raw_entries) - trailing,
        )
    return tuple(
        IndexEntry.unpack(
            raw_entries[offset : offset + INDEX_ENTRY_SIZE],
            offset=INDEX_HEADER_SIZE + offset,
        )
        for offset in range(0, len(raw_entries), INDEX_ENTRY_SIZE)
    )


def _read_header_at(
    stream: BinaryIO, *, offset: int, file_size: int
) -> BlockHeader:
    if offset < 0 or offset > file_size:
        raise IndexFormatError(f"index offset {offset} is outside the data file")
    stream.seek(offset)
    raw = stream.read(BLOCK_HEADER_SIZE)
    if len(raw) != BLOCK_HEADER_SIZE:
        raise IndexFormatError(
            f"index points to an incomplete block header at offset {offset}",
            offset=offset,
        )
    return BlockHeader.unpack(raw, offset=offset)


def validate_index(
    data_path: str | os.PathLike[str], entries: Sequence[IndexEntry]
) -> None:
    """Cross-check every index entry and ensure it covers the entire data file."""

    path = Path(data_path)
    file_size = path.stat().st_size
    expected_offset = 0
    previous_sequence: int | None = None
    minimum_sample_index = 0
    with path.open("rb") as stream:
        for ordinal, entry in enumerate(entries):
            if previous_sequence is not None and entry.sequence <= previous_sequence:
                raise IndexFormatError(
                    f"index sequence at ordinal {ordinal} is {entry.sequence}; "
                    f"expected at least {previous_sequence + 1}"
                )
            if entry.file_offset != expected_offset:
                raise IndexFormatError(
                    f"index offset at ordinal {ordinal} is {entry.file_offset}; "
                    f"expected {expected_offset}"
                )
            if entry.first_sample_index < minimum_sample_index:
                raise IndexFormatError(
                    f"index sample at ordinal {ordinal} is {entry.first_sample_index}; "
                    f"expected at least {minimum_sample_index}"
                )
            try:
                header = _read_header_at(
                    stream, offset=entry.file_offset, file_size=file_size
                )
            except BinaryBlockError as exc:
                raise IndexFormatError(
                    f"index entry {ordinal} does not point to a valid block: {exc}",
                    offset=entry.file_offset,
                ) from exc
            if (
                header.sequence != entry.sequence
                or header.first_sample_index != entry.first_sample_index
                or header.host_monotonic_ns != entry.host_monotonic_ns
            ):
                raise IndexFormatError(
                    f"index entry {ordinal} disagrees with its block header",
                    offset=entry.file_offset,
                )
            block_end = (
                entry.file_offset + BLOCK_HEADER_SIZE + header.payload_nbytes
            )
            if block_end > file_size:
                raise IndexFormatError(
                    f"index entry {ordinal} points to a truncated payload",
                    offset=entry.file_offset,
                )
            expected_offset = block_end
            previous_sequence = entry.sequence
            minimum_sample_index = header.first_sample_index + header.sample_count
    if expected_offset != file_size:
        raise IndexFormatError(
            f"index covers {expected_offset} bytes but data file has {file_size} bytes",
            offset=expected_offset,
        )


def _write_index_atomic(
    index_path: Path, entries: Sequence[IndexEntry]
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(
        f".{index_path.name}.{os.getpid()}.rebuilding"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(index_file_header())
            for entry in entries:
                stream.write(entry.pack())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, index_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def rebuild_index(
    data_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str] | None = None,
    *,
    validate_crc: bool = True,
) -> tuple[IndexEntry, ...]:
    """Strictly scan the complete binary file and atomically replace its index."""

    data = Path(data_path)
    if index_path is None:
        _, derived_index = companion_paths(data)
        destination = derived_index
    else:
        destination = Path(index_path)
    scan = scan_binary_file(data, validate_crc=validate_crc)
    if scan.error is not None:
        raise scan.error
    _write_index_atomic(destination, scan.entries)
    return scan.entries


class BlockBinaryReader:
    """Indexed and ordered NumPy reader for a complete ultrasound artifact."""

    def __init__(
        self,
        data_path: str | os.PathLike[str],
        *,
        meta_path: str | os.PathLike[str] | None = None,
        index_path: str | os.PathLike[str] | None = None,
        validate_crc: bool = True,
        auto_rebuild_index: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        derived_meta, derived_index = companion_paths(self.data_path)
        self.meta_path = Path(meta_path) if meta_path is not None else derived_meta
        self.index_path = (
            Path(index_path) if index_path is not None else derived_index
        )
        self.validate_crc = bool(validate_crc)
        self.metadata = load_ultrasound_metadata(self.meta_path)
        self.dtype = normalise_storage_dtype(
            self.metadata.get("numpy_dtype", self.metadata["dtype"])
        )
        self.sample_shape = tuple(int(value) for value in self.metadata["sample_shape"])
        self._items_per_sample = math.prod(self.sample_shape)

        try:
            entries = load_index(self.index_path)
            validate_index(self.data_path, entries)
        except (OSError, IndexFormatError):
            if not auto_rebuild_index:
                raise
            entries = rebuild_index(
                self.data_path,
                self.index_path,
                validate_crc=self.validate_crc,
            )
        self.entries = entries
        self._entry_by_sequence = {entry.sequence: entry for entry in entries}
        self._first_sample_indices = [entry.first_sample_index for entry in entries]
        self._stream: BinaryIO | None = self.data_path.open("rb")
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def block_count(self) -> int:
        return len(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def _ensure_open(self) -> BinaryIO:
        if self._closed or self._stream is None:
            raise ValueError("I/O operation on closed BlockBinaryReader")
        return self._stream

    def _read_entry(self, entry: IndexEntry) -> BlockRecord:
        stream = self._ensure_open()
        file_size = self.data_path.stat().st_size
        header = _read_header_at(
            stream, offset=entry.file_offset, file_size=file_size
        )
        if (
            header.sequence != entry.sequence
            or header.first_sample_index != entry.first_sample_index
            or header.host_monotonic_ns != entry.host_monotonic_ns
        ):
            raise IndexFormatError(
                f"index entry for sequence {entry.sequence} is stale",
                offset=entry.file_offset,
            )
        expected_payload_nbytes = (
            header.sample_count * self._items_per_sample * self.dtype.itemsize
        )
        if header.payload_nbytes != expected_payload_nbytes:
            raise BlockFormatError(
                f"block {header.sequence} payload_nbytes={header.payload_nbytes}, "
                f"but metadata requires {expected_payload_nbytes}",
                offset=entry.file_offset,
            )
        payload = stream.read(header.payload_nbytes)
        if len(payload) != header.payload_nbytes:
            raise _truncated_payload(
                offset=entry.file_offset + BLOCK_HEADER_SIZE,
                expected_nbytes=header.payload_nbytes,
                actual_nbytes=len(payload),
                sequence=header.sequence,
            )
        if self.validate_crc:
            actual_crc = zlib.crc32(payload) & UINT32_MAX
            if actual_crc != header.payload_crc32:
                raise CRCMismatchError(
                    offset=entry.file_offset,
                    sequence=header.sequence,
                    expected_crc32=header.payload_crc32,
                    actual_crc32=actual_crc,
                )
        array = np.frombuffer(payload, dtype=self.dtype).reshape(
            (header.sample_count, *self.sample_shape), order="C"
        )
        return BlockRecord(
            header=header,
            data=array,
            file_offset=entry.file_offset,
        )

    def read_block(
        self,
        sequence: int | None = None,
        *,
        ordinal: int | None = None,
    ) -> BlockRecord:
        """Read by stored sequence, or explicitly by zero-based index ordinal."""

        if (sequence is None) == (ordinal is None):
            raise TypeError("provide exactly one of sequence or ordinal")
        if ordinal is not None:
            return self.read_block_by_ordinal(ordinal)
        assert sequence is not None
        try:
            entry = self._entry_by_sequence[sequence]
        except KeyError as exc:
            raise KeyError(f"no ultrasound block with sequence {sequence}") from exc
        return self._read_entry(entry)

    def read_block_by_sequence(self, sequence: int) -> BlockRecord:
        return self.read_block(sequence)

    def read_block_by_ordinal(self, ordinal: int) -> BlockRecord:
        try:
            entry = self.entries[ordinal]
        except IndexError as exc:
            raise IndexError(f"ultrasound block ordinal out of range: {ordinal}") from exc
        return self._read_entry(entry)

    def read_block_containing_sample(self, sample_index: int) -> BlockRecord:
        """Locate a sample through first-sample entries, respecting any gaps."""

        if not self.entries:
            raise KeyError(f"sample index {sample_index} is not present")
        ordinal = bisect_right(self._first_sample_indices, sample_index) - 1
        if ordinal < 0:
            raise KeyError(f"sample index {sample_index} is not present")
        record = self.read_block_by_ordinal(ordinal)
        start = record.header.first_sample_index
        if sample_index >= start + record.header.sample_count:
            raise KeyError(f"sample index {sample_index} falls in a data gap")
        return record

    def iter_blocks(
        self,
        *,
        start_ordinal: int = 0,
        stop_ordinal: int | None = None,
    ) -> Iterator[BlockRecord]:
        """Yield every indexed block in exact on-disk order."""

        stop = len(self.entries) if stop_ordinal is None else stop_ordinal
        for ordinal in range(start_ordinal, stop):
            yield self.read_block_by_ordinal(ordinal)

    def __iter__(self) -> Iterator[BlockRecord]:
        return self.iter_blocks()

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._closed = True

    def __enter__(self) -> "BlockBinaryReader":
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
    "BinaryFileScan",
    "BlockBinaryReader",
    "BlockRecord",
    "load_index",
    "rebuild_index",
    "scan_binary_file",
    "validate_index",
]
