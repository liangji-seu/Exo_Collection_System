"""Crash recovery primitives for append-only ultrasound artifacts.

This module only determines and optionally publishes the valid prefix of an
ultrasound binary file.  Trial state transitions and recovery audit-log writes
belong to the orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

from exo_collection.readers.binary_block import (
    BinaryFileScan,
    rebuild_index,
    scan_binary_file,
)
from exo_collection.writers.binary_block import (
    BLOCK_HEADER_SIZE,
    BinaryBlockError,
    BlockHeader,
    CRCMismatchError,
    TruncatedBlockError,
    companion_paths,
)


class UnsafeUltrasoundRecoveryError(BinaryBlockError):
    """Requested truncation would hide possible middle-file corruption."""


@dataclass(frozen=True, slots=True)
class UltrasoundRecoveryResult:
    """Auditable scan/recovery result for one ultrasound binary artifact."""

    data_path: Path
    index_path: Path
    complete_block_count: int
    valid_bytes: int
    file_size: int
    error_kind: str | None
    error_reason: str | None
    error_offset: int | None
    tail_recoverable: bool
    intermediate_corruption: bool
    truncated: bool = False
    truncated_nbytes: int = 0
    index_rebuilt: bool = False

    @property
    def complete_blocks(self) -> int:
        return self.complete_block_count

    @property
    def is_clean(self) -> bool:
        return self.error_reason is None and self.valid_bytes == self.file_size

    @property
    def original_file_size(self) -> int:
        """The pre-recovery size retained even after optional truncation."""

        return self.file_size


# A concise alias for callers that only scan and do not mutate.
UltrasoundScanResult = UltrasoundRecoveryResult


def _is_last_block_crc_failure(scan: BinaryFileScan) -> bool:
    error = scan.error
    if not isinstance(error, CRCMismatchError) or error.offset is None:
        return False
    try:
        with scan.data_path.open("rb") as stream:
            stream.seek(error.offset)
            raw_header = stream.read(BLOCK_HEADER_SIZE)
        header = BlockHeader.unpack(raw_header, offset=error.offset)
    except (OSError, BinaryBlockError):
        return False
    return error.offset + BLOCK_HEADER_SIZE + header.payload_nbytes == scan.file_size


def _result_from_scan(scan: BinaryFileScan, *, index_path: Path) -> UltrasoundRecoveryResult:
    error = scan.error
    if error is None:
        return UltrasoundRecoveryResult(
            data_path=scan.data_path,
            index_path=index_path,
            complete_block_count=scan.complete_block_count,
            valid_bytes=scan.valid_bytes,
            file_size=scan.file_size,
            error_kind=None,
            error_reason=None,
            error_offset=None,
            tail_recoverable=False,
            intermediate_corruption=False,
        )

    # Truncation errors produced by scan_binary_file necessarily end at physical
    # EOF.  A CRC failure is safe to drop only when that complete-but-invalid
    # block is the final physical block.  Header/sequence failures remain unsafe
    # because treating them as a tail could discard later valid bytes.
    tail_recoverable = isinstance(error, TruncatedBlockError) or _is_last_block_crc_failure(scan)
    return UltrasoundRecoveryResult(
        data_path=scan.data_path,
        index_path=index_path,
        complete_block_count=scan.complete_block_count,
        valid_bytes=scan.valid_bytes,
        file_size=scan.file_size,
        error_kind=type(error).__name__,
        error_reason=str(error),
        error_offset=error.offset,
        tail_recoverable=tail_recoverable,
        intermediate_corruption=not tail_recoverable,
    )


def scan_ultrasound_file(
    data_path: str | os.PathLike[str],
    *,
    index_path: str | os.PathLike[str] | None = None,
    validate_crc: bool = True,
) -> UltrasoundRecoveryResult:
    """Return the complete prefix and first error without changing any file."""

    data = Path(data_path)
    if index_path is None:
        _, derived_index = companion_paths(data)
        index = derived_index
    else:
        index = Path(index_path)
    scan = scan_binary_file(data, validate_crc=validate_crc)
    return _result_from_scan(scan, index_path=index)


def recover_ultrasound_file(
    data_path: str | os.PathLike[str],
    *,
    index_path: str | os.PathLike[str] | None = None,
    truncate: bool = False,
    rebuild_idx: bool = True,
    validate_crc: bool = True,
) -> UltrasoundRecoveryResult:
    """Inspect, optionally truncate a crash tail, and rebuild the index.

    With ``truncate=False`` this function is read-only.  With ``truncate=True``
    it removes only an objectively incomplete tail (or a CRC-invalid final
    block).  Structural, sequence, and non-final CRC failures raise
    :class:`UnsafeUltrasoundRecoveryError` and the source file is untouched.
    """

    report = scan_ultrasound_file(
        data_path,
        index_path=index_path,
        validate_crc=validate_crc,
    )
    if report.error_reason is not None:
        if not truncate:
            return report
        if not report.tail_recoverable:
            raise UnsafeUltrasoundRecoveryError(
                "refusing to truncate possible middle-file corruption: "
                f"{report.error_reason}",
                offset=report.error_offset,
            )
        with report.data_path.open("r+b") as stream:
            stream.truncate(report.valid_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        report = replace(
            report,
            truncated=True,
            truncated_nbytes=report.file_size - report.valid_bytes,
        )

    if rebuild_idx:
        # After a safe truncation (or a clean scan), this strict second scan
        # both validates the result and atomically publishes the derived index.
        rebuild_index(
            report.data_path,
            report.index_path,
            validate_crc=validate_crc,
        )
        report = replace(report, index_rebuilt=True)
    return report


def rebuild_ultrasound_index(
    data_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str] | None = None,
    *,
    validate_crc: bool = True,
):
    """Public recovery-layer wrapper around strict index reconstruction."""

    return rebuild_index(
        data_path,
        index_path=index_path,
        validate_crc=validate_crc,
    )


__all__ = [
    "UnsafeUltrasoundRecoveryError",
    "UltrasoundRecoveryResult",
    "UltrasoundScanResult",
    "rebuild_ultrasound_index",
    "recover_ultrasound_file",
    "scan_ultrasound_file",
]
