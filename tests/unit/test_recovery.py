from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exo_collection.readers.binary_block import BlockBinaryReader, load_index
from exo_collection.storage.recovery import (
    UnsafeUltrasoundRecoveryError,
    recover_ultrasound_file,
    scan_ultrasound_file,
)
from exo_collection.writers.binary_block import (
    BLOCK_HEADER_SIZE,
    BlockBinaryWriter,
    companion_paths,
)


def _make_three_blocks(path: Path) -> tuple[int, int, int]:
    sizes: list[int] = []
    with BlockBinaryWriter(
        path,
        dtype=np.uint16,
        sample_shape=(4,),
        metadata={"clock_domain": "ultrasound_device_clock"},
    ) as writer:
        for sequence in range(3):
            result = writer.append(
                np.full((2, 4), sequence, dtype=np.uint16),
                device_timestamp=sequence * 10,
                host_monotonic_ns=sequence * 100 + 1,
                host_utc_ns=sequence * 100 + 2,
            )
            sizes.append(BLOCK_HEADER_SIZE + result.header.payload_nbytes)
    return tuple(sizes)  # type: ignore[return-value]


def test_scan_reports_incomplete_tail_without_mutation(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin.partial"
    sizes = _make_three_blocks(data_path)
    original_size = data_path.stat().st_size
    with data_path.open("r+b") as stream:
        stream.truncate(original_size - 5)

    damaged_size = data_path.stat().st_size
    report = scan_ultrasound_file(data_path)
    assert report.complete_block_count == 2
    assert report.valid_bytes == sizes[0] + sizes[1]
    assert report.file_size == damaged_size
    assert report.error_kind == "TruncatedBlockError"
    assert report.error_reason
    assert report.tail_recoverable
    assert not report.intermediate_corruption
    assert data_path.stat().st_size == damaged_size

    no_change = recover_ultrasound_file(data_path, truncate=False)
    assert not no_change.truncated
    assert not no_change.index_rebuilt
    assert data_path.stat().st_size == damaged_size


@pytest.mark.parametrize("partial_tail_nbytes", [1, BLOCK_HEADER_SIZE - 1])
def test_recovery_truncates_partial_header_and_rebuilds_index(
    tmp_path: Path, partial_tail_nbytes: int
) -> None:
    data_path = tmp_path / "ultrasound.bin.partial"
    sizes = _make_three_blocks(data_path)
    clean_bytes = sum(sizes)
    with data_path.open("ab") as stream:
        stream.write(b"X" * partial_tail_nbytes)
    _, index_path = companion_paths(data_path)
    index_path.write_bytes(b"broken derived index")

    report = recover_ultrasound_file(data_path, truncate=True)
    assert report.truncated
    assert report.truncated_nbytes == partial_tail_nbytes
    assert report.valid_bytes == clean_bytes
    assert report.index_rebuilt
    assert report.error_reason  # retained for the recovery audit record
    assert data_path.stat().st_size == clean_bytes
    assert len(load_index(index_path)) == 3
    with BlockBinaryReader(data_path) as reader:
        assert len(reader) == 3


def test_recovery_removes_crc_invalid_final_block(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin.partial"
    sizes = _make_three_blocks(data_path)
    with data_path.open("r+b") as stream:
        stream.seek(sum(sizes[:2]) + BLOCK_HEADER_SIZE + 1)
        value = stream.read(1)
        stream.seek(sum(sizes[:2]) + BLOCK_HEADER_SIZE + 1)
        stream.write(bytes([value[0] ^ 0x80]))

    scan = scan_ultrasound_file(data_path)
    assert scan.complete_block_count == 2
    assert scan.error_kind == "CRCMismatchError"
    assert scan.tail_recoverable

    report = recover_ultrasound_file(data_path, truncate=True)
    assert report.truncated
    assert report.valid_bytes == sum(sizes[:2])
    with BlockBinaryReader(data_path) as reader:
        assert len(reader) == 2


def test_middle_crc_corruption_is_never_silently_skipped_or_truncated(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "ultrasound.bin.partial"
    sizes = _make_three_blocks(data_path)
    original_size = data_path.stat().st_size
    with data_path.open("r+b") as stream:
        stream.seek(sizes[0] + BLOCK_HEADER_SIZE + 2)
        value = stream.read(1)
        stream.seek(sizes[0] + BLOCK_HEADER_SIZE + 2)
        stream.write(bytes([value[0] ^ 0x01]))

    report = scan_ultrasound_file(data_path)
    assert report.complete_block_count == 1
    assert report.valid_bytes == sizes[0]
    assert report.error_kind == "CRCMismatchError"
    assert not report.tail_recoverable
    assert report.intermediate_corruption

    with pytest.raises(UnsafeUltrasoundRecoveryError, match="middle-file"):
        recover_ultrasound_file(data_path, truncate=True)
    assert data_path.stat().st_size == original_size


def test_middle_header_corruption_stops_at_exact_bad_block(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin.partial"
    sizes = _make_three_blocks(data_path)
    original_size = data_path.stat().st_size
    with data_path.open("r+b") as stream:
        stream.seek(sizes[0])
        stream.write(b"BADMAGIC")

    report = scan_ultrasound_file(data_path)
    assert report.complete_block_count == 1
    assert report.valid_bytes == sizes[0]
    assert report.error_kind == "BlockFormatError"
    assert report.intermediate_corruption
    with pytest.raises(UnsafeUltrasoundRecoveryError):
        recover_ultrasound_file(data_path, truncate=True)
    assert data_path.stat().st_size == original_size


def test_clean_recovery_can_rebuild_only_the_index(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin"
    _make_three_blocks(data_path)
    _, index_path = companion_paths(data_path)
    index_path.unlink()

    report = recover_ultrasound_file(data_path)
    assert report.is_clean
    assert report.index_rebuilt
    assert not report.truncated
    assert len(load_index(index_path)) == 3
