from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from exo_collection.readers.binary_block import (
    BlockBinaryReader,
    load_index,
    scan_binary_file,
)
from exo_collection.writers.binary_block import (
    BLOCK_HEADER_SIZE,
    BLOCK_MAGIC,
    FORMAT_VERSION,
    CRCMismatchError,
    BlockBinaryWriter,
    IndexFormatError,
    companion_paths,
)


def _writer(path: Path, *, mode: str = "x") -> BlockBinaryWriter:
    return BlockBinaryWriter(
        path,
        dtype=np.int16,
        sample_shape=(2, 3),
        metadata={
            "clock_domain": "ultrasound_device_clock",
            "channels": ["rf"],
            "nominal_frame_rate_hz": 100.0,
        },
        mode=mode,
    )


def test_header_is_fixed_explicit_little_endian_and_metadata_is_complete(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "ultrasound.bin"
    samples = np.arange(24, dtype=np.int16).reshape(4, 2, 3)
    with _writer(data_path) as writer:
        result = writer.append(
            samples,
            device_timestamp=1234,
            host_monotonic_ns=5678,
            host_utc_ns=9012,
            flags=3,
        )

    raw = data_path.read_bytes()
    assert BLOCK_HEADER_SIZE == struct.calcsize("<8sHHQQQQqQQII") == 76
    fields = struct.unpack("<8sHHQQQQqQQII", raw[:BLOCK_HEADER_SIZE])
    assert fields[:3] == (BLOCK_MAGIC, FORMAT_VERSION, BLOCK_HEADER_SIZE)
    assert fields[3:11] == (0, 0, 4, samples.nbytes, 1234, 5678, 9012, 3)
    assert fields[11] == result.header.payload_crc32
    assert raw[BLOCK_HEADER_SIZE:] == samples.astype("<i2").tobytes(order="C")

    metadata_path, index_path = companion_paths(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["endianness"] == "little"
    assert metadata["dtype"] == "int16"
    assert metadata["sample_shape"] == [2, 3]
    assert metadata["compression"] == "none"
    assert metadata["clock_domain"] == "ultrasound_device_clock"
    assert index_path.exists()


def test_round_trip_by_sequence_ordinal_sample_index_and_order(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin"
    batches = [
        np.arange(12, dtype=np.int16).reshape(2, 2, 3),
        np.arange(18, dtype=np.int16).reshape(3, 2, 3) + 100,
        np.arange(6, dtype=np.int16).reshape(1, 2, 3) + 200,
    ]
    with _writer(data_path) as writer:
        writer.append(
            batches[0],
            device_timestamp=10,
            host_monotonic_ns=1_000,
            host_utc_ns=2_000,
        )
        writer.append(
            batches[1],
            first_sample_index=10,
            device_timestamp=20,
            host_monotonic_ns=2_000,
            host_utc_ns=3_000,
        )
        writer.append(
            batches[2],
            device_timestamp=30,
            host_monotonic_ns=3_000,
            host_utc_ns=4_000,
        )

    with BlockBinaryReader(data_path) as reader:
        assert len(reader) == 3
        np.testing.assert_array_equal(reader.read_block(1).data, batches[1])
        np.testing.assert_array_equal(
            reader.read_block(ordinal=2).data, batches[2]
        )
        assert reader.read_block_containing_sample(11).header.sequence == 1
        with pytest.raises(KeyError, match="data gap"):
            reader.read_block_containing_sample(5)
        assert [record.header.sequence for record in reader] == [0, 1, 2]
        assert [record.header.host_utc_ns for record in reader] == [
            2_000,
            3_000,
            4_000,
        ]


@pytest.mark.parametrize("index_damage", ["missing", "truncated", "wrong-offset"])
def test_reader_rebuilds_missing_or_invalid_index(
    tmp_path: Path, index_damage: str
) -> None:
    data_path = tmp_path / "ultrasound.bin"
    first = np.ones((2, 2, 3), dtype=np.int16)
    second = np.full((2, 2, 3), 7, dtype=np.int16)
    with _writer(data_path) as writer:
        writer.append(first, host_monotonic_ns=100, host_utc_ns=200)
        writer.append(second, host_monotonic_ns=300, host_utc_ns=400)

    _, index_path = companion_paths(data_path)
    if index_damage == "missing":
        index_path.unlink()
    elif index_damage == "truncated":
        index_path.write_bytes(index_path.read_bytes()[:-3])
    else:
        damaged = bytearray(index_path.read_bytes())
        # First entry file_offset is the second uint64 in the 32-byte record.
        struct.pack_into("<Q", damaged, 16 + 8, 123)
        index_path.write_bytes(damaged)

    with BlockBinaryReader(data_path) as reader:
        np.testing.assert_array_equal(reader.read_block(1).data, second)
    entries = load_index(index_path)
    assert [entry.file_offset for entry in entries] == [
        0,
        BLOCK_HEADER_SIZE + first.nbytes,
    ]


def test_reader_detects_payload_crc_corruption(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin"
    with _writer(data_path) as writer:
        writer.append(
            np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            host_monotonic_ns=100,
            host_utc_ns=200,
        )

    with data_path.open("r+b") as stream:
        stream.seek(BLOCK_HEADER_SIZE + 4)
        original = stream.read(1)
        stream.seek(BLOCK_HEADER_SIZE + 4)
        stream.write(bytes([original[0] ^ 0xFF]))

    scan = scan_binary_file(data_path)
    assert isinstance(scan.error, CRCMismatchError)
    with BlockBinaryReader(data_path) as reader:
        with pytest.raises(CRCMismatchError):
            reader.read_block(0)


def test_resume_mode_only_appends_after_a_strict_scan(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin.partial"
    first = np.ones((2, 2, 3), dtype=np.int16)
    second = np.full((1, 2, 3), 2, dtype=np.int16)
    with _writer(data_path) as writer:
        writer.append(first, host_monotonic_ns=100, host_utc_ns=200)

    with _writer(data_path, mode="a") as writer:
        assert writer.next_sequence == 1
        assert writer.next_sample_index == 2
        writer.append(second, host_monotonic_ns=300, host_utc_ns=400)

    with BlockBinaryReader(data_path) as reader:
        assert len(reader) == 2
        np.testing.assert_array_equal(reader.read_block(0).data, first)
        np.testing.assert_array_equal(reader.read_block(1).data, second)


def test_reader_can_be_told_not_to_rebuild_index(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin"
    with _writer(data_path) as writer:
        writer.append(
            np.zeros((1, 2, 3), dtype=np.int16),
            host_monotonic_ns=100,
            host_utc_ns=200,
        )
    _, index_path = companion_paths(data_path)
    index_path.write_bytes(b"bad")
    with pytest.raises(IndexFormatError):
        BlockBinaryReader(data_path, auto_rebuild_index=False)


def test_writer_rejects_shape_and_sequence_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "ultrasound.bin"
    with _writer(data_path) as writer:
        with pytest.raises(ValueError, match="shape"):
            writer.append(np.zeros((2, 6), dtype=np.int16))
        with pytest.raises(ValueError, match="sequence must be 0"):
            writer.append(
                np.zeros((1, 2, 3), dtype=np.int16),
                sequence=2,
            )
