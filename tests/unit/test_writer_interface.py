from __future__ import annotations

import numpy as np
import pytest

from exo_collection.writers import (
    BlockBinaryWriter,
    Hdf5SignalWriter,
    Writer,
    built_in_writer_registry,
    resolve_writer_type,
)


def test_built_in_writers_share_the_lifecycle_protocol(tmp_path) -> None:
    binary = BlockBinaryWriter(
        tmp_path / "ultrasound.bin.partial",
        dtype="uint16",
        sample_shape=(2,),
        metadata={"clock_domain": "sim"},
    )
    hdf5 = Hdf5SignalWriter(
        tmp_path / "imu.h5.partial",
        channels=("ax",),
        units=("m/s2",),
        device_metadata="imu",
        sample_shape=(1,),
    )
    try:
        assert isinstance(binary, Writer)
        assert isinstance(hdf5, Writer)
        binary.write(np.zeros((1, 2), dtype=np.uint16))
        hdf5.append(np.zeros((1, 1)), host_monotonic_ns=1)
    finally:
        binary.close()
        hdf5.close()


def test_writer_registry_is_static_and_rejects_unknown_types() -> None:
    registry = built_in_writer_registry()
    assert registry == {
        "block_binary": BlockBinaryWriter,
        "hdf5_signal": Hdf5SignalWriter,
    }
    registry.clear()
    assert resolve_writer_type("block_binary") is BlockBinaryWriter
    with pytest.raises(KeyError, match="unknown Writer type"):
        resolve_writer_type("vendor_magic")
