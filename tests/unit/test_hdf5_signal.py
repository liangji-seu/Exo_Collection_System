from __future__ import annotations

import json
from uuid import uuid4

import h5py
import numpy as np
import pytest

from exo_collection.domain.events import SampleBatch, SyncPulseEvent
from exo_collection.writers import Hdf5SignalWriter, Hdf5SignalWriterError


def test_writer_creates_canonical_structure_and_appends_chunks(tmp_path) -> None:
    path = tmp_path / "imu.h5"
    with Hdf5SignalWriter(
        path,
        channels=("ax", "ay", "az"),
        units=("m/s2",) * 3,
        device_metadata={"device_id": "imu_1", "serial": "SIM"},
        trial_metadata={"trial_uuid": "trial-001", "subject_code": "001"},
        clock_model={"a": 1.0, "b": 0.0},
        sample_shape=(3,),
        chunk_rows=4,
        nominal_rate_hz=100.0,
    ) as writer:
        writer.append(
            np.arange(9, dtype=np.float32).reshape(3, 3),
            sample_index=10,
            device_time=1_000_000_000,
            host_monotonic_ns=2_000_000_000,
            sample_rate_hz=100.0,
        )
        writer.append(
            np.ones((2, 3), dtype=np.float32),
            sample_index=13,
            device_time=1_030_000_000,
            host_monotonic_ns=2_030_000_000,
            sample_rate_hz=100.0,
        )
        assert writer.sample_count == 5

    with h5py.File(path, "r") as handle:
        assert set(handle) == {"samples", "events", "metadata"}
        assert set(handle["samples"]) == {
            "data",
            "sample_index",
            "device_time",
            "host_monotonic_ns",
            "host_utc_ns",
            "source_sequence",
        }
        assert "discontinuities" in handle["events"]
        assert handle["samples/data"].shape == (5, 3)
        np.testing.assert_array_equal(handle["samples/sample_index"][:], [10, 11, 12, 13, 14])
        assert handle.attrs["sample_count"] == 5
        assert bool(handle.attrs["closed_cleanly"])
        channels = [value.decode() if isinstance(value, bytes) else value for value in handle["metadata/channels"][:]]
        assert channels == ["ax", "ay", "az"]
        assert json.loads(handle["metadata/device"][()]) == {
            "device_id": "imu_1",
            "serial": "SIM",
        }
        assert json.loads(handle["metadata/trial"][()]) == {
            "subject_code": "001",
            "trial_uuid": "trial-001",
        }


def test_append_batch_supports_multidevice_shape_and_optional_events(tmp_path) -> None:
    path = tmp_path / "multi_imu.h5"
    pulse = SyncPulseEvent(
        device_id="daq",
        modality="sync_pulse",
        clock_domain="daq_clock",
        pulse_id="daq:1",
        source_device="daq",
        edge_type="rising",
        sample_index=20,
        amplitude=5.0,
        detection_threshold=2.5,
        confidence=0.95,
        detector_version="test",
    )
    batch = SampleBatch(
        session_uuid=uuid4(),
        trial_uuid=uuid4(),
        device_id="imu_array",
        modality="imu",
        clock_domain="imu_clock",
        host_monotonic_ns=4_000_000_000,
        first_sample_index=20,
        sample_count=4,
        sequence_number=2,
        device_timestamp=8_000_000_000,
        sample_rate_hz=200.0,
        data=np.ones((4, 2, 12), dtype=np.float32),
    )
    with Hdf5SignalWriter(
        path,
        channels=tuple(f"c{i}" for i in range(12)),
        units=("u",) * 12,
        device_metadata={"device_id": "imu_array", "device_ids": ["a", "b"]},
        sample_shape=(2, 12),
        chunk_rows=4,
    ) as writer:
        writer.append_batch(batch, events=[pulse])

    with h5py.File(path, "r") as handle:
        assert handle["samples/data"].shape == (4, 2, 12)
        assert np.all(handle["samples/source_sequence"][:] == 2)
        assert np.all(handle["samples/host_utc_ns"][:] > 0)
        assert len(handle["events/records"]) == 1
        payload = json.loads(handle["events/records"][0])
        assert payload["event_type"] == "sync_pulse"
        assert payload["pulse_id"] == "daq:1"


def test_sample_gap_is_recorded_as_discontinuity(tmp_path) -> None:
    path = tmp_path / "encoder.h5"
    with Hdf5SignalWriter(
        path,
        channels=("left", "right"),
        units=("rad", "rad"),
        device_metadata={"device_id": "encoder", "hardware_tick_hz": 1000},
        sample_shape=(2,),
        nominal_rate_hz=200,
    ) as writer:
        writer.append(
            np.zeros((2, 2)),
            sample_index=0,
            device_time=0,
            host_monotonic_ns=1_000,
            sample_rate_hz=200,
        )
        writer.append(
            np.zeros((2, 2)),
            sample_index=5,
            device_time=25,
            host_monotonic_ns=10_001_000,
            sample_rate_hz=200,
        )
    with h5py.File(path, "r") as handle:
        event = handle["events/discontinuities"][0]
        assert event["sample_index"] == 5
        assert event["kind"].decode() == "sample_index_gap"
        details = json.loads(event["details_json"])
        assert details["missing_count"] == 3
        np.testing.assert_allclose(handle["samples/device_time"][:2], [0, 5])


def test_context_exception_marks_file_unclean(tmp_path) -> None:
    path = tmp_path / "unclean.h5"
    with pytest.raises(RuntimeError):
        with Hdf5SignalWriter(
            path,
            channels=("voltage",),
            units=("V",),
            device_metadata="daq",
        ):
            raise RuntimeError("injected writer failure")
    with h5py.File(path, "r") as handle:
        assert not bool(handle.attrs["closed_cleanly"])


def test_writer_rejects_bad_shapes_and_nonmonotonic_indices(tmp_path) -> None:
    path = tmp_path / "bad.h5"
    writer = Hdf5SignalWriter(
        path,
        channels=("x", "y"),
        units=("u", "u"),
        device_metadata="sim",
    )
    with pytest.raises(Hdf5SignalWriterError, match="data shape"):
        writer.append(
            np.ones((2, 3)),
            sample_index=0,
            host_monotonic_ns=10,
            sample_rate_hz=100,
        )
    writer.append(
        np.ones((2, 2)),
        sample_index=3,
        host_monotonic_ns=10,
        sample_rate_hz=100,
    )
    with pytest.raises(Hdf5SignalWriterError, match="increase"):
        writer.append(
            np.ones((1, 2)),
            sample_index=2,
            host_monotonic_ns=30_000_000,
            sample_rate_hz=100,
        )
    writer.close()
    with pytest.raises(Hdf5SignalWriterError, match="closed"):
        writer.flush()
