from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.layout import TrialLayout
from exo_collection.storage.recovery_manager import (
    discover_recoverable_trials,
    inspect_recording_directory,
    repair_recording_directory,
)
from exo_collection.writers.binary_block import BlockBinaryWriter
from exo_collection.writers.hdf5_signal import Hdf5SignalWriter


def test_interrupted_trial_is_discovered_and_safely_repaired(tmp_path) -> None:
    layout = TrialLayout.build(tmp_path, uuid4(), uuid4(), uuid4(), uuid4())
    layout.create_recording()
    ultrasound_path = layout.partial_path("raw/ultrasound.bin")
    with BlockBinaryWriter(
        ultrasound_path,
        dtype="uint16",
        sample_shape=(4,),
        metadata={"clock_domain": "sim_clock"},
    ) as writer:
        writer.append(np.arange(8, dtype=np.uint16).reshape(2, 4), host_monotonic_ns=10)
    with ultrasound_path.open("ab") as stream:
        stream.write(b"incomplete-tail")

    hdf5_path = layout.partial_path("raw/imu.h5")
    with Hdf5SignalWriter(
        hdf5_path,
        channels=("ax",),
        units=("m/s2",),
        device_metadata={"device_id": "imu_sim"},
        sample_shape=(1,),
        nominal_rate_hz=100,
    ) as writer:
        writer.append(np.ones((3, 1)), host_monotonic_ns=100, sample_rate_hz=100)

    discovered = discover_recoverable_trials(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].state is TrialState.RECOVERABLE
    assert discovered[0].recoverable
    assert discovered[0].ultrasound is not None
    assert discovered[0].ultrasound.tail_recoverable
    assert discovered[0].hdf5_files[0].closed_cleanly

    repaired = repair_recording_directory(layout.recording_directory)
    assert repaired.repair_log_path is not None and repaired.repair_log_path.is_file()
    assert repaired.ultrasound is not None and repaired.ultrasound.is_clean
    assert layout.recording_directory.exists()
    assert not layout.final_directory.exists()
    assert inspect_recording_directory(layout.recording_directory).state is TrialState.RECOVERABLE


def test_repair_refuses_to_touch_an_active_recording(tmp_path) -> None:
    layout = TrialLayout.build(tmp_path, uuid4(), uuid4(), uuid4(), uuid4())
    layout.create_recording()
    ultrasound_path = layout.partial_path("raw/ultrasound.bin")
    with BlockBinaryWriter(
        ultrasound_path,
        dtype="uint16",
        sample_shape=(4,),
        metadata={"clock_domain": "sim_clock"},
    ) as writer:
        writer.append(np.arange(8, dtype=np.uint16).reshape(2, 4), host_monotonic_ns=10)
    with ultrasound_path.open("ab") as stream:
        stream.write(b"incomplete-tail")
    original = ultrasound_path.read_bytes()

    with AcquisitionLock(tmp_path, layout.trial_uuid):
        with pytest.raises(FileExistsError, match="collector lock"):
            repair_recording_directory(layout.recording_directory)

    assert ultrasound_path.read_bytes() == original
    assert not tuple((layout.recording_directory / "reports").glob("recovery-*.json*"))
