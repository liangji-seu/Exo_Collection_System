from __future__ import annotations

from uuid import uuid4

import numpy as np

from exo_collection.adapters.base import AdapterState, TrialContext
from exo_collection.adapters.ultrasound.elonxi import ElonxiUltrasoundAdapter


class FakeElonxiBackend:
    def __init__(self) -> None:
        self.resolved_device_ip = "192.0.2.10"
        self.on_ultrasound = None
        self.on_rel = None
        self.on_notification = None
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def connect(self, on_ultrasound, on_rel_data, on_notification) -> None:
        self.on_ultrasound = on_ultrasound
        self.on_rel = on_rel_data
        self.on_notification = on_notification

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1

    def emit(self, payload) -> None:
        assert self.on_ultrasound is not None
        self.on_ultrasound(payload)


def context() -> TrialContext:
    return TrialContext(trial_uuid=uuid4(), session_uuid=uuid4())


def payload(frames: int = 1, samples: int = 1000):
    return {
        channel: [np.arange(samples, dtype=np.uint16) + channel for _ in range(frames)]
        for channel in (1, 2, 3, 4)
    }


def running_adapter():
    backend = FakeElonxiBackend()
    adapter = ElonxiUltrasoundAdapter(backend=backend)
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    return adapter, backend


def test_four_channels_form_one_contiguous_frame_batch() -> None:
    adapter, backend = running_adapter()
    assert backend.on_rel is not None
    backend.on_rel(True, 42)
    backend.emit(payload(frames=2))
    event = adapter.get_event(timeout=0.1)
    assert event.data.shape == (2, 4, 1000)
    assert event.data.dtype == np.uint16
    assert event.data.flags.c_contiguous
    assert event.sequence_number == 0
    assert event.first_frame_index == 0
    assert event.device_timestamp == 42
    adapter.stop()
    adapter.close()
    assert (backend.started, backend.stopped, backend.closed) == (1, 1, 1)


def test_sequences_and_indices_are_monotonic() -> None:
    adapter, backend = running_adapter()
    backend.emit(payload(frames=2))
    backend.emit(payload(frames=1))
    first = adapter.get_event(timeout=0.1)
    second = adapter.get_event(timeout=0.1)
    assert (first.sequence_number, first.first_frame_index) == (0, 0)
    assert (second.sequence_number, second.first_frame_index) == (1, 2)
    adapter.stop()
    adapter.close()


def test_missing_channel_faults_without_fabricating_a_frame() -> None:
    adapter, backend = running_adapter()
    invalid = payload()
    del invalid[4]
    backend.emit(invalid)
    assert adapter.state is AdapterState.FAULTED
    assert adapter.get_event(timeout=0) is None
    adapter.close()


def test_wrong_waveform_length_faults() -> None:
    adapter, backend = running_adapter()
    backend.emit(payload(samples=999))
    assert adapter.state is AdapterState.FAULTED
    assert adapter.health().metrics["malformed_callbacks"] == 1
    adapter.close()


def test_module_import_does_not_require_pythonnet() -> None:
    descriptor = ElonxiUltrasoundAdapter(backend=FakeElonxiBackend()).descriptor()
    assert descriptor.sample_shape == (4, 1000)
    assert descriptor.metadata["simulated"] is False
