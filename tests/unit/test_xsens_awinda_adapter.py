from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from exo_collection.adapters.base import AdapterError, TrialContext
from exo_collection.adapters.imu.xsens_awinda import (
    XsensAwindaImuAdapter,
    parse_xsens_packet,
)


class Euler:
    def __init__(self, values=(10.0, 20.0, 30.0)) -> None:
        self.values = values

    def x(self): return self.values[0]
    def y(self): return self.values[1]
    def z(self): return self.values[2]


class Packet:
    def __init__(self, counter: int | None, offset: float = 0.0) -> None:
        self.counter = counter
        self.offset = offset

    def containsCalibratedData(self): return True
    def containsOrientation(self): return True
    def calibratedAcceleration(self): return (1 + self.offset, 2, 3)
    def calibratedGyroscopeData(self): return (4, 5, 6)
    def calibratedMagneticField(self): return (7, 8, 9)
    def orientationEuler(self): return Euler()
    def packetCounter(self):
        if self.counter is None:
            raise TypeError("no packet counter")
        return self.counter
    def sampleTimeFine(self):
        if self.counter is None:
            raise TypeError("no sample time")
        return self.counter * 100


class FakeAwindaBackend:
    def __init__(self, ids=("A", "B", "C")) -> None:
        self.device_ids = tuple(ids)
        self.actual_rate_hz = 200.0
        self.callback = None
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def connect(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1

    def emit(self, device_id: str, packet: Packet, host_ns: int) -> None:
        assert self.callback is not None
        self.callback(device_id, packet, host_ns)


def context() -> TrialContext:
    return TrialContext(trial_uuid=uuid4(), session_uuid=uuid4())


def running_adapter(ids=("A", "B", "C")):
    backend = FakeAwindaBackend(ids)
    adapter = XsensAwindaImuAdapter(backend=backend)
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    return adapter, backend


def test_packet_parser_has_exact_twelve_fields() -> None:
    values = parse_xsens_packet(Packet(1))
    assert values.shape == (12,)
    np.testing.assert_allclose(values, np.arange(1, 10).tolist() + [10, 20, 30])


def test_three_devices_are_grouped_in_stable_real_id_order() -> None:
    adapter, backend = running_adapter(("C", "A", "B"))
    backend.emit("B", Packet(7, 20), 30)
    backend.emit("C", Packet(7, 0), 10)
    assert adapter.get_event(timeout=0) is None
    backend.emit("A", Packet(7, 10), 20)
    event = adapter.get_event(timeout=0.1)
    assert event.data.shape == (1, 3, 12)
    assert event.data.dtype == np.float32
    assert event.data[0, :, 0].tolist() == [1.0, 11.0, 21.0]
    assert event.host_monotonic_ns == 10
    assert event.device_timestamp == 700
    assert adapter.descriptor().metadata["device_ids"] == ["C", "A", "B"]
    adapter.stop()
    adapter.close()
    assert (backend.started, backend.stopped, backend.closed) == (1, 1, 1)


def test_missing_device_does_not_create_zero_padded_sample() -> None:
    adapter, backend = running_adapter()
    backend.emit("A", Packet(3), 1)
    backend.emit("B", Packet(3), 2)
    assert adapter.get_event(timeout=0) is None
    adapter.stop()
    assert adapter.health().dropped_packets == 1
    adapter.close()


def test_counterless_fallback_groups_by_per_device_arrival_index() -> None:
    adapter, backend = running_adapter()
    backend.emit("C", Packet(None, 2), 30)
    backend.emit("A", Packet(None, 0), 10)
    backend.emit("B", Packet(None, 1), 20)
    event = adapter.get_event(timeout=0.1)
    assert event is not None
    assert event.device_timestamp is None
    adapter.stop()
    adapter.close()


def test_backend_device_count_must_be_exactly_three() -> None:
    adapter = XsensAwindaImuAdapter(backend=FakeAwindaBackend(("A", "B")))
    with pytest.raises(AdapterError, match="3"):
        adapter.connect()
    adapter.close()


def test_unexpected_device_faults_instead_of_relabeling() -> None:
    adapter, backend = running_adapter()
    backend.emit("UNKNOWN", Packet(1), 1)
    with pytest.raises(AdapterError):
        adapter.raise_if_faulted()
    adapter.close()


def test_module_import_does_not_require_xsens_sdk() -> None:
    descriptor = XsensAwindaImuAdapter(backend=FakeAwindaBackend()).descriptor()
    assert descriptor.sample_shape == (3, 12)
    assert descriptor.metadata["simulated"] is False
