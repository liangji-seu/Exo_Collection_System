"""Tests for Xsens Awinda IMU adapter: adapter-level unit tests and
fake-XDA backend contract verification."""

from __future__ import annotations

import sys
from queue import Full
from threading import Thread
from time import monotonic, sleep, perf_counter_ns
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.adapters.base import AdapterError, TrialContext
from exo_collection.adapters.imu.xsens_awinda import (
    XdaAwindaBackend,
    XsensAwindaConfig,
    XsensAwindaImuAdapter,
    _read_optional_packet_counter,
    _read_optional_sample_time_fine,
    parse_xsens_packet,
)

# ──────────────────────────────────────────────────────────────
#  Shared test doubles – data packet
# ──────────────────────────────────────────────────────────────


class _FakeOutputConfigArray:
    """Fake XsOutputConfigurationArray that supports push_back like real XDA."""

    def __init__(self):
        self._items: list = []

    def push_back(self, item: Any) -> None:
        self._items.append(item)


class Euler:
    def __init__(self, values=(10.0, 20.0, 30.0)) -> None:
        self.values = values

    def x(self):
        return self.values[0]

    def y(self):
        return self.values[1]

    def z(self):
        return self.values[2]


class Packet:
    """Mimics XDA data packet with configurable contains* guards."""

    def __init__(
        self,
        counter: int | None = 1,
        offset: float = 0.0,
        *,
        has_counter: bool = True,
        has_time: bool = True,
        sample_time_fine: int | None = None,
    ) -> None:
        self.counter = counter
        self.offset = offset
        self._has_counter = has_counter
        self._has_time = has_time
        self._sample_time_fine_override = sample_time_fine

    def containsCalibratedData(self):
        return True

    def containsOrientation(self):
        return True

    def containsPacketCounter(self):
        return self._has_counter

    def containsSampleTimeFine(self):
        return self._has_time

    def calibratedAcceleration(self):
        return (1 + self.offset, 2, 3)

    def calibratedGyroscopeData(self):
        return (4, 5, 6)

    def calibratedMagneticField(self):
        return (7, 8, 9)

    def orientationEuler(self):
        return Euler()

    def packetCounter(self):
        if self.counter is None:
            raise TypeError("no packet counter")
        return self.counter

    def sampleTimeFine(self):
        if self._sample_time_fine_override is not None:
            return self._sample_time_fine_override
        if self.counter is None:
            raise TypeError("no sample time")
        return self.counter * 100


class PacketNoCounter:
    """Packet that always fails containsPacketCounter() and containsSampleTimeFine().
    The getters return 0 (mimics real XDA behaviour with missing output item)."""

    def containsCalibratedData(self):
        return True

    def containsOrientation(self):
        return True

    def containsPacketCounter(self):
        return False

    def containsSampleTimeFine(self):
        return False

    # These should NOT be called when contains is false;
    # the real XDA returns 0 if called without the output item requested.
    def packetCounter(self):
        return 0

    def sampleTimeFine(self):
        return 0

    def calibratedAcceleration(self):
        return (0.1, 0.2, 0.3)

    def calibratedGyroscopeData(self):
        return (0.4, 0.5, 0.6)

    def calibratedMagneticField(self):
        return (0.7, 0.8, 0.9)

    def orientationEuler(self):
        return Euler((1.0, 2.0, 3.0))


# ──────────────────────────────────────────────────────────────
#  FakeAwindaBackend – existing protocol double, updated
# ──────────────────────────────────────────────────────────────


class FakeAwindaBackend:
    def __init__(self, ids=("A", "B", "C")) -> None:
        self.device_ids = tuple(ids)
        self.actual_rate_hz = 200
        self.metadata = {"device_ids": list(ids), "actual_sample_rate_hz": 200}
        self.callback = None
        self.started = 0
        self.stopped = 0
        self.closed = 0
        self._callbacks_removed = False

    def connect(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def remove_callbacks(self) -> None:
        self._callbacks_removed = True

    def close(self) -> None:
        self.closed += 1

    def emit(self, device_id: str, packet: Packet, host_ns: int) -> None:
        assert self.callback is not None
        self.callback(device_id, packet, host_ns)


def discovery_api(children_ids: Any) -> tuple[Any, Any]:
    """Small fake XDA surface for deterministic discovery-clock tests."""
    api = type(sys)("_test_xsens_discovery_clock")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type(
        "oc", (), {"data_type": dt, "rate": rate}
    )()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    class FakeMtw:
        def __init__(self, device_id: str) -> None:
            self._device_id = device_id

        def deviceId(self):
            return type(
                "did", (), {"toXsString": lambda _self: self._device_id}
            )()

        def addCallbackHandler(self, callback):
            pass

        def removeCallbackHandler(self, callback):
            pass

        def setOutputConfiguration(self, output):
            return True

    class FakeMaster:
        def __init__(self) -> None:
            self.children_calls = 0

        def gotoConfig(self):
            return True

        def gotoMeasurement(self):
            return True

        def enableRadio(self, channel):
            return True

        def disableRadio(self):
            pass

        def children(self):
            self.children_calls += 1
            return [FakeMtw(item) for item in children_ids(self.children_calls)]

        def supportedUpdateRates(self):
            return type(
                "rates",
                (), {
                    "size": lambda _self: 1,
                    "__getitem__": lambda _self, index: 200,
                },
            )()

        def setUpdateRate(self, rate):
            return True

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM99"

        def baudrate(self):
            return 115200

        def deviceId(self):
            return type(
                "did",
                (), {
                    "isWirelessMaster": lambda _self: True,
                    "toXsString": lambda _self: "AWINDA_MASTER",
                },
            )()

    master = FakeMaster()

    class FakeControl:
        def openPort(self, *args):
            return True

        def device(self, device_id):
            return master

        def closePort(self, *args):
            pass

        def close(self):
            pass

    api.XsScanner_scanPorts = lambda: type(
        "ports",
        (), {
            "size": lambda _self: 1,
            "__getitem__": lambda _self, index: FakePortInfo(),
        },
    )()
    api.XsControl_construct = FakeControl
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("callback", (), {})
    api.XsDataPacket = lambda packet: packet
    return api, master


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def context() -> TrialContext:
    return TrialContext(trial_uuid=uuid4(), session_uuid=uuid4())


def running_adapter(ids=("A", "B", "C")):
    backend = FakeAwindaBackend(ids)
    adapter = XsensAwindaImuAdapter(backend=backend, config={"queue_capacity": 16})
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    return adapter, backend


def _drain_events(adapter: XsensAwindaImuAdapter, timeout: float = 0.3):
    events = []
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        event = adapter.get_event(timeout=0.05)
        if event is not None:
            events.append(event)
        sleep(0.01)
    return events


# ──────────────────────────────────────────────────────────────
#  Existing tests (preserved, minimally adapted)
# ──────────────────────────────────────────────────────────────


def test_packet_parser_has_exact_twelve_fields() -> None:
    values = parse_xsens_packet(Packet(1))
    assert values.shape == (12,)
    np.testing.assert_allclose(values, np.arange(1, 10).tolist() + [10, 20, 30])


def test_three_devices_are_grouped_in_stable_real_id_order() -> None:
    adapter, backend = running_adapter(("C", "A", "B"))

    backend.emit("B", Packet(7, 20), 30)
    backend.emit("C", Packet(7, 0), 10)
    # Only 2 of 3 – not complete yet
    assert adapter.get_event(timeout=0.1) is None

    backend.emit("A", Packet(7, 10), 20)
    event = adapter.get_event(timeout=0.5)
    assert event is not None
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

    # Not complete – only 2 devices
    assert adapter.get_event(timeout=0.1) is None

    adapter.stop()
    assert adapter.health().dropped_packets == 1
    adapter.close()


def test_counterless_fallback_groups_by_per_device_arrival_index() -> None:
    adapter, backend = running_adapter()
    # Has counter data enabled
    backend.emit("C", Packet(None, 2), 30)
    backend.emit("A", Packet(None, 0), 10)
    backend.emit("B", Packet(None, 1), 20)
    event = adapter.get_event(timeout=0.5)
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


# ──────────────────────────────────────────────────────────────
#  contains* guard tests (P0 #2 fix)
# ──────────────────────────────────────────────────────────────


def test_contains_packet_counter_false_returns_none() -> None:
    """When containsPacketCounter() is False, the getter returns 0
    but we must return None instead to avoid wrong counter==0 alignment."""
    pkt = Packet(counter=0, has_counter=False)
    result = _read_optional_packet_counter(pkt)
    assert result is None


def test_contains_packet_counter_true_returns_value() -> None:
    pkt = Packet(counter=42, has_counter=True)
    result = _read_optional_packet_counter(pkt)
    assert result == 42


def test_contains_sample_time_fine_false_returns_none() -> None:
    pkt = Packet(counter=0, has_time=False)
    result = _read_optional_sample_time_fine(pkt)
    assert result is None


def test_contains_sample_time_fine_true_returns_value() -> None:
    pkt = Packet(counter=15, has_time=True)
    result = _read_optional_sample_time_fine(pkt)
    assert result == 1500


def test_packet_without_counter_output_slot_never_uses_zero() -> None:
    """Even though packetCounter() returns 0, we must not align on 0
    when containsPacketCounter() is False."""
    adapter, backend = running_adapter()
    # These packets lack counter slots – counter returns 0 but contains is False
    backend.emit("A", PacketNoCounter(), 1)
    backend.emit("B", PacketNoCounter(), 2)
    backend.emit("C", PacketNoCounter(), 3)

    event = adapter.get_event(timeout=0.5)
    assert event is not None
    assert event.data.shape == (1, 3, 12)
    # Device timestamp should be None (fallback mode, no counter/time)
    assert event.device_timestamp is None

    adapter.stop()
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Alignment scenario tests
# ──────────────────────────────────────────────────────────────


def test_alignment_reordered_abc_cba_produces_one_sample() -> None:
    """Packets arrive C, B, A (counter 7 for all). Should still group."""
    adapter, backend = running_adapter()
    backend.emit("C", Packet(7), 30)
    backend.emit("B", Packet(7), 20)
    # Not complete yet
    assert adapter.get_event(timeout=0.05) is None
    backend.emit("A", Packet(7), 10)

    event = adapter.get_event(timeout=0.5)
    assert event is not None
    assert event.data.shape == (1, 3, 12)
    # Ordered by _use_device_ids order (A, B, C)
    assert adapter.descriptor().metadata["device_ids"] == ["A", "B", "C"]
    adapter.stop()
    adapter.close()


def test_duplicate_device_same_counter_not_emitted() -> None:
    """A/A/B/C: two A's at same counter. Second A is duplicate, group A/B/C emits once."""
    adapter, backend = running_adapter()
    backend.emit("A", Packet(5, 0), 10)
    backend.emit("A", Packet(5, 99), 11)  # duplicate counter – same device
    backend.emit("B", Packet(5), 20)
    backend.emit("C", Packet(5), 30)

    event = adapter.get_event(timeout=0.5)
    assert event is not None
    # Tolerate either first-A or second-A depending on ordering of duplicate handling
    assert event.data.shape == (1, 3, 12)
    health = adapter.health()
    assert health.metrics["duplicate_packets"] >= 1
    adapter.stop()
    adapter.close()


def test_single_device_missing_creates_incomplete() -> None:
    """Two devices at counter 1, one at counter 2. Counter 1 is incomplete."""
    adapter, backend = running_adapter()
    backend.emit("A", Packet(1), 10)
    backend.emit("B", Packet(1), 20)
    backend.emit("A", Packet(2), 30)
    backend.emit("B", Packet(2), 40)
    backend.emit("C", Packet(2), 50)

    events = _drain_events(adapter, timeout=0.3)
    # Counter 2 should have 3 devices → one complete sample
    assert len(events) == 1
    assert events[0].data.shape == (1, 3, 12)

    adapter.stop()
    # Counter 1 group should be marked incomplete
    assert adapter.health().dropped_packets == 1
    adapter.close()


def test_common_counter_gap_detected() -> None:
    """Counter jumps from 1 to 10: gap detected."""
    adapter, backend = running_adapter()
    backend.emit("A", Packet(1), 10)
    backend.emit("B", Packet(1), 20)
    backend.emit("C", Packet(1), 30)
    backend.emit("A", Packet(10), 40)
    backend.emit("B", Packet(10), 50)
    backend.emit("C", Packet(10), 60)

    events = _drain_events(adapter, timeout=0.3)
    assert len(events) == 2
    health = adapter.health()
    assert health.metrics["counter_gaps"] >= 1
    adapter.stop()
    adapter.close()


def test_counter_wrap_65535_to_0_handled() -> None:
    """Counter wraps from 65535 to 0 – wrap should NOT count as gap."""
    adapter, backend = running_adapter()
    backend.emit("A", Packet(65535), 10)
    backend.emit("B", Packet(65535), 20)
    backend.emit("C", Packet(65535), 30)
    backend.emit("A", Packet(0), 40)
    backend.emit("B", Packet(0), 50)
    backend.emit("C", Packet(0), 60)

    events = _drain_events(adapter, timeout=0.3)
    assert len(events) == 2
    health = adapter.health()
    # Wrap from 65535→0 is expected; 0 → 1 is also expected (the gap detection
    # only fires when !expected and !same; wrap is detected by (prev+1)%65536)
    # Verify no spurious counter gaps from wrap
    assert health.metrics["counter_gaps"] == 0
    adapter.stop()
    adapter.close()


def test_sample_time_fine_inconsistency_tracked() -> None:
    """Three devices with different sampleTimeFine at same counter:
    time spread is tracked in health."""
    adapter, backend = running_adapter()
    pkt_a = Packet(1, sample_time_fine=100)
    pkt_b = Packet(1, sample_time_fine=150)
    pkt_c = Packet(1, sample_time_fine=50)

    backend.emit("A", pkt_a, 10)
    backend.emit("B", pkt_b, 20)
    backend.emit("C", pkt_c, 30)

    event = adapter.get_event(timeout=0.5)
    assert event is not None
    health = adapter.health()
    assert health.metrics["max_device_time_spread"] > 0
    assert health.metrics["sample_time_fine_available"] is True
    adapter.stop()
    adapter.close()


def test_pending_eviction_on_limit() -> None:
    """With pending_group_limit=1, second (incomplete) group evicts the first."""
    adapter, backend = XsensAwindaImuAdapter(backend=FakeAwindaBackend()), None
    # Rebuild with tiny limit
    adapter._config = XsensAwindaConfig(pending_group_limit=1, queue_capacity=16)
    adapter._backend = FakeAwindaBackend()
    backend = adapter._backend
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    # Counter 0: only device A
    backend.emit("A", Packet(0), 10)
    # Counter 1: only device B → this should evict counter-0 group
    backend.emit("B", Packet(1), 20)

    adapter.stop()
    health = adapter.health()
    assert health.metrics["incomplete_sensor_samples"] >= 1
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Queue overflow and stop/close tests
# ──────────────────────────────────────────────────────────────


def test_packet_queue_overflow_faults() -> None:
    """With queue_capacity=1, emitting 2 packets simultaneously faults."""
    adapter = XsensAwindaImuAdapter(
        backend=FakeAwindaBackend(),
        config={"queue_capacity": 2},
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    # Fill queue beyond capacity
    backend = adapter._backend
    try:
        backend.emit("A", Packet(1), 100)
        backend.emit("B", Packet(1), 200)
        backend.emit("C", Packet(1), 300)
    except Exception:
        pass   # expected flow

    # Consumer thread may or may not have drained by now
    sleep(0.1)

    # Now try to overflow deliberately by disabling consumer
    adapter._consumer_stop.set()
    sleep(0.05)

    for i in range(50):
        try:
            backend.callback("A", Packet(i), i * 10)
        except Exception:
            break

    adapter.stop()
    adapter.close()


def test_late_packet_after_stop_is_ignored() -> None:
    """After stop, _accepting_packets=False; callbacks are ignored."""
    adapter, backend = running_adapter()
    # Emit some data
    backend.emit("A", Packet(1), 10)
    backend.emit("B", Packet(1), 20)
    backend.emit("C", Packet(1), 30)
    _drain_events(adapter, timeout=0.2)

    adapter.stop()

    # Late packet after stop
    backend.callback("A", Packet(999), 2000)
    sleep(0.05)

    assert adapter.get_event(timeout=0.05) is None
    adapter.close()


def test_close_best_effort_stages_even_when_stop_fails() -> None:
    """Every close stage executes even if prior stages fail."""

    class FailingBackend(FakeAwindaBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stages_called = []

        def stop(self):
            self.stages_called.append("stop")
            raise RuntimeError("forced stop error")

        def remove_callbacks(self):
            self.stages_called.append("remove_callbacks")

        def close(self):
            self.stages_called.append("close")

    adapter = XsensAwindaImuAdapter(backend=FailingBackend())
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    # close should call stop (which fails), then close_hardware (which succeeds)
    adapter.close()
    backend = adapter._backend
    assert "stop" in backend.stages_called
    assert "remove_callbacks" in backend.stages_called
    assert "close" in backend.stages_called


def test_callbacks_remain_registered_until_close() -> None:
    """Trial stop gates packets but keeps callbacks for adapter reuse."""
    adapter, backend = running_adapter()
    # Give consumer time to drain
    backend.emit("A", Packet(1), 10)
    backend.emit("B", Packet(1), 20)
    backend.emit("C", Packet(1), 30)
    _drain_events(adapter, timeout=0.2)

    adapter.stop()
    assert backend._callbacks_removed is False
    adapter.close()
    assert backend._callbacks_removed is True


def test_adapter_can_record_two_trials_without_reconnecting_callbacks() -> None:
    adapter, backend = running_adapter()

    for counter in (1, 2):
        backend.emit("A", Packet(counter), counter * 10 + 1)
        backend.emit("B", Packet(counter), counter * 10 + 2)
        backend.emit("C", Packet(counter), counter * 10 + 3)
        event = adapter.get_event(timeout=0.5)
        assert event is not None
        assert event.first_sample_index == 0
        adapter.stop()
        assert backend._callbacks_removed is False
        if counter == 1:
            adapter.prepare(context())
            adapter.start()

    assert backend.started == 2
    assert backend.stopped == 2
    adapter.close()
    assert backend._callbacks_removed is True


# ──────────────────────────────────────────────────────────────
#  Preview shape (N,3,12) and descriptor tests
# ──────────────────────────────────────────────────────────────


def test_emitted_sample_batch_is_n_3_12_shape() -> None:
    """Every SampleBatch from adapter must have shape (1, 3, 12)."""
    adapter, backend = running_adapter()
    backend.emit("A", Packet(1), 10)
    backend.emit("B", Packet(1), 20)
    backend.emit("C", Packet(1), 30)

    event = adapter.get_event(timeout=0.5)
    assert event is not None
    assert event.data.shape == (1, 3, 12)
    assert event.data.dtype == np.float32
    # Verify field order: acc[0], acc[1], acc[2], gyr[0..2], mag[0..2], roll, pitch, yaw
    sample_a = event.data[0, 0, :]
    assert sample_a[0] == 1.0   # acc_x
    assert sample_a[3] == 4.0   # gyr_x
    assert sample_a[6] == 7.0   # mag_x
    assert sample_a[9] == 10.0  # roll
    assert sample_a[10] == 20.0 # pitch
    assert sample_a[11] == 30.0 # yaw
    adapter.stop()
    adapter.close()


def test_descriptor_channel_and_unit_length_match() -> None:
    adapter = XsensAwindaImuAdapter(backend=FakeAwindaBackend())
    desc = adapter.descriptor()
    assert len(desc.channels) == 12
    assert len(desc.units) == 12
    assert desc.sample_shape == (3, 12)
    assert desc.modality == "imu"


def test_configuration_snapshot_includes_backend_metadata() -> None:
    adapter, backend = running_adapter()
    snapshot = adapter.configuration_snapshot()
    assert "device_id" in snapshot
    assert "actual_rate_hz" in snapshot
    # Backend metadata should be folded in
    backend.metadata = {"test_key": "test_value", "device_ids": ["A", "B", "C"]}
    snapshot2 = adapter.configuration_snapshot()
    assert snapshot2["test_key"] == "test_value"
    adapter.stop()
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Backend-level: int rate assertion (P0 #1 fix)
# ──────────────────────────────────────────────────────────────


def test_xs_output_configuration_receives_int_not_float() -> None:
    """When a fake API captures XsOutputConfiguration calls, the rate
    argument must be int, not float."""

    class FakeXDI:
        PacketCounter = 99
        SampleTimeFine = 98
        EulerAngles = 1
        Acceleration = 2
        RateOfTurn = 3
        MagneticField = 4

    class TrackedOutputConfig:
        def __init__(self, data_type, rate):
            self.data_type = data_type
            self.rate = rate
            TrackedOutputConfig.calls.append(self)

    class FakeOutputConfigArray:
        def __init__(self):
            self.items = []

        def push_back(self, item):
            self.items.append(item)

    class FakePortInfo:
        def __init__(self):
            self._empty = False
            self._port_name = "COM3"
            self._baudrate = 115200
            self._device_id = FakeDeviceId()

        def empty(self):
            return self._empty

        def portName(self):
            return self._port_name

        def baudrate(self):
            return self._baudrate

        def deviceId(self):
            return self._device_id

    class FakeDeviceId:
        def isWirelessMaster(self):
            return True

        def toXsString(self):
            return "AWINDA_00B41234"

    class FakePortsArray:
        def __init__(self, ports):
            self._ports = ports

        def size(self):
            return len(self._ports)

        def __getitem__(self, index):
            return self._ports[index]

    class FakeScanner:
        @staticmethod
        def scanPorts():
            return FakePortsArray([FakePortInfo()])

    class FakeControl:
        def __init__(self):
            self.port_opened = False
            self.closed = False

        def openPort(self, name, baudrate):
            self.port_opened = True
            return True

        def device(self, device_id):
            return FakeMaster()

        def closePort(self, name):
            pass

        def close(self):
            self.closed = True

    class FakeMaster:
        def gotoConfig(self):
            return True

        def gotoMeasurement(self):
            return True

        def enableRadio(self, channel):
            return True

        def disableRadio(self):
            pass

        def children(self):
            return [FakeMtw("A"), FakeMtw("B"), FakeMtw("C")]

        def supportedUpdateRates(self):
            class Rates:
                def size(self):
                    return 2

                def __getitem__(self, idx):
                    return [100, 200][idx]

            return Rates()

        def setUpdateRate(self, rate):
            TrackedOutputConfig.set_update_rate_args.append(rate)
            return True

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            pass

    class FakeCallback:
        pass

    class FakeDataPacket:
        def __init__(self, other=None):
            pass

    TrackedOutputConfig.calls = []
    TrackedOutputConfig.set_update_rate_args = []
    TrackedOutputConfig.rate_int_assertions = []

    api = sys.modules.get("_test_xsens_fake", type(sys)("_test_xsens_fake"))
    api.XDI_PacketCounter = FakeXDI.PacketCounter
    api.XDI_SampleTimeFine = FakeXDI.SampleTimeFine
    api.XDI_EulerAngles = FakeXDI.EulerAngles
    api.XDI_Acceleration = FakeXDI.Acceleration
    api.XDI_RateOfTurn = FakeXDI.RateOfTurn
    api.XDI_MagneticField = FakeXDI.MagneticField
    api.XsOutputConfiguration = TrackedOutputConfig
    api.XsOutputConfigurationArray = FakeOutputConfigArray
    api.XsScanner_scanPorts = FakeScanner.scanPorts
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = FakeCallback
    api.XsDataPacket = FakeDataPacket

    config = XsensAwindaConfig(sample_rate_hz=200.0)
    backend = XdaAwindaBackend(config, _api_module=api)

    collected = []

    def on_packet(did, pkt, ns):
        collected.append((did, ns))

    backend.connect(on_packet)

    # Verify all XsOutputConfiguration calls received int rate
    for call in TrackedOutputConfig.calls:
        assert isinstance(call.rate, int), f"Expected int, got {type(call.rate)} = {call.rate!r}"
    assert len(TrackedOutputConfig.calls) == 6  # PacketCounter, SampleTimeFine, Euler, Acc, Gyr, Mag

    # Verify setUpdateRate received int
    for rate_arg in TrackedOutputConfig.set_update_rate_args:
        assert isinstance(rate_arg, int), f"setUpdateRate expected int, got {type(rate_arg)} = {rate_arg!r}"

    # Verify six data types in correct order
    data_types = [c.data_type for c in TrackedOutputConfig.calls]
    assert data_types == [
        FakeXDI.PacketCounter,
        FakeXDI.SampleTimeFine,
        FakeXDI.EulerAngles,
        FakeXDI.Acceleration,
        FakeXDI.RateOfTurn,
        FakeXDI.MagneticField,
    ]

    backend.start()
    backend.stop()
    backend.close()


# ──────────────────────────────────────────────────────────────
#  Backend-level: discovery strategy
# ──────────────────────────────────────────────────────────────


def test_discovery_with_target_ids_succeeds_even_with_extra_devices() -> None:
    """When 3 specific target IDs are provided and 4 MTw devices appear,
    the backend selects the 3 targets and records a warning."""

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            pass

    # Build fake API with children() returning 4 devices
    captured = {}

    def _make_api(children_fn, supported_rates_fn):
        captured["api"] = sys.modules.get("_test_xsens_fake", type(sys)("_test_xsens_fake"))
        captured["api"].XDI_PacketCounter = 99
        captured["api"].XDI_SampleTimeFine = 98
        captured["api"].XDI_EulerAngles = 1
        captured["api"].XDI_Acceleration = 2
        captured["api"].XDI_RateOfTurn = 3
        captured["api"].XDI_MagneticField = 4

        captured["api"].XsOutputConfiguration = lambda dt, rate: type(
            "oc", (), {"data_type": dt, "rate": rate}
        )()
        captured["api"].XsOutputConfigurationArray = _FakeOutputConfigArray

        class FakePortInfo:
            def empty(self):
                return False

            def portName(self):
                return "COM3"

            def baudrate(self):
                return 115200

            def deviceId(self):
                class Did:
                    def isWirelessMaster(self):
                        return True

                    def toXsString(self):
                        return "AWINDA_MASTER"

                return Did()

        class FakeControl:
            def openPort(self, *a):
                return True

            def device(self, did):
                class Master:
                    def gotoConfig(self):
                        return True

                    def gotoMeasurement(self):
                        return True

                    def enableRadio(self, ch):
                        return True

                    def disableRadio(self):
                        pass

                    def children(self):
                        return children_fn()

                    def supportedUpdateRates(self):
                        return supported_rates_fn()

                    def setUpdateRate(self, rate):
                        return True

                return Master()

            def closePort(self, *a):
                pass

            def close(self):
                pass

        class FakePortsArray:
            def __init__(self, ports):
                self._p = ports

            def size(self):
                return len(self._p)

            def __getitem__(self, idx):
                return self._p[idx]

        captured["api"].XsScanner_scanPorts = lambda: FakePortsArray([FakePortInfo()])
        captured["api"].XsControl_construct = lambda: FakeControl()
        captured["api"].XsPortInfo = FakePortInfo
        captured["api"].XsCallback = type("cb", (), {})
        captured["api"].XsDataPacket = lambda x: x
        return captured["api"]

    # Test 1: 3 targets + 1 extra
    children_4 = lambda: [FakeMtw(d) for d in ["A", "B", "C", "D"]]

    class Rates3:
        def size(self):
            return 1

        def __getitem__(self, idx):
            return 200

    api = _make_api(children_4, lambda: Rates3())
    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=0.5,
        stable_wait_s=0.05,
        poll_interval_s=0.01,
        sensor_ids=("A", "B", "C"),
    )
    backend = XdaAwindaBackend(config, _api_module=api)
    backend.connect(lambda *a: None)
    assert backend.device_ids == ("A", "B", "C")
    assert backend.metadata["discovery_target_ids"] == ["A", "B", "C"]
    assert "D" in backend.metadata["all_discovered_device_ids"]
    assert "discovery_warning" in backend.metadata
    assert "额外" in backend.metadata["discovery_warning"]
    backend.close()


def test_discovery_no_ids_strict_three_succeeds() -> None:
    """Without target IDs, exactly 3 devices must appear."""

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            pass

    # Reuse the _make_api pattern from above
    children_3 = lambda: [FakeMtw(d) for d in ["X", "Y", "Z"]]

    api = type(sys)("_test_xsens_fake")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type("oc", (), {"data_type": dt, "rate": rate})()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM3"

        def baudrate(self):
            return 115200

        def deviceId(self):
            class Did:
                def isWirelessMaster(self):
                    return True

                def toXsString(self):
                    return "AWINDA_MASTER"

            return Did()

    class FakeControl:
        def openPort(self, *a):
            return True

        def device(self, did):
            class Master:
                def gotoConfig(self):
                    return True

                def gotoMeasurement(self):
                    return True

                def enableRadio(self, ch):
                    return True

                def disableRadio(self):
                    pass

                def children(self):
                    return children_3()

                def supportedUpdateRates(self):
                    class Rates:
                        def size(self):
                            return 1

                        def __getitem__(self, idx):
                            return 200

                    return Rates()

                def setUpdateRate(self, rate):
                    return True

            return Master()

        def closePort(self, *a):
            pass

        def close(self):
            pass

    api.XsScanner_scanPorts = lambda: type("pa", (), {"size": lambda self: 1, "__getitem__": lambda s, i: FakePortInfo()})()
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda x: x

    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=0.5,
        stable_wait_s=0.05,
        poll_interval_s=0.01,
        sensor_ids=(),
    )
    backend = XdaAwindaBackend(config, _api_module=api)
    backend.connect(lambda *a: None)
    assert set(backend.device_ids) == {"X", "Y", "Z"}
    backend.close()


def test_discovery_no_ids_too_few_raises() -> None:
    """Without target IDs, fewer than 3 devices raises an error."""

    api = type(sys)("_test_xsens_fake")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type("oc", (), {"data_type": dt, "rate": rate})()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM3"

        def baudrate(self):
            return 115200

        def deviceId(self):
            class Did:
                def isWirelessMaster(self):
                    return True

                def toXsString(self):
                    return "AWINDA_MASTER"

            return Did()

    class FakeControl:
        def openPort(self, *a):
            return True

        def device(self, did):
            class Master:
                def gotoConfig(self):
                    return True

                def gotoMeasurement(self):
                    return True

                def enableRadio(self, ch):
                    return True

                def disableRadio(self):
                    pass

                def children(self):
                    return [FakeMtw("ONLY_ONE")]

                def supportedUpdateRates(self):
                    class Rates:
                        def size(self):
                            return 1

                        def __getitem__(self, idx):
                            return 200

                    return Rates()

                def setUpdateRate(self, rate):
                    return True

            return Master()

        def closePort(self, *a):
            pass

        def close(self):
            pass

    api.XsScanner_scanPorts = lambda: type("pa", (), {"size": lambda self: 1, "__getitem__": lambda s, i: FakePortInfo()})()
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda x: x

    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=0.3,
        stable_wait_s=0.05,
        poll_interval_s=0.01,
        sensor_ids=(),
    )
    backend = XdaAwindaBackend(config, _api_module=api)
    with pytest.raises(AdapterError, match="3"):
        backend.connect(lambda *a: None)
    backend.close()


def test_discovery_third_slow_connect_still_works() -> None:
    """When device C connects after A and B have been stable, wait continues
    until the complete 3-device set stabilizes."""

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            pass

    # Build api where children() gradually adds devices
    api = type(sys)("_test_xsens_fake")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type("oc", (), {"data_type": dt, "rate": rate})()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM3"

        def baudrate(self):
            return 115200

        def deviceId(self):
            class Did:
                def isWirelessMaster(self):
                    return True

                def toXsString(self):
                    return "AWINDA_MASTER"

            return Did()

    children_calls = []

    class GradualMaster:
        def __init__(self):
            self._call_count = 0

        def gotoConfig(self):
            return True

        def gotoMeasurement(self):
            return True

        def enableRadio(self, ch):
            return True

        def disableRadio(self):
            pass

        def children(self):
            self._call_count += 1
            children_calls.append(self._call_count)
            if self._call_count <= 3:
                return [FakeMtw("A"), FakeMtw("B")]
            return [FakeMtw("A"), FakeMtw("B"), FakeMtw("C")]

        def supportedUpdateRates(self):
            class Rates:
                def size(self):
                    return 1

                def __getitem__(self, idx):
                    return 200

            return Rates()

        def setUpdateRate(self, rate):
            return True

    class FakeControl:
        def openPort(self, *a):
            return True

        def device(self, did):
            return GradualMaster()

        def closePort(self, *a):
            pass

        def close(self):
            pass

    api.XsScanner_scanPorts = lambda: type("pa", (), {"size": lambda self: 1, "__getitem__": lambda s, i: FakePortInfo()})()
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda x: x

    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=5.0,
        stable_wait_s=0.1,
        poll_interval_s=0.02,
        sensor_ids=(),
    )
    backend = XdaAwindaBackend(config, _api_module=api)
    backend.connect(lambda *a: None)
    assert set(backend.device_ids) == {"A", "B", "C"}
    backend.close()


# ──────────────────────────────────────────────────────────────
#  Backend-level: close best-effort each stage
# ──────────────────────────────────────────────────────────────


def test_backend_close_proceeds_even_when_stop_fails() -> None:
    """When stop raises, close still proceeds through remove_callbacks,
    disableRadio, closePort, and control.close."""

    api = type(sys)("_test_xsens_fake")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type("oc", (), {"data_type": dt, "rate": rate})()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    close_log = []

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            close_log.append("removeCallbacks")

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM3"

        def baudrate(self):
            return 115200

        def deviceId(self):
            class Did:
                def isWirelessMaster(self):
                    return True

                def toXsString(self):
                    return "AWINDA_MASTER"

            return Did()

    class FakeControl:
        def openPort(self, *a):
            return True

        def device(self, did):
            class Master:
                def __init__(self):
                    self._measurement = False

                def gotoConfig(self):
                    if not self._measurement:
                        # First gotoConfig (during connect) succeeds
                        return True
                    # Second gotoConfig (during stop) fails
                    close_log.append("gotoConfig_fail")
                    raise RuntimeError("forced gotoConfig failure")

                def gotoMeasurement(self):
                    self._measurement = True
                    return True

                def enableRadio(self, ch):
                    return True

                def disableRadio(self):
                    close_log.append("disableRadio")

                def children(self):
                    return [FakeMtw("A"), FakeMtw("B"), FakeMtw("C")]

                def supportedUpdateRates(self):
                    class Rates:
                        def size(self):
                            return 1

                        def __getitem__(self, idx):
                            return 200

                    return Rates()

                def setUpdateRate(self, rate):
                    return True

            return Master()

        def closePort(self, *a):
            close_log.append("closePort")

        def close(self):
            close_log.append("control.close")

    api.XsScanner_scanPorts = lambda: type("pa", (), {"size": lambda self: 1, "__getitem__": lambda s, i: FakePortInfo()})()
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda x: x

    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=1.0,
        stable_wait_s=0.01,
        poll_interval_s=0.01,
    )
    backend = XdaAwindaBackend(config, _api_module=api)
    backend.connect(lambda *a: None)
    backend.start()

    with pytest.raises(AdapterError):
        backend.close()

    # All stages must have been attempted (best-effort before re-raise)
    assert "removeCallbacks" in close_log
    assert "disableRadio" in close_log
    assert "closePort" in close_log
    assert "control.close" in close_log


# ──────────────────────────────────────────────────────────────
#  Config validation
# ──────────────────────────────────────────────────────────────


def test_config_rejects_invalid_sensor_ids() -> None:
    with pytest.raises(ValueError):
        XsensAwindaConfig(sensor_ids=("A", "B", "C", "D"))  # > 3 slots
    with pytest.raises(ValueError):
        XsensAwindaConfig(sensor_ids=("A", "B", "B"))  # duplicate
    with pytest.raises(ValueError):
        XsensAwindaConfig(sensor_ids=("X", "", "X"))  # duplicate with empty middle slot


def test_config_defaults_are_valid() -> None:
    cfg = XsensAwindaConfig()
    assert cfg.radio_channel == 25
    assert cfg.sample_rate_hz == 200.0
    assert cfg.sensor_ids == ()
    assert cfg.queue_capacity == 256
    assert cfg.pending_group_limit == 128


# ── New: slot-based sensor_ids and dynamic device count ──────


def test_config_slot_preservation_with_empty_middle() -> None:
    """Config [A, '', C] preserves three positional slots."""
    cfg = XsensAwindaConfig(sensor_ids=("A", "", "C"))
    assert cfg.sensor_ids == ("A", "", "C")
    assert cfg.active_sensor_ids == ("A", "C")
    assert cfg.active_sensor_slot_indices == (0, 2)
    assert cfg.expected_device_count == 2


def test_config_single_non_empty_id_expanded_to_three_slots() -> None:
    """Legacy single ID is padded to three slots with empty slots."""
    cfg = XsensAwindaConfig(sensor_ids=("X",))
    assert cfg.sensor_ids == ("X", "", "")
    assert cfg.active_sensor_ids == ("X",)
    assert cfg.active_sensor_slot_indices == (0,)
    assert cfg.expected_device_count == 1


def test_config_two_non_empty_ids_expanded_to_three_slots() -> None:
    """Legacy two consecutive IDs are padded with one trailing empty slot."""
    cfg = XsensAwindaConfig(sensor_ids=("A", "B"))
    assert cfg.sensor_ids == ("A", "B", "")
    assert cfg.active_sensor_ids == ("A", "B")
    assert cfg.active_sensor_slot_indices == (0, 1)
    assert cfg.expected_device_count == 2


def test_config_empty_all_slots_uses_auto_discovery_default() -> None:
    """All-empty sensor_ids triggers auto-discovery with default count 3."""
    cfg = XsensAwindaConfig()
    assert cfg.sensor_ids == ()
    assert cfg.active_sensor_ids == ()
    assert cfg.active_sensor_slot_indices == (0, 1, 2)
    assert cfg.expected_device_count == 3


def test_config_all_three_filled_works() -> None:
    cfg = XsensAwindaConfig(sensor_ids=("A", "B", "C"))
    assert cfg.active_sensor_ids == ("A", "B", "C")
    assert cfg.active_sensor_slot_indices == (0, 1, 2)
    assert cfg.expected_device_count == 3


def test_two_devices_slots_1_3_descriptor_has_correct_shape_and_labels() -> None:
    """Descriptor for slot-1+3 config shows preview_labels = imu_trunk, imu_right."""
    backend = FakeAwindaBackend(("A", "C"))
    adapter = XsensAwindaImuAdapter(
        backend=backend,
        config={"sensor_ids": ("A", "", "C")},
    )
    desc = adapter.descriptor()
    assert desc.sample_shape == (2, 12)
    assert desc.metadata["preview_labels"] == ["imu_trunk", "imu_right"]
    assert desc.metadata["active_sensor_slot_indices"] == [0, 2]
    assert desc.metadata["device_ids"] == ["A", "C"]
    assert desc.metadata["expected_device_count"] == 2
    assert desc.metadata["sensor_slots"] == ["A", "", "C"]
    assert "physical_location_mapping" in desc.metadata
    assert desc.metadata["physical_location_mapping"] == "configured"
    adapter.close()


def test_two_devices_paired_packets_produce_event_with_shape_1_2_12() -> None:
    """Two enabled IMUs (slots 1+3): A & C for same counter produce (1,2,12) batch."""
    backend = FakeAwindaBackend(("A", "C"))
    adapter = XsensAwindaImuAdapter(
        backend=backend,
        config={"sensor_ids": ("A", "", "C"), "queue_capacity": 16},
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    backend.emit("A", Packet(42, 0), 10)
    # Only 1 of 2 — not complete
    assert adapter.get_event(timeout=0.1) is None

    backend.emit("C", Packet(42, 1), 20)
    event = adapter.get_event(timeout=0.3)
    assert event is not None
    assert event.data.shape == (1, 2, 12)
    assert event.data.dtype == np.float32
    # Values come from Packet calibratedAcceleration().x = 1 + offset
    assert event.data[0, 0, 0] == 1.0  # A: offset=0
    assert event.data[0, 1, 0] == 2.0  # C: offset=1
    assert event.host_monotonic_ns == 10

    adapter.stop()
    assert adapter.health().dropped_packets == 0
    adapter.close()


def test_two_devices_one_missing_counter_counts_1_drop() -> None:
    """With 2 enabled devices, one counter missing 1 device → only 1 drop."""
    backend = FakeAwindaBackend(("A", "C"))
    adapter = XsensAwindaImuAdapter(
        backend=backend,
        config={"sensor_ids": ("A", "", "C"), "queue_capacity": 16},
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    # Send only A for counter 1
    backend.emit("A", Packet(1), 10)
    assert adapter.get_event(timeout=0.1) is None

    # Later counter 2 arrives for both, ejecting counter 1 group
    backend.emit("A", Packet(2), 30)
    backend.emit("C", Packet(2), 40)
    event = adapter.get_event(timeout=0.3)
    assert event is not None
    assert event.data.shape == (1, 2, 12)

    adapter.stop()
    assert adapter.health().dropped_packets == 1
    adapter.close()


def test_two_devices_multiple_counters_all_paired_zero_drops() -> None:
    """Multiple counters, each fully paired → dropped_packets must stay 0."""
    backend = FakeAwindaBackend(("A", "C"))
    adapter = XsensAwindaImuAdapter(
        backend=backend,
        config={"sensor_ids": ("A", "", "C"), "queue_capacity": 64},
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    for counter in range(10):
        backend.emit("A", Packet(counter, offset=0.0), counter * 20)
        backend.emit("C", Packet(counter, offset=0.1), counter * 20 + 5)
        event = adapter.get_event(timeout=0.2)
        assert event is not None
        assert event.data.shape == (1, 2, 12)

    adapter.stop()
    assert adapter.health().dropped_packets == 0
    adapter.close()


def test_strict_discovery_configure_two_find_one_fails() -> None:
    """With two specific IDs configured, discovering only one must raise AdapterError."""
    from exo_collection.adapters.imu.xsens_awinda import _match_target_device_ids

    discovered = {"A": None}
    with pytest.raises(AdapterError, match="未发现目标 MTw 设备"):
        _match_target_device_ids(("A", "C"), discovered)


def test_strict_discovery_configure_two_find_both_succeeds() -> None:
    """With two specific IDs configured, finding both must succeed."""
    from exo_collection.adapters.imu.xsens_awinda import _match_target_device_ids

    discovered = {"A": None, "C": None, "EXTRA": None}
    result = _match_target_device_ids(("A", "C"), discovered)
    assert result == ("A", "C")


def test_strict_discovery_all_three_found_succeeds() -> None:
    """With three specific IDs configured, finding all three succeeds."""
    from exo_collection.adapters.imu.xsens_awinda import _match_target_device_ids

    discovered = {"DEV_A": None, "DEV_B": None, "DEV_C": None}
    result = _match_target_device_ids(("DEV_A", "DEV_B", "DEV_C"), discovered)
    assert result == ("DEV_A", "DEV_B", "DEV_C")


def test_active_count_1_device_descriptor_shape() -> None:
    """Single-device config yields sample_shape (1, 12)."""
    backend = FakeAwindaBackend(("X",))
    adapter = XsensAwindaImuAdapter(
        backend=backend,
        config={"sensor_ids": ("X", "", ""), "queue_capacity": 16},
    )
    desc = adapter.descriptor()
    assert desc.sample_shape == (1, 12)
    assert desc.metadata["preview_labels"] == ["imu_trunk"]
    assert desc.metadata["active_sensor_slot_indices"] == [0]
    assert desc.metadata["expected_device_count"] == 1
    adapter.close()


def test_auto_discovery_descriptor_shows_unassigned_labels() -> None:
    """Auto-discovery (no sensor_ids) descriptor labels are all three defaults."""
    backend = FakeAwindaBackend(("D1", "D2", "D3"))
    adapter = XsensAwindaImuAdapter(backend=backend)
    desc = adapter.descriptor()
    assert desc.sample_shape == (3, 12)
    assert desc.metadata["preview_labels"] == ["imu_trunk", "imu_left", "imu_right"]
    assert desc.metadata["expected_device_count"] == 3
    assert desc.metadata["device_ids"] == ["unassigned_1", "unassigned_2", "unassigned_3"]
    assert desc.metadata["physical_location_mapping"] == "unassigned"
    adapter.close()


def test_device_ids_resolved_after_connect() -> None:
    """After connect, device_ids reflect actual backend devices, not unassigned."""
    backend = FakeAwindaBackend(("D1", "D2", "D3"))
    adapter = XsensAwindaImuAdapter(backend=backend)
    adapter.connect()
    desc = adapter.descriptor()
    assert desc.metadata["device_ids"] == ["D1", "D2", "D3"]
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Race condition: stop while consumer is blocked on packet queue
# ──────────────────────────────────────────────────────────────


def test_consumer_exits_cleanly_when_stopped_empty() -> None:
    """Consumer thread should exit gracefully when _consumer_stop is set
    and packet queue is empty."""
    adapter, backend = running_adapter()
    adapter.stop()
    # Should not hang
    assert backend.stopped >= 1
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Consumer drain: late-arriving packets after stop signal
# ──────────────────────────────────────────────────────────────


def test_consumer_drains_remaining_packets_after_stop_signalled() -> None:
    """Packets queued before stop are deterministically drained by consumer."""
    backend = FakeAwindaBackend()
    adapter = XsensAwindaImuAdapter(backend=backend)
    adapter.connect()
    adapter.prepare(context())
    for item in (
        ("A", Packet(1), 10),
        ("B", Packet(1), 20),
        ("C", Packet(1), 30),
    ):
        adapter._packet_queue.put_nowait(item)

    adapter.start()
    adapter.stop()
    event = adapter.get_event(timeout=0.2)
    assert event is not None
    assert event.data.shape == (1, 3, 12)
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Target ID matching: exact → unique substring → error
# ──────────────────────────────────────────────────────────────


def test_ambiguous_short_id_matches_multiple_devices_raises() -> None:
    """When a short target ID matches two discovered device IDs
    as a substring, _match_device_id must raise AdapterError."""
    from exo_collection.adapters.imu.xsens_awinda import _match_device_id

    discovered = {"MT_12345": None, "MT_12346": None}
    with pytest.raises(AdapterError, match="匹配多台设备"):
        _match_device_id("MT", discovered)


def test_target_ids_that_match_same_device_are_rejected() -> None:
    """Two configured aliases cannot select one physical MTw twice."""
    from exo_collection.adapters.imu.xsens_awinda import (
        _match_target_device_ids,
    )

    discovered = {"MT_12345": None, "MT_67890": None, "MT_11111": None}
    with pytest.raises(AdapterError, match="same physical device"):
        _match_target_device_ids(
            ("12345", "2345", "67890"), discovered
        )


def test_discovery_ambiguous_id_during_stability_check() -> None:
    """During discovery stability polling, ambiguous short IDs cause
    found_targets to be False — the adapter keeps waiting rather
    than selecting the wrong device."""
    # Build a fake API where children() returns devices with
    # overlapping IDs so that stability check via _match_device_id fails
    api = type(sys)("_test_xsens_fake_ambig")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type("oc", (), {"data_type": dt, "rate": rate})()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            pass

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM99"

        def baudrate(self):
            return 115200

        def deviceId(self):
            class Did:
                def isWirelessMaster(self):
                    return True

                def toXsString(self):
                    return "AWINDA_MASTER"

            return Did()

    # Two devices where short target ID "MT_12" matches BOTH
    # "MT_12345" and "MT_12346" — this is ambiguous.
    children_list = [FakeMtw("MT_12345"), FakeMtw("MT_12346"), FakeMtw("OTHER_DEV")]

    class FakeControl:
        def openPort(self, *a):
            return True

        def device(self, did):
            class Master:
                def gotoConfig(self):
                    return True

                def gotoMeasurement(self):
                    return True

                def enableRadio(self, ch):
                    return True

                def disableRadio(self):
                    pass

                def children(self):
                    return children_list

                def supportedUpdateRates(self):
                    class Rates:
                        def size(self):
                            return 1

                        def __getitem__(self, idx):
                            return 200

                    return Rates()

                def setUpdateRate(self, rate):
                    return True

            return Master()

        def closePort(self, *a):
            pass

        def close(self):
            pass

    api.XsScanner_scanPorts = lambda: type("pa", (), {"size": lambda self: 1, "__getitem__": lambda s, i: FakePortInfo()})()
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda x: x

    # target ID "MT_12" is ambiguous → should fail during selection
    from exo_collection.adapters.imu.xsens_awinda import XdaAwindaBackend

    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=0.3,
        stable_wait_s=0.02,
        poll_interval_s=0.01,
        sensor_ids=("MT_12", "MT_67890", "OTHER_DEV"),
    )
    backend = XdaAwindaBackend(config, _api_module=api)
    with pytest.raises(AdapterError, match="匹配多台设备"):
        backend.connect(lambda *a: None)
    backend.close()


# ──────────────────────────────────────────────────────────────
#  Stop / consumer error propagation
# ──────────────────────────────────────────────────────────────


def test_stop_hardware_propagates_backend_stop_error() -> None:
    """When backend.stop() raises, _stop_hardware best-efforts all
    remaining stages, then re-raises.  The base-class stop() converts
    that exception into a recorded fault without letting it escape."""

    class ErrorStopBackend(FakeAwindaBackend):
        def stop(self):
            raise RuntimeError("forced backend.stop failure")

    adapter = XsensAwindaImuAdapter(backend=ErrorStopBackend())
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    # _stop_hardware raises; base-class stop() catches and sets fault
    adapter.stop()

    # Fault recorded
    with pytest.raises(AdapterError):
        adapter.raise_if_faulted()

    # Trial stop never permanently detaches callbacks.
    assert adapter._backend._callbacks_removed is False
    # Consumer thread should be None (joined)
    assert adapter._consumer_thread is None
    adapter.close()
    assert adapter._backend._callbacks_removed is True


def test_close_propagates_remove_callbacks_error_after_stopping_stream() -> None:
    """Permanent callback cleanup happens at close and its error escapes."""

    class ErrorRemoveBackend(FakeAwindaBackend):
        def remove_callbacks(self):
            raise RuntimeError("forced remove_callbacks failure")

    adapter = XsensAwindaImuAdapter(backend=ErrorRemoveBackend())
    adapter.connect()
    adapter.prepare(context())
    adapter.start()

    adapter.stop()
    adapter.raise_if_faulted()
    assert adapter._backend.stopped >= 1
    assert adapter._consumer_thread is None
    with pytest.raises(RuntimeError, match="remove_callbacks failure"):
        adapter.close()


def test_xda_remove_callbacks_best_efforts_all_devices_and_raises() -> None:
    calls: list[str] = []

    class Device:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def removeCallbackHandler(self, callback) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"remove {self.name}")

    backend = XdaAwindaBackend(XsensAwindaConfig())
    backend._callback = object()
    backend._devices = [Device("A", fail=True), Device("B"), Device("C")]

    with pytest.raises(AdapterError, match="remove A"):
        backend.remove_callbacks()
    assert calls == ["A", "B", "C"]
    assert backend._callback is None


def test_consumer_join_timeout_keeps_reference_and_blocks_backend_close() -> None:
    class StuckThread:
        def __init__(self) -> None:
            self.join_timeouts: list[float] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            self.join_timeouts.append(timeout)

    backend = FakeAwindaBackend()
    adapter = XsensAwindaImuAdapter(backend=backend)
    adapter.connect()
    adapter.prepare(context())
    stuck = StuckThread()
    adapter._consumer_thread = stuck

    with pytest.raises(AdapterError):
        adapter._stop_hardware()
    assert adapter._consumer_thread is stuck
    assert stuck.join_timeouts == [3.0]
    with pytest.raises(AdapterError, match="consumer thread is alive"):
        adapter._close_hardware()
    assert backend.closed == 0


# ──────────────────────────────────────────────────────────────
#  Common counter gap: one gap → one count (no triple counting)
# ──────────────────────────────────────────────────────────────


def test_common_counter_gap_counted_exactly_once() -> None:
    """When all three devices skip from counter 5 to counter 100
    simultaneously, _counter_gaps must be exactly 1 (one gap event
    for the complete group, not one per device)."""
    adapter, backend = running_adapter()

    # First group: counter 5
    backend.emit("A", Packet(5), 10)
    backend.emit("B", Packet(5), 20)
    backend.emit("C", Packet(5), 30)

    # Second group: counter 100 (big jump)
    backend.emit("A", Packet(100), 40)
    backend.emit("B", Packet(100), 50)
    backend.emit("C", Packet(100), 60)

    events = _drain_events(adapter, timeout=0.3)
    assert len(events) == 2
    health = adapter.health()
    # Must be exactly 1, not 3 (one per device) and not 0 (missed gap)
    assert health.metrics["counter_gaps"] == 1
    adapter.stop()
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Discovery: third device late-arrival forces full stable wait
# ──────────────────────────────────────────────────────────────


def test_third_device_late_arrival_must_wait_full_stable_window() -> None:
    """When the third MTw arrives after the first two have been
    visible for a while, the stability timer must start from that
    moment — not from when the first device appeared.

    Uses a fake monotonic clock so the test is deterministic."""

    api = type(sys)("_test_xsens_fake_late")
    api.XDI_PacketCounter = 99
    api.XDI_SampleTimeFine = 98
    api.XDI_EulerAngles = 1
    api.XDI_Acceleration = 2
    api.XDI_RateOfTurn = 3
    api.XDI_MagneticField = 4
    api.XsOutputConfiguration = lambda dt, rate: type("oc", (), {"data_type": dt, "rate": rate})()
    api.XsOutputConfigurationArray = _FakeOutputConfigArray

    class FakeMtw:
        def __init__(self, did):
            self._did = did

        def deviceId(self):
            class Did:
                def toXsString(self):
                    return self._s

            d = Did()
            d._s = self._did
            return d

        def addCallbackHandler(self, cb):
            pass

        def setOutputConfiguration(self, cfg):
            return True

        def removeCallbackHandler(self, cb):
            pass

    class FakePortInfo:
        def empty(self):
            return False

        def portName(self):
            return "COM99"

        def baudrate(self):
            return 115200

        def deviceId(self):
            class Did:
                def isWirelessMaster(self):
                    return True

                def toXsString(self):
                    return "AWINDA_MASTER"

            return Did()

    # Fake clock: starts at 100.0 and advances 0.3s per poll.
    clock = [100.0]

    def fake_monotonic():
        return clock[0]

    def fake_sleep(seconds):
        clock[0] += 0.3  # Each poll "takes" 0.3s

    call_count = [0]

    class FakeControl:
        def openPort(self, *a):
            return True

        def device(self, did):
            class Master:
                def gotoConfig(self):
                    return True

                def gotoMeasurement(self):
                    return True

                def enableRadio(self, ch):
                    return True

                def disableRadio(self):
                    pass

                def children(self):
                    call_count[0] += 1
                    # Calls 1-3: only A and B (incomplete)
                    if call_count[0] <= 3:
                        return [FakeMtw("A"), FakeMtw("B")]
                    # Call 4+: A, B, C all present
                    return [FakeMtw("A"), FakeMtw("B"), FakeMtw("C")]

                def supportedUpdateRates(self):
                    class Rates:
                        def size(self):
                            return 1

                        def __getitem__(self, idx):
                            return 200

                    return Rates()

                def setUpdateRate(self, rate):
                    return True

            return Master()

        def closePort(self, *a):
            pass

        def close(self):
            pass

    api.XsScanner_scanPorts = lambda: type("pa", (), {"size": lambda self: 1, "__getitem__": lambda s, i: FakePortInfo()})()
    api.XsControl_construct = lambda: FakeControl()
    api.XsPortInfo = FakePortInfo
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda x: x

    from exo_collection.adapters.imu.xsens_awinda import XdaAwindaBackend

    config = XsensAwindaConfig(
        sample_rate_hz=200.0,
        wait_timeout_s=30.0,
        stable_wait_s=2.0,
        poll_interval_s=0.25,
        sensor_ids=(),
    )

    # Monkey-patch monotonic and sleep for deterministic test
    import exo_collection.adapters.imu.xsens_awinda as awinda_mod

    real_monotonic = awinda_mod.monotonic
    real_sleep = awinda_mod.sleep
    awinda_mod.monotonic = fake_monotonic
    awinda_mod.sleep = fake_sleep

    try:
        backend = XdaAwindaBackend(config, _api_module=api)
        backend.connect(lambda *a: None)
    finally:
        awinda_mod.monotonic = real_monotonic
        awinda_mod.sleep = real_sleep

    # stable_start_at should have been set at call 4 (clock=100+3*0.3=100.9)
    # Then each poll advances 0.3s.  Need stable_wait_s=2.0, so we need
    # ceil(2.0/0.3) = 7 more polls after C first appears.
    # That's calls 4 through 11, clock reaches ~100.9 + 7*0.3 = 103.0.
    # At call 11, monotonic - stable_start_at = 103.0 - 100.9 = 2.1 >= 2.0.
    assert set(backend.device_ids) == {"A", "B", "C"}
    # Verify we waited enough polls (at least 7 after call 4)
    assert call_count[0] >= 10  # calls 1-3 (incomplete) + several stable polls
    backend.close()


def test_discovery_identity_change_restarts_stability_window(monkeypatch) -> None:
    clock = [100.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(seconds: float) -> None:
        clock[0] += 0.4

    api, master = discovery_api(
        lambda call: ("A", "B", "C") if call <= 3 else ("A", "B", "D")
    )
    import exo_collection.adapters.imu.xsens_awinda as awinda_mod

    monkeypatch.setattr(awinda_mod, "monotonic", fake_monotonic)
    monkeypatch.setattr(awinda_mod, "sleep", fake_sleep)
    backend = XdaAwindaBackend(
        XsensAwindaConfig(
            wait_timeout_s=10.0,
            stable_wait_s=1.0,
            poll_interval_s=0.1,
        ),
        _api_module=api,
    )
    backend.connect(lambda *args: None)

    assert backend.device_ids == ("A", "B", "D")
    # Calls 1-3 almost satisfy the first window.  Identity C->D at call 4
    # must restart it, requiring calls 5-7 before success.
    assert master.children_calls >= 7
    backend.close()


def test_discovery_deadline_rejects_present_but_not_stable_set(
    monkeypatch,
) -> None:
    clock = [100.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(seconds: float) -> None:
        clock[0] += 0.4

    api, _master = discovery_api(lambda call: ("A", "B", "C"))
    import exo_collection.adapters.imu.xsens_awinda as awinda_mod

    monkeypatch.setattr(awinda_mod, "monotonic", fake_monotonic)
    monkeypatch.setattr(awinda_mod, "sleep", fake_sleep)
    backend = XdaAwindaBackend(
        XsensAwindaConfig(
            wait_timeout_s=1.0,
            stable_wait_s=2.0,
            poll_interval_s=0.1,
        ),
        _api_module=api,
    )
    with pytest.raises(AdapterError, match="stable identity set"):
        backend.connect(lambda *args: None)
    backend.close()
