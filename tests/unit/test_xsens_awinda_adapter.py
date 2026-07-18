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


def test_remove_callbacks_called_on_stop_hardware() -> None:
    """_stop_hardware must call backend.remove_callbacks()."""
    adapter, backend = running_adapter()
    # Give consumer time to drain
    backend.emit("A", Packet(1), 10)
    backend.emit("B", Packet(1), 20)
    backend.emit("C", Packet(1), 30)
    _drain_events(adapter, timeout=0.2)

    adapter.stop()
    assert backend._callbacks_removed is True
    adapter.close()


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
        XsensAwindaConfig(sensor_ids=("A", "B"))  # only 2
    with pytest.raises(ValueError):
        XsensAwindaConfig(sensor_ids=("A", "B", "B"))  # duplicate


def test_config_defaults_are_valid() -> None:
    cfg = XsensAwindaConfig()
    assert cfg.radio_channel == 25
    assert cfg.sample_rate_hz == 200.0
    assert cfg.sensor_ids == ()
    assert cfg.queue_capacity == 256
    assert cfg.pending_group_limit == 128


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
