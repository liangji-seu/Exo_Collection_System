"""Tests for the Xsens MTw direct-USB (wired) adapter.

Focuses on the two things that differ from the Awinda wireless adapter:
(1) host-timestamp bucket alignment instead of shared PacketCounter, and
(2) the wired backend's per-device open/configure path with no master.
"""

from __future__ import annotations

import sys
from time import monotonic, sleep
from uuid import uuid4

import pytest

from exo_collection.adapters.base import AdapterError, TrialContext
from exo_collection.adapters.imu.xsens_mtw_usb import (
    XdaMtwUsbBackend,
    XsensMtwUsbConfig,
    XsensMtwUsbImuAdapter,
)


# ──────────────────────────────────────────────────────────────
#  Shared test doubles – data packet
# ──────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────
#  FakeMtwUsbBackend – protocol double for adapter-level tests
# ──────────────────────────────────────────────────────────────


class FakeMtwUsbBackend:
    def __init__(self, ids=("A", "B", "C")) -> None:
        self.device_ids = tuple(ids)
        self.actual_rate_hz = 100
        self.metadata = {
            "device_ids": list(ids),
            "actual_sample_rate_hz": 100,
            "transport": "direct_usb_only",
            "cross_device_hardware_sync_verified": False,
        }
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
    backend = FakeMtwUsbBackend(ids)
    adapter = XsensMtwUsbImuAdapter(backend=backend, config={"queue_capacity": 16})
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    return adapter, backend


def _drain_events(adapter: XsensMtwUsbImuAdapter, timeout: float = 0.3):
    events = []
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        event = adapter.get_event(timeout=0.05)
        if event is not None:
            events.append(event)
        sleep(0.01)
    return events


# ──────────────────────────────────────────────────────────────
#  Config validation
# ──────────────────────────────────────────────────────────────


def test_config_defaults_are_valid() -> None:
    cfg = XsensMtwUsbConfig()
    assert cfg.device_id == "imu_xsens_mtw_usb"
    assert cfg.clock_domain == "imu_xsens_mtw_usb_clock"
    assert cfg.sample_rate_hz == 100.0
    assert cfg.expected_device_count == 3
    assert cfg.sensor_ids == ()
    assert cfg.radio_channel == 25
    assert cfg.pending_group_limit == 128
    assert cfg.queue_capacity == 256


def test_config_slot_preservation_with_empty_middle() -> None:
    cfg = XsensMtwUsbConfig(sensor_ids=("A", "", "C"))
    assert cfg.sensor_ids == ("A", "", "C")
    assert cfg.active_sensor_ids == ("A", "C")
    assert cfg.active_sensor_slot_indices == (0, 2)
    assert cfg.expected_device_count == 2


def test_config_rejects_invalid_sensor_ids() -> None:
    with pytest.raises(ValueError):
        XsensMtwUsbConfig(sensor_ids=("A", "B", "C", "D"))  # > 3 slots
    with pytest.raises(ValueError):
        XsensMtwUsbConfig(sensor_ids=("A", "B", "B"))  # duplicate


# ──────────────────────────────────────────────────────────────
#  Descriptor
# ──────────────────────────────────────────────────────────────


def test_descriptor_shape_alignment_mode_and_labels() -> None:
    backend = FakeMtwUsbBackend(("A", "B", "C"))
    adapter = XsensMtwUsbImuAdapter(backend=backend)
    desc = adapter.descriptor()
    assert desc.sample_shape == (3, 12)
    assert desc.modality == "imu"
    assert desc.metadata["alignment_mode"] == "host_timestamp_bucket"
    assert desc.metadata["cross_device_hardware_sync_verified"] is False
    assert desc.metadata["transport"] == "direct_usb_only"
    assert desc.metadata["preview_labels"] == [
        "imu_left_leg",
        "imu_right_leg",
        "imu_pelvis",
    ]
    adapter.close()


def test_two_slot_descriptor_labels() -> None:
    backend = FakeMtwUsbBackend(("A", "C"))
    adapter = XsensMtwUsbImuAdapter(
        backend=backend,
        config={"sensor_ids": ("A", "", "C")},
    )
    desc = adapter.descriptor()
    assert desc.sample_shape == (2, 12)
    assert desc.metadata["preview_labels"] == ["imu_left_leg", "imu_pelvis"]
    assert desc.metadata["active_sensor_slot_indices"] == [0, 2]
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Host-timestamp bucket alignment (the wired-specific behavior)
# ──────────────────────────────────────────────────────────────


def test_three_devices_same_bucket_grouped_by_host_time() -> None:
    """Different counters still group when their host timestamps land in the
    same bucket — wired alignment ignores the per-device counter."""
    adapter, backend = running_adapter()
    adapter._t0_ns = 0
    adapter._period_ns = 10  # 10 ns per nominal bucket

    backend.emit("A", Packet(1), 1)
    backend.emit("B", Packet(2), 2)
    assert adapter.get_event(timeout=0.1) is None  # only 2 of 3

    backend.emit("C", Packet(3), 3)
    event = adapter.get_event(timeout=0.5)
    assert event is not None
    assert event.data.shape == (1, 3, 12)
    assert event.device_timestamp is None  # no unified device clock
    adapter.stop()
    adapter.close()


def test_devices_in_different_buckets_do_not_merge() -> None:
    """A and B land in bucket 0, C lands far away in bucket 2 — no complete
    sample is produced and every incomplete group is counted."""
    adapter, backend = running_adapter()
    adapter._t0_ns = 0
    adapter._period_ns = 10

    backend.emit("A", Packet(1), 0)
    backend.emit("B", Packet(2), 1)
    backend.emit("C", Packet(3), 20)  # key = round(20/10) = 2

    sleep(0.05)
    assert adapter.get_event(timeout=0.1) is None

    adapter.stop()
    # bucket 0 missing C (1), bucket 2 missing A and B (2) → 3 incomplete
    assert adapter.health().dropped_packets == 3
    adapter.close()


def test_host_timestamp_reconstructed_from_bucket() -> None:
    """Emitted host_monotonic_ns is uniform (t0 + key*period), not the raw
    arrival timestamp."""
    adapter, backend = running_adapter()
    adapter._t0_ns = 10_000_000
    adapter._period_ns = 10_000_000  # 10 ms nominal bucket

    # Arrivals are jittered around the 10 ms boundary (8/12/14 ms after t0),
    # so they all quantise to bucket key == 1.
    backend.emit("A", Packet(1), 10_000_000 + 8_000_000)
    backend.emit("B", Packet(1), 10_000_000 + 12_000_000)
    backend.emit("C", Packet(1), 10_000_000 + 14_000_000)

    event = adapter.get_event(timeout=0.5)
    assert event is not None
    # key == 1 → host = t0 + 1*period = 20_000_000 (uniform, jitter hidden)
    assert event.host_monotonic_ns == 20_000_000
    adapter.stop()
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Per-device counter gap detection
# ──────────────────────────────────────────────────────────────


def test_per_device_counter_gap_detected() -> None:
    """Each wired MTw tracks its own counter; gaps accumulate per device."""
    adapter, backend = running_adapter()
    adapter._t0_ns = 0
    adapter._period_ns = 10

    backend.emit("A", Packet(1), 0)
    backend.emit("B", Packet(1), 1)
    backend.emit("C", Packet(1), 2)

    backend.emit("A", Packet(10), 20)
    backend.emit("B", Packet(11), 21)
    backend.emit("C", Packet(12), 22)

    events = _drain_events(adapter, timeout=0.3)
    assert len(events) == 2
    # A: 1→10 (8 missing), B: 1→11 (9), C: 1→12 (10)
    assert adapter.health().metrics["counter_gaps"] == 27
    adapter.stop()
    adapter.close()


# ──────────────────────────────────────────────────────────────
#  Backend-level tests with a fake XDA API
# ──────────────────────────────────────────────────────────────


class _FakeArray(list):
    def size(self):
        return len(self)


class _FakeOutputConfigArray(list):
    def push_back(self, item):
        self.append(item)


class _FakeDeviceId:
    def __init__(self, did: str, mtw: bool = True) -> None:
        self._did = did
        self._mtw = mtw

    def isMtw(self) -> bool:
        return self._mtw

    def toXsString(self) -> str:
        return self._did


class _FakePortInfo:
    def __init__(self, did: str, mtw: bool = True) -> None:
        self._did = did
        self._mtw = mtw

    def deviceId(self):
        return _FakeDeviceId(self._did, self._mtw)

    def portName(self):
        return f"COM_{self._did}"

    def baudrate(self):
        return 115200


class FakeMtwDevice:
    """One wired MTw device as seen through the fake XDA surface."""

    def __init__(
        self,
        did: str,
        *,
        connected: int = 2,  # 2 == XCS_PluggedIn
        rate: int = 100,
        supported: tuple[int, ...] = (100, 50, 25),
    ) -> None:
        self._did = did
        self._connected = connected
        self._rate = rate
        self._supported = supported
        self.output_config: list = []
        self.set_rate: int | None = None
        self.callback = None

    def deviceId(self):
        return _FakeDeviceId(self._did, True)

    def connectivityState(self):
        return self._connected

    def gotoConfig(self):
        return True

    def setOptions(self, *args):
        pass

    def setOutputConfiguration(self, cfg):
        self.output_config = list(cfg)
        return True

    def supportedUpdateRates(self):
        return _FakeArray(self._supported)

    def setUpdateRate(self, rate):
        self.set_rate = int(rate)
        return True

    def updateRate(self):
        return self._rate

    def addCallbackHandler(self, cb):
        self.callback = cb

    def removeCallbackHandler(self, cb):
        self.callback = None

    def gotoMeasurement(self):
        return True


def make_api(devices: list[FakeMtwDevice]):
    byid = {d._did: d for d in devices}

    class FakeControl:
        def __init__(self) -> None:
            self.closed = False

        def openPort(self, name, baudrate):
            return True

        def device(self, device_id):
            key = str(device_id.toXsString()).upper()
            return byid.get(key)

        def close(self):
            self.closed = True

    api = type(sys)("_test_xsens_mtw_usb")
    api.XsScanner_scanPorts = lambda: _FakeArray(
        [_FakePortInfo(d._did, True) for d in devices]
    )
    api.XsControl_construct = lambda: FakeControl()
    api.XsCallback = type("cb", (), {})
    api.XsDataPacket = lambda p: p
    api.XCS_PluggedIn = 2
    api.XSO_Orientation = 1
    api.XSO_Calibrate = 2
    api.XsOutputConfigurationArray = _FakeOutputConfigArray
    api.XsOutputConfiguration = lambda data_id, freq: (data_id, freq)
    api.XDI_PacketCounter = 1
    api.XDI_SampleTimeFine = 2
    api.XDI_Acceleration = 3
    api.XDI_RateOfTurn = 4
    api.XDI_MagneticField = 5
    api.XDI_EulerAngles = 6
    return api, byid


def _backend(devices, **config_kwargs):
    api, _ = make_api(devices)
    config = XsensMtwUsbConfig(sample_rate_hz=100.0, **config_kwargs)
    return XdaMtwUsbBackend(config, _api_module=api)


def test_backend_connect_wired_success() -> None:
    devs = [
        FakeMtwDevice("10B42626"),
        FakeMtwDevice("10B4260D"),
        FakeMtwDevice("10B4261F"),
    ]
    backend = _backend(
        devs, sensor_ids=("10B42626", "10B4260D", "10B4261F")
    )
    collected = []
    backend.connect(lambda did, pkt, ns: collected.append(did))

    assert backend.device_ids == ("10B42626", "10B4260D", "10B4261F")
    assert backend.actual_rate_hz == 100
    for dev in devs:
        assert len(dev.output_config) == 6
        assert dev.set_rate == 100
        assert dev.callback is not None
    backend.close()


def test_backend_rejects_wireless_state() -> None:
    devs = [
        FakeMtwDevice("A", connected=3),  # not XCS_PluggedIn
        FakeMtwDevice("B", connected=3),
        FakeMtwDevice("C", connected=3),
    ]
    backend = _backend(devs)
    with pytest.raises(AdapterError, match="XCS_PluggedIn"):
        backend.connect(lambda *a: None)
    backend.close()


def test_backend_rejects_unsupported_rate() -> None:
    devs = [
        FakeMtwDevice("A", rate=80, supported=(80, 40)),
        FakeMtwDevice("B", rate=80, supported=(80, 40)),
        FakeMtwDevice("C", rate=80, supported=(80, 40)),
    ]
    backend = _backend(devs)
    with pytest.raises(AdapterError, match="不支持 100"):
        backend.connect(lambda *a: None)
    backend.close()


def test_backend_auto_mode_no_mtw_raises() -> None:
    api, _ = make_api([])
    config = XsensMtwUsbConfig(sample_rate_hz=100.0)
    backend = XdaMtwUsbBackend(config, _api_module=api)
    with pytest.raises(AdapterError, match="未找到任何配置"):
        backend.connect(lambda *a: None)
    backend.close()


def test_backend_configured_id_not_found_raises() -> None:
    devs = [FakeMtwDevice("X")]
    backend = _backend(devs, sensor_ids=("A", "", ""))
    with pytest.raises(AdapterError, match="未发现目标 MTw 设备"):
        backend.connect(lambda *a: None)
    backend.close()


def test_backend_auto_mode_takes_first_n_devices() -> None:
    devs = [
        FakeMtwDevice("DDD"),
        FakeMtwDevice("AAA"),
        FakeMtwDevice("CCC"),
        FakeMtwDevice("BBB"),  # extra, ignored (expected_device_count = 3)
    ]
    backend = _backend(devs)
    backend.connect(lambda *a: None)
    # sorted order: AAA, BBB, CCC (DDD is 4th and dropped)
    assert backend.device_ids == ("AAA", "BBB", "CCC")
    backend.close()
