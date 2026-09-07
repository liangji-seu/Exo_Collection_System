"""Xsens MTw direct-USB (wired) adapter for one to three MTw sensors.

Replaces the Awinda wireless backend: every MTw is connected over its own USB
data cable, so there is no wireless master and no cross-device hardware sync.
The MTw units run on independent crystals.  Samples are therefore aligned on
the host by quantising callback arrival time into nominal-sample buckets
rather than by a shared PacketCounter (which the wireless master used to keep
in lock-step).

The emitted ``SampleBatch`` keeps the exact downstream contract of the Awinda
adapter — ``sample_shape=(N, 12)`` and the same 12 IMU channels — so the
preview, hdf5 writer and opensim pipeline are untouched.
"""

from __future__ import annotations

import traceback
from collections import deque
from dataclasses import asdict, dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter_ns, time_ns
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.adapters.imu.xsens_awinda import (
    _COUNTER_WRAP_MOD,
    _IMU_SLOT_PREVIEW_LABELS,
    _PendingGroup,
    _match_device_id,
    _read_optional_packet_counter,
    _read_optional_sample_time_fine,
    parse_xsens_packet,
)
from exo_collection.domain.events import SampleBatch


# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class XsensMtwUsbConfig:
    device_id: str = "imu_xsens_mtw_usb"
    clock_domain: str = "imu_xsens_mtw_usb_clock"
    # Retained for compatibility with the unchanged settings dialog; a wired
    # MTw has no radio channel, so this value is ignored by the backend.
    radio_channel: int = 25
    sample_rate_hz: float = 100.0
    expected_device_count: int = 3
    sensor_ids: tuple[str, ...] = ()
    pending_group_limit: int = 128
    queue_capacity: int = 256

    def __post_init__(self) -> None:
        ids = tuple(str(item).strip() for item in self.sensor_ids)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("MTw USB requires a positive sample rate")
        if ids:
            # Migrate legacy consecutive 1/2-ID settings into positional slots.
            if len(ids) < 3 and all(ids):
                ids = (*ids, *("" for _ in range(3 - len(ids))))
            if len(ids) != 3:
                raise ValueError(
                    "sensor_ids must be empty or contain exactly three positional slots"
                )
            active_ids = tuple(item for item in ids if item)
            if not active_ids or len(set(active_ids)) != len(active_ids):
                raise ValueError("enabled sensor IDs must be unique")
            object.__setattr__(self, "expected_device_count", len(active_ids))
        elif not 1 <= self.expected_device_count <= 3:
            raise ValueError("expected_device_count must be in [1, 3]")
        object.__setattr__(self, "sensor_ids", ids)
        if self.pending_group_limit <= 0 or self.queue_capacity <= 0:
            raise ValueError("pending_group_limit and queue_capacity must be positive")

    @property
    def active_sensor_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self.sensor_ids if item)

    @property
    def active_sensor_slot_indices(self) -> tuple[int, ...]:
        if not self.sensor_ids:
            return tuple(range(self.expected_device_count))
        return tuple(index for index, item in enumerate(self.sensor_ids) if item)


def _coerce_config(
    value: XsensMtwUsbConfig | Mapping[str, Any] | None,
) -> XsensMtwUsbConfig:
    if value is None:
        return XsensMtwUsbConfig()
    if isinstance(value, XsensMtwUsbConfig):
        return value
    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw.pop("id")
    if "update_rate" in raw and "sample_rate_hz" not in raw:
        raw["sample_rate_hz"] = raw.pop("update_rate")
    if "expected_count" in raw and "expected_device_count" not in raw:
        raw["expected_device_count"] = raw.pop("expected_count")
    allowed = XsensMtwUsbConfig.__dataclass_fields__
    return XsensMtwUsbConfig(**{key: item for key, item in raw.items() if key in allowed})


# ──────────────────────────────────────────────────────────────
#  Backend protocol
# ──────────────────────────────────────────────────────────────

class MtwUsbBackend(Protocol):
    device_ids: tuple[str, ...]
    actual_rate_hz: int
    metadata: Mapping[str, Any]

    def connect(self, on_packet: Callable[[str, Any, int], None]) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def remove_callbacks(self) -> None: ...


# ──────────────────────────────────────────────────────────────
#  XDA  Backend (direct USB, no master)
# ──────────────────────────────────────────────────────────────

class XdaMtwUsbBackend:
    """Own the vendor XDA objects for several independently-wired MTw units."""

    def __init__(self, config: XsensMtwUsbConfig, *, _api_module: Any = None) -> None:
        self.config = config
        self.device_ids: tuple[str, ...] = ()
        self.actual_rate_hz = int(round(config.sample_rate_hz))
        self.metadata: dict[str, Any] = {}
        self._api = _api_module
        self._xda: Any = None
        self._control: Any = None
        self._devices: list[tuple[str, Any, Any]] = []  # (name, device, port)
        self._callback: Any = None
        self._measurement_started = False

    def connect(self, on_packet: Callable[[str, Any, int], None]) -> None:
        if self._api is not None:
            xda = self._api
        else:
            try:
                import xsensdeviceapi as xda
            except ImportError as exc:
                raise AdapterError(
                    "未安装 Xsens MT SDK 的 Python 3.11 x64 xsensdeviceapi wheel。"
                ) from exc
        self._xda = xda
        control = xda.XsControl_construct()
        if not control:
            raise AdapterError("XsControl_construct 失败")
        self._control = control
        rate_hz_int = int(round(self.config.sample_rate_hz))

        # 1. scan ports and keep only directly-wired MTw units
        ports = xda.XsScanner_scanPorts()
        mtw_ports: dict[str, Any] = {}
        inventory: list[dict[str, Any]] = []
        for index in range(ports.size()):
            p = ports[index]
            did = str(p.deviceId().toXsString()).upper()
            mtw = bool(p.deviceId().isMtw())
            inventory.append(
                {
                    "id": did,
                    "port": str(p.portName()),
                    "baudrate": int(p.baudrate()),
                    "is_mtw": mtw,
                }
            )
            if mtw:
                if did in mtw_ports:
                    raise AdapterError(f"扫描出现重复 ID {did}，请检查连接。")
                mtw_ports[did] = p
        self.metadata["scan_inventory"] = inventory

        target_ids = self.config.active_sensor_ids
        if target_ids:
            ordered_ids = tuple(
                _match_device_id(target, mtw_ports) for target in target_ids
            )
        else:
            ordered_ids = tuple(sorted(mtw_ports))
            if len(ordered_ids) > self.config.expected_device_count:
                ordered_ids = ordered_ids[: self.config.expected_device_count]

        if not ordered_ids:
            found = ", ".join(sorted(mtw_ports)) if mtw_ports else "（无 MTw）"
            configured = ", ".join(target_ids) if target_ids else "（未配置 ID）"
            raise AdapterError(
                "未找到任何配置的 USB MTw。"
                f"已发现 MTw：{found}；配置：{configured}。"
                "请拔掉 Dongle，用数据线直连；关闭 Collector/MT Manager，检查驱动和 ID。"
            )

        # 2. open + configure each device independently
        supported_rates: list[int] = []
        for name, did in zip(
            (f"mtw_{i + 1}" for i in range(len(ordered_ids))), ordered_ids
        ):
            p = mtw_ports[did]
            if not control.openPort(p.portName(), p.baudrate()):
                raise AdapterError(f"{name} 端口 {p.portName()} 打开失败；可能被其他程序占用。")
            dev = control.device(p.deviceId())
            if not dev:
                raise AdapterError(f"{name} 设备对象不可用")
            if dev.connectivityState() != xda.XCS_PluggedIn:
                raise AdapterError(f"{name} SDK 未确认有线状态 XCS_PluggedIn，拒绝继续。")
            if not dev.gotoConfig():
                raise AdapterError(f"{name} 不能进入配置模式")
            dev.setOptions(xda.XSO_Orientation | xda.XSO_Calibrate, 0)
            # PacketCounter/SampleTimeFine carry no explicit frequency (they ride
            # along with each packet); the proven mtw_usb demo requests them at 0.
            output = xda.XsOutputConfigurationArray()
            for data_id, freq in (
                (xda.XDI_PacketCounter, 0),
                (xda.XDI_SampleTimeFine, 0),
                (xda.XDI_Acceleration, rate_hz_int),
                (xda.XDI_RateOfTurn, rate_hz_int),
                (xda.XDI_MagneticField, rate_hz_int),
                (xda.XDI_EulerAngles, rate_hz_int),
            ):
                output.push_back(xda.XsOutputConfiguration(data_id, freq))
            if not dev.setOutputConfiguration(output):
                raise AdapterError(f"{name} 设置输出配置失败")
            supported_raw = dev.supportedUpdateRates()
            supported_rates = [int(supported_raw[i]) for i in range(supported_raw.size())]
            if rate_hz_int not in supported_rates:
                raise AdapterError(f"{name} 不支持 {rate_hz_int} Hz；设备报告：{supported_rates}")
            if not dev.setUpdateRate(rate_hz_int) or int(dev.updateRate()) != rate_hz_int:
                raise AdapterError(f"{name} 设置/读回 {rate_hz_int} Hz 失败")
            self._devices.append((name, dev, p))

        # 3. register one callback shared across devices
        class Callback(xda.XsCallback):
            def __init__(self) -> None:
                xda.XsCallback.__init__(self)

            def onLiveDataAvailable(self, device: Any, packet: Any) -> None:
                copied = xda.XsDataPacket(packet)
                on_packet(
                    str(device.deviceId().toXsString()),
                    copied,
                    perf_counter_ns(),
                )

        callback = Callback()
        self._callback = callback
        for _name, dev, _p in self._devices:
            dev.addCallbackHandler(callback)

        self.device_ids = ordered_ids
        self.actual_rate_hz = rate_hz_int
        self.metadata.update(
            {
                "transport": "direct_usb_only",
                "cross_device_hardware_sync_verified": False,
                "device_ids": list(ordered_ids),
                "actual_sample_rate_hz": rate_hz_int,
                "supported_update_rates_hz": supported_rates,
                "sensor_slots": list(self.config.sensor_ids),
                "active_sensor_slot_indices": list(self.config.active_sensor_slot_indices),
                "expected_device_count": self.config.expected_device_count,
            }
        )
        try:
            sdk_version = getattr(xda, "XsVersion", None)
            if sdk_version is not None:
                self.metadata["sdk_version"] = sdk_version().toXsString()
        except Exception:
            pass

    def remove_callbacks(self) -> None:
        callback = self._callback
        if callback is None:
            return
        first_error: BaseException | None = None
        for _name, dev, _p in self._devices:
            try:
                dev.removeCallbackHandler(callback)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._callback = None
        if first_error is not None:
            raise AdapterError(
                f"Failed to remove one or more MTw callbacks: {first_error}"
            ) from first_error

    def start(self) -> None:
        for name, dev, _p in self._devices:
            if not dev.gotoMeasurement():
                raise AdapterError(f"{name} 无法进入测量模式")
        self._measurement_started = True

    def stop(self) -> None:
        if not self._measurement_started:
            return
        for name, dev, _p in self._devices:
            if not dev.gotoConfig():
                raise AdapterError(f"{name} 停止时无法返回配置模式")
        self._measurement_started = False

    def close(self) -> None:
        first_error: BaseException | None = None

        # 1. stop measurement
        try:
            self.stop()
        except BaseException:
            if first_error is None:
                first_error = AdapterError(
                    f"MTw USB stop error: {traceback.format_exc()}"
                )

        # 2. remove callback handlers
        try:
            self.remove_callbacks()
        except BaseException:
            if first_error is None:
                first_error = AdapterError(
                    f"MTw USB removeCallbacks error: {traceback.format_exc()}"
                )

        # 3. close control (releases all open ports)
        if self._control is not None:
            try:
                self._control.close()
            except BaseException:
                if first_error is None:
                    first_error = AdapterError(
                        f"MTw USB control.close error: {traceback.format_exc()}"
                    )

        self._callback = None
        self._devices = []
        self._control = None

        if first_error is not None:
            raise first_error


# ──────────────────────────────────────────────────────────────
#  Adapter
# ──────────────────────────────────────────────────────────────

class XsensMtwUsbImuAdapter(QueuedHardwareAdapter):
    """Wired callback→packet-queue→consumer pipeline aligned on host time.

    Callback thread copies each ``XsDataPacket`` and pushes
    ``(device_id, packet, host_ns)`` to a bounded queue.  The consumer parses
    the same 12 fields, then groups the three (or N) units by a host-timestamp
    bucket (quantised to the nominal sample period) instead of a shared
    PacketCounter.
    """

    def __init__(
        self,
        config: XsensMtwUsbConfig | Mapping[str, Any] | None = None,
        *,
        backend: MtwUsbBackend | None = None,
    ) -> None:
        self._config = _coerce_config(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or XdaMtwUsbBackend(self._config)
        self._sensor_ids: tuple[str, ...] = self._config.active_sensor_ids
        self._use_device_ids: tuple[str, ...] = ()
        self._active_slot_indices: tuple[int, ...] = (
            self._config.active_sensor_slot_indices
        )

        self._packet_queue: Queue[tuple[str, Any, int | None]] = Queue(
            maxsize=self._config.queue_capacity
        )
        self._consumer_thread: Thread | None = None
        self._consumer_stop = Event()

        self._pending: dict[
            str, deque[tuple[int, np.ndarray, int | None, int | None]]
        ] = {}
        self._pending_lock = Lock()
        self._per_device_last_counter: dict[str, int] = {}

        self._sample_index = 0
        self._batch_sequence = 0
        self._incomplete_samples = 0
        self._malformed_packets = 0
        self._duplicate_packets = 0
        self._counter_gaps = 0
        self._max_arrival_spread_ns = 0
        self._accepting_packets = False
        self._period_ns: float = 1e9 / 100.0

    # ── descriptor / snapshot ──────────────────────────────

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        ids = self._use_device_ids or self._sensor_ids or tuple(
            f"unassigned_{index + 1}" for index in range(cfg.expected_device_count)
        )
        active_count = len(ids)
        slot_indices = self._active_slot_indices or tuple(range(active_count))
        preview_labels = tuple(
            _IMU_SLOT_PREVIEW_LABELS[index] for index in slot_indices
        )
        backend_meta = dict(getattr(self._backend, "metadata", {}))
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="imu",
            display_name=f"Xsens MTw USB {active_count}-MTw array",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=IMU_CHANNELS,
            units=IMU_UNITS,
            nominal_rate_hz=float(
                getattr(self._backend, "actual_rate_hz", cfg.sample_rate_hz)
            ),
            sample_shape=(active_count, len(IMU_CHANNELS)),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": False,
                "manufacturer": "Xsens",
                "system": "MTw direct-USB (wired)",
                "transport": "direct_usb_only",
                "cross_device_hardware_sync_verified": False,
                "device_ids": list(ids),
                "sensor_slots": list(cfg.sensor_ids),
                "active_sensor_slot_indices": list(slot_indices),
                "preview_labels": list(preview_labels),
                "physical_location_mapping": (
                    "configured" if cfg.active_sensor_ids else "unassigned"
                ),
                "alignment_mode": "host_timestamp_bucket",
                "expected_device_count": active_count,
                **{k: v for k, v in backend_meta.items()
                   if k not in {"device_ids", "expected_device_count",
                                "pending_group_limit", "queue_capacity"}},
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        backend_meta = dict(getattr(self._backend, "metadata", {}))
        return {
            **asdict(self._config),
            "resolved_sensor_ids": list(self._use_device_ids or self._sensor_ids),
            "active_sensor_slot_indices": list(self._active_slot_indices),
            "actual_rate_hz": getattr(self._backend, "actual_rate_hz", None),
            **{k: v for k, v in backend_meta.items()
               if k not in {"pending_group_limit", "queue_capacity"}},
            "alignment_mode": "host_timestamp_bucket",
        }

    # ── lifecycle hooks ────────────────────────────────────

    def _connect_hardware(self) -> None:
        self._backend.connect(self._on_packet)
        ids = tuple(self._backend.device_ids)
        if len(ids) != self._config.expected_device_count or len(set(ids)) != len(ids):
            raise AdapterError(
                f"MTw USB 后端必须提供 {self._config.expected_device_count} 个唯一 MTw ID，"
                f"实际为 {ids}"
            )
        self._use_device_ids = ids
        backend_slots = tuple(
            int(index)
            for index in getattr(self._backend, "metadata", {}).get(
                "active_sensor_slot_indices", ()
            )
        )
        if len(backend_slots) == len(ids) and all(0 <= index < 3 for index in backend_slots):
            self._active_slot_indices = backend_slots
        elif self._config.active_sensor_ids:
            self._active_slot_indices = self._config.active_sensor_slot_indices
        else:
            self._active_slot_indices = tuple(range(len(ids)))

    def _reset_trial_state(self) -> None:
        with self._pending_lock:
            self._pending.clear()
        self._per_device_last_counter = {}
        self._sample_index = 0
        self._batch_sequence = 0
        self._incomplete_samples = 0
        self._malformed_packets = 0
        self._duplicate_packets = 0
        self._counter_gaps = 0
        self._max_arrival_spread_ns = 0
        rate_hz = float(
            getattr(self._backend, "actual_rate_hz", self._config.sample_rate_hz)
        )
        self._period_ns = 1e9 / rate_hz
        while not self._packet_queue.empty():
            try:
                self._packet_queue.get_nowait()
            except Empty:
                break

    def _start_hardware(self) -> None:
        self._accepting_packets = True
        self._consumer_stop.clear()
        self._consumer_thread = Thread(
            target=self._consumer_loop,
            name="xsens-mtw-usb-consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        self._backend.start()

    def _stop_hardware(self) -> None:
        first_error: BaseException | None = None

        # 1. stop accepting new packets from callback
        self._accepting_packets = False

        # 2. transition hardware to config mode (best-effort)
        try:
            self._backend.stop()
        except BaseException as exc:
            if first_error is None:
                first_error = exc

        # 3. Keep vendor callbacks registered across Trial boundaries.  The
        # callback entry point is gated by _accepting_packets, while permanent
        # callback removal belongs to backend.close().

        # 4. signal consumer thread — it drains remaining queue items, then exits
        self._consumer_stop.set()

        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=3.0)
            if self._consumer_thread.is_alive():
                timeout_error = AdapterError("MTw USB 消费线程未能在 3 秒内停止")
                self._set_fault(timeout_error)
                if first_error is None:
                    first_error = timeout_error
        if self._consumer_thread is not None and not self._consumer_thread.is_alive():
            self._consumer_thread = None

        # 5. Only touch pending groups after the sole consumer has stopped.
        if self._consumer_thread is None:
            with self._pending_lock:
                for queue in self._pending.values():
                    self._incomplete_samples += len(queue)
                self._pending.clear()

        if first_error is not None:
            raise first_error

    def _close_hardware(self) -> None:
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            raise AdapterError(
                "Refusing to close MTw USB backend while consumer thread is alive"
            )
        first_error: BaseException | None = None
        try:
            self._backend.remove_callbacks()
        except BaseException as exc:
            first_error = exc
        try:
            self._backend.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error

    # ── callback entry point ───────────────────────────────

    def _on_packet(
        self, device_id: str, packet: Any, host_ns: int | None = None
    ) -> None:
        if not self._accepting_packets:
            return
        if device_id not in self._use_device_ids:
            self._set_fault(AdapterError(f"收到未配置 MTw {device_id} 的数据"))
            return

        received_ns = perf_counter_ns() if host_ns is None else int(host_ns)
        try:
            self._packet_queue.put_nowait((device_id, packet, received_ns))
        except Full:
            error = AdapterError(
                f"MTw USB packet queue overflow (capacity={self._config.queue_capacity})"
            )
            self._set_fault(error)

    # ── consumer loop ──────────────────────────────────────

    def _consumer_loop(self) -> None:
        while True:
            try:
                item = self._packet_queue.get(timeout=0.05)
            except Empty:
                if self._consumer_stop.is_set():
                    break
                continue
            try:
                self._process_one_packet(*item)
            except BaseException as exc:
                self._set_fault(exc)

    def _process_one_packet(
        self, device_id: str, packet: Any, host_ns: int
    ) -> None:
        try:
            row = parse_xsens_packet(packet)
        except BaseException:
            self._malformed_packets += 1
            return

        counter = _read_optional_packet_counter(packet)
        sample_time = _read_optional_sample_time_fine(packet)

        # Per-device counter gap detection: unlike Awinda, each wired MTw has an
        # independent crystal, so a shared/common counter cannot be assumed.
        if counter is not None:
            last = self._per_device_last_counter.get(device_id)
            if last is not None:
                delta = (counter - last) % _COUNTER_WRAP_MOD
                if 0 < delta < _COUNTER_WRAP_MOD // 2:
                    self._counter_gaps += delta - 1
            self._per_device_last_counter[device_id] = counter

        # Host-timestamp window alignment (wired: no shared counter).  Each MTw
        # streams on its own crystal, so its samples carry a fixed phase offset
        # relative to the others; a strict round() bucket would split
        # near-simultaneous samples across neighbouring buckets.  Queue per
        # device instead and pair the oldest sample of every device whenever
        # their host arrival times fall within one nominal period of each other.
        with self._pending_lock:
            queue = self._pending.setdefault(device_id, deque())
            if counter is not None and queue and queue[-1][2] == counter:
                self._duplicate_packets += 1
                return
            queue.append((host_ns, row, counter, sample_time))
            self._enforce_pending_limit_locked()
            ready = self._drain_aligned_groups_locked()

        for group_host_ns, group in ready:
            self._emit_group(group_host_ns, group)

    def _enforce_pending_limit_locked(self) -> None:
        """Evict the globally-oldest queued sample when the pending queues grow
        past the configured bound (guards against an idle/offline device letting
        the other devices' queues grow without bound).  Caller holds the lock."""
        limit = self._config.pending_group_limit * max(1, len(self._use_device_ids))
        while True:
            total = sum(len(queue) for queue in self._pending.values())
            if total <= limit:
                return
            oldest_did = min(
                (did for did in self._pending if self._pending[did]),
                key=lambda did: self._pending[did][0][0],
            )
            self._pending[oldest_did].popleft()
            self._incomplete_samples += 1

    def _drain_aligned_groups_locked(
        self,
    ) -> list[tuple[int, _PendingGroup]]:
        """Pair the oldest sample of every active device into complete groups.

        Caller holds ``_pending_lock``.  A group is complete when all devices
        have queued samples whose host arrival times fall within one nominal
        period (``self._period_ns``) — wide enough to absorb the fixed
        inter-device phase offset the independent crystals introduce.  Samples
        that grow too old to ever pair are evicted and counted as incomplete.
        """
        active_ids = self._use_device_ids
        window_ns = self._period_ns
        ready: list[tuple[int, _PendingGroup]] = []
        while True:
            heads = {}
            for did in active_ids:
                queue = self._pending.get(did)
                if not queue:
                    return ready
                heads[did] = queue[0]

            host_times = [entry[0] for entry in heads.values()]
            spread = max(host_times) - min(host_times)

            if spread <= window_ns:
                group = _PendingGroup(
                    rows={},
                    host_times={},
                    device_times={},
                    device_counters={},
                )
                for did in active_ids:
                    host_ns, row, counter, sample_time = self._pending[did].popleft()
                    group.rows[did] = row
                    group.host_times[did] = host_ns
                    group.device_times[did] = (
                        sample_time if sample_time is not None else counter
                    )
                    group.device_counters[did] = counter
                ready.append((min(host_times), group))
            else:
                # The globally-oldest sample can never align — evict it.
                oldest_did = min(
                    active_ids, key=lambda did: self._pending[did][0][0]
                )
                self._pending[oldest_did].popleft()
                self._incomplete_samples += 1

    def _emit_group(self, host_ns: int, group: _PendingGroup) -> None:
        data = np.ascontiguousarray(
            np.stack(
                [group.rows[device_id] for device_id in self._use_device_ids],
                axis=0,
            )[None, ...],
            dtype=np.float32,
        )

        host_times_list = [
            group.host_times[device_id] for device_id in self._use_device_ids
        ]
        arrival_spread = max(host_times_list) - min(host_times_list)
        if arrival_spread > self._max_arrival_spread_ns:
            self._max_arrival_spread_ns = arrival_spread

        event = SampleBatch(
            session_uuid=(
                str(self._trial.session_uuid)
                if self._trial is not None and self._trial.session_uuid is not None
                else None
            ),
            trial_uuid=str(self._trial.trial_uuid) if self._trial is not None else None,
            device_id=self._config.device_id,
            modality="imu",
            clock_domain=self._config.clock_domain,
            host_monotonic_ns=host_ns,
            host_utc_ns=time_ns(),
            first_sample_index=self._sample_index,
            sample_count=1,
            sequence_number=self._batch_sequence,
            # No unified device clock across independently-wired MTw units.
            device_timestamp=None,
            sample_rate_hz=float(self._backend.actual_rate_hz),
            data=data,
        )
        self._publish_raw(event, item_count=1, host_monotonic_ns=host_ns)
        self._sample_index += 1
        self._batch_sequence += 1

    # ── health ─────────────────────────────────────────────

    def _dropped_packets(self) -> int:
        return self._incomplete_samples

    def _sequence_gaps(self) -> int:
        return self._counter_gaps

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        with self._pending_lock:
            pending_groups = sum(len(queue) for queue in self._pending.values())
        return {
            "pending_alignment_groups": pending_groups,
            "incomplete_sensor_samples": self._incomplete_samples,
            "malformed_packets": self._malformed_packets,
            "duplicate_packets": self._duplicate_packets,
            "counter_gaps": self._counter_gaps,
            "max_arrival_spread_ns": self._max_arrival_spread_ns,
            "alignment_mode": "host_timestamp_bucket",
            "cross_device_hardware_sync_verified": False,
            "resolved_sensor_ids": ",".join(
                self._use_device_ids or self._sensor_ids
            ),
            "packet_queue_size": self._packet_queue.qsize(),
        }


__all__ = [
    "MtwUsbBackend",
    "XdaMtwUsbBackend",
    "XsensMtwUsbConfig",
    "XsensMtwUsbImuAdapter",
]
