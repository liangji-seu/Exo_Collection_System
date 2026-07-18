"""Xsens Awinda WirelessMaster and three-MTw hardware adapter."""

from __future__ import annotations

import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic, perf_counter_ns, sleep, time_ns
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.domain.events import SampleBatch


# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class XsensAwindaConfig:
    device_id: str = "imu_xsens_awinda"
    clock_domain: str = "imu_xsens_awinda_clock"
    radio_channel: int = 25
    sample_rate_hz: float = 200.0
    expected_device_count: int = 3
    sensor_ids: tuple[str, ...] = ()
    wait_timeout_s: float = 15.0
    stable_wait_s: float = 3.0
    poll_interval_s: float = 0.25
    pending_group_limit: int = 128
    queue_capacity: int = 256

    def __post_init__(self) -> None:
        ids = tuple(str(item).strip() for item in self.sensor_ids)
        object.__setattr__(self, "sensor_ids", ids)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if not 11 <= self.radio_channel <= 25:
            raise ValueError("Awinda radio_channel must be in [11, 25]")
        if self.sample_rate_hz <= 0 or self.expected_device_count != 3:
            raise ValueError("Awinda requires a positive rate and exactly three MTw devices")
        if ids and (len(ids) != 3 or len(set(ids)) != 3 or any(not item for item in ids)):
            raise ValueError("sensor_ids must be empty or contain three unique real device IDs")
        if self.wait_timeout_s <= 0 or self.stable_wait_s < 0 or self.poll_interval_s <= 0:
            raise ValueError("invalid Awinda discovery timing")
        if self.pending_group_limit <= 0 or self.queue_capacity <= 0:
            raise ValueError("pending_group_limit and queue_capacity must be positive")


def _coerce_config(value: XsensAwindaConfig | Mapping[str, Any] | None) -> XsensAwindaConfig:
    if value is None:
        return XsensAwindaConfig()
    if isinstance(value, XsensAwindaConfig):
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
    allowed = XsensAwindaConfig.__dataclass_fields__
    return XsensAwindaConfig(**{key: item for key, item in raw.items() if key in allowed})


# ──────────────────────────────────────────────────────────────
#  Backend protocol
# ──────────────────────────────────────────────────────────────

class AwindaBackend(Protocol):
    device_ids: tuple[str, ...]
    actual_rate_hz: int
    metadata: Mapping[str, Any]

    def connect(
        self,
        on_packet: Callable[[str, Any, int], None],
    ) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def remove_callbacks(self) -> None: ...


# ──────────────────────────────────────────────────────────────
#  XDA  Backend
# ──────────────────────────────────────────────────────────────

class XdaAwindaBackend:
    """Own the vendor XDA objects and expose only copied live packets."""

    def __init__(self, config: XsensAwindaConfig, *, _api_module: Any = None) -> None:
        self.config = config
        self.device_ids: tuple[str, ...] = ()
        self.actual_rate_hz = int(round(config.sample_rate_hz))
        self.metadata: dict[str, Any] = {}
        self._api = _api_module
        self._xda: Any = None
        self._control: Any = None
        self._master: Any = None
        self._master_port: Any = None
        self._devices: list[Any] = []
        self._callback: Any = None
        self._measurement_started = False
        self._radio_enabled = False
        self._all_device_ids: list[str] = []

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

        # 1. scan for WirelessMaster
        ports = xda.XsScanner_scanPorts()
        master_port = xda.XsPortInfo()
        for index in range(ports.size()):
            candidate = ports[index]
            if candidate.deviceId().isWirelessMaster():
                master_port = candidate
                break
        if master_port.empty():
            raise AdapterError("未找到 Xsens Awinda WirelessMaster，请检查 USB 连接。")
        self._master_port = master_port

        master_id = master_port.deviceId()
        master_id_str = master_id.toXsString()
        try:
            master_product = master_id.productCode()  # may not exist on all SDK versions
        except (AttributeError, TypeError):
            master_product = None
        master_port_name = master_port.portName()

        # 2. open port
        if not control.openPort(master_port_name, master_port.baudrate()):
            raise AdapterError(f"无法打开 Awinda 端口 {master_port_name}")

        master = control.device(master_id)
        if not master:
            raise AdapterError("无法取得 Awinda WirelessMaster 设备对象")
        self._master = master

        # 3. goto config
        if not master.gotoConfig():
            raise AdapterError("Awinda 无法进入配置模式")

        # 4. enable radio
        if not master.enableRadio(self.config.radio_channel):
            raise AdapterError(f"Awinda 无法开启无线信道 {self.config.radio_channel}")
        self._radio_enabled = True

        # 5. discover MTw children
        target_ids = self.config.sensor_ids  # tuple of 3 str, or empty
        deadline = monotonic() + self.config.wait_timeout_s
        first_seen_at: float | None = None
        all_devices: list[Any] = []
        discovered_map: dict[str, Any] = {}

        while monotonic() < deadline:
            all_devices = list(master.children())
            if all_devices and first_seen_at is None:
                first_seen_at = monotonic()
            discovered_map = {
                str(d.deviceId().toXsString()): d for d in all_devices
            }

            if target_ids:
                # Wait until all target IDs appear, then hold stable
                found_targets = all(
                    any(tid in did for did in discovered_map)
                    for tid in target_ids
                )
                if found_targets and first_seen_at is not None:
                    if monotonic() - first_seen_at >= self.config.stable_wait_s:
                        break
            else:
                # Strict: exactly 3, stable
                if len(all_devices) == 3 and first_seen_at is not None:
                    if monotonic() - first_seen_at >= self.config.stable_wait_s:
                        break

            sleep(self.config.poll_interval_s)

        self._all_device_ids = sorted(discovered_map.keys())

        if target_ids:
            # Select the target 3; warn in health if extras exist
            selected_map: dict[str, Any] = {}
            matched_ordered: list[str] = []
            for tid in target_ids:
                matched = next(
                    (did for did in discovered_map if tid in did), None
                )
                if matched is None:
                    raise AdapterError(
                        f"未发现目标 MTw 设备: {tid}; 已发现: {sorted(discovered_map.keys())}"
                    )
                selected_map[matched] = discovered_map[matched]
                matched_ordered.append(matched)
            extras = sorted(set(discovered_map.keys()) - set(selected_map.keys()))
            if extras:
                self._discovery_warning = (
                    f"目标 3 台之外发现额外 MTw 设备: {extras}，已忽略"
                )
            else:
                self._discovery_warning = None
            ordered_ids = tuple(matched_ordered)
            self._devices = [selected_map[did] for did in ordered_ids]
        else:
            if len(all_devices) != 3:
                raise AdapterError(
                    f"Awinda 需要 3 台 MTw，实际发现 {len(all_devices)} 台。"
                )
            ordered_ids = tuple(sorted(discovered_map.keys()))
            self._devices = [discovered_map[did] for did in ordered_ids]
            self._discovery_warning = None

        self.device_ids = tuple(ordered_ids)

        # 6. supported rates and configure rate
        supported_raw = master.supportedUpdateRates()
        supported_rates = [int(supported_raw[i]) for i in range(supported_raw.size())]
        requested = int(round(self.config.sample_rate_hz))
        if requested not in supported_rates:
            raise AdapterError(
                f"Awinda 不支持 {requested} Hz；支持值为 {supported_rates}"
            )
        rate_hz_int = requested
        if not master.setUpdateRate(rate_hz_int):
            raise AdapterError(f"Awinda 设置采样率 {rate_hz_int} Hz 失败")
        self.actual_rate_hz = rate_hz_int

        # 7. output configuration: six items including PacketCounter and SampleTimeFine
        output = xda.XsOutputConfigurationArray()
        output.push_back(xda.XsOutputConfiguration(xda.XDI_PacketCounter, rate_hz_int))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_SampleTimeFine, rate_hz_int))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_EulerAngles, rate_hz_int))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_Acceleration, rate_hz_int))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_RateOfTurn, rate_hz_int))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, rate_hz_int))

        # 8. set output config and register callback for each device
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
        for dev in self._devices:
            dev.addCallbackHandler(callback)
            if not dev.setOutputConfiguration(output):
                raise AdapterError(
                    f"MTw {dev.deviceId().toXsString()} 配置输出失败"
                )

        # 9. capture metadata for configuration snapshot
        self.metadata = {
            "master_device_id": master_id_str,
            "master_port": master_port_name,
            "radio_channel": self.config.radio_channel,
            "actual_sample_rate_hz": rate_hz_int,
            "supported_update_rates_hz": supported_rates,
            "device_ids": list(ordered_ids),
            "all_discovered_device_ids": self._all_device_ids,
            "discovery_target_ids": list(target_ids) if target_ids else [],
            "expected_device_count": self.config.expected_device_count,
            "pending_group_limit": self.config.pending_group_limit,
            "queue_capacity": self.config.queue_capacity,
        }
        if master_product is not None:
            self.metadata["master_product_code"] = str(master_product)
        try:
            sdk_version = getattr(xda, "XsVersion", None)
            if sdk_version is not None:
                full = sdk_version()
                self.metadata["sdk_version"] = full.toXsString()
        except Exception:
            pass
        if self._discovery_warning:
            self.metadata["discovery_warning"] = self._discovery_warning

    def remove_callbacks(self) -> None:
        if self._callback is not None:
            for dev in self._devices:
                try:
                    dev.removeCallbackHandler(self._callback)
                except BaseException:
                    pass

    def start(self) -> None:
        if self._master is None or not self._master.gotoMeasurement():
            raise AdapterError("Awinda 无法进入测量模式")
        self._measurement_started = True

    def stop(self) -> None:
        if self._master is not None and self._measurement_started:
            if not self._master.gotoConfig():
                raise AdapterError("Awinda 停止时无法返回配置模式")
            self._measurement_started = False

    def close(self) -> None:
        first_error: BaseException | None = None

        # 1. stop measurement
        try:
            self.stop()
        except BaseException:
            if first_error is None:
                first_error = AdapterError(
                    f"Awinda stop error: {traceback.format_exc()}"
                )

        # 2. remove callback handlers
        try:
            self.remove_callbacks()
        except BaseException:
            if first_error is None:
                first_error = AdapterError(
                    f"Awinda removeCallbacks error: {traceback.format_exc()}"
                )

        # 3. disable radio (best-effort, ignore if master is already None)
        if self._master is not None and self._radio_enabled:
            try:
                self._master.disableRadio()
            except BaseException:
                if first_error is None:
                    first_error = AdapterError(
                        f"Awinda disableRadio error: {traceback.format_exc()}"
                    )
            finally:
                self._radio_enabled = False

        # 4. close port
        if self._control is not None and self._master_port is not None:
            try:
                if not self._master_port.empty():
                    self._control.closePort(self._master_port.portName())
            except BaseException:
                if first_error is None:
                    first_error = AdapterError(
                        f"Awinda closePort error: {traceback.format_exc()}"
                    )

        # 5. close control
        if self._control is not None:
            try:
                self._control.close()
            except BaseException:
                if first_error is None:
                    first_error = AdapterError(
                        f"Awinda control.close error: {traceback.format_exc()}"
                    )

        self._callback = None
        self._devices = []
        self._master = None
        self._master_port = None
        self._control = None

        if first_error is not None:
            raise first_error


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _PendingGroup:
    rows: dict[str, np.ndarray]
    host_times: dict[str, int]
    device_times: dict[str, int | None]  # sampleTimeFine or packetCounter
    device_counters: dict[str, int | None]  # packetCounter


def _vector3(value: Any) -> tuple[float, float, float]:
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (IndexError, TypeError):
        try:
            return float(value.x()), float(value.y()), float(value.z())
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterError("Xsens 向量不是可识别的三轴数据") from exc


def parse_xsens_packet(packet: Any) -> np.ndarray:
    """Parse the twelve fields proven by the legacy Awinda implementation."""

    if not packet.containsCalibratedData():
        raise AdapterError("Xsens 数据包缺少 calibrated data")
    if not packet.containsOrientation():
        raise AdapterError("Xsens 数据包缺少 Euler orientation")
    acceleration = _vector3(packet.calibratedAcceleration())
    gyroscope = _vector3(packet.calibratedGyroscopeData())
    magnetic = _vector3(packet.calibratedMagneticField())
    euler = packet.orientationEuler()
    orientation = (float(euler.x()), float(euler.y()), float(euler.z()))
    values = np.asarray((*acceleration, *gyroscope, *magnetic, *orientation), dtype=np.float32)
    if values.shape != (len(IMU_CHANNELS),) or not np.isfinite(values).all():
        raise AdapterError("Xsens 数据包包含无效或非有限数值")
    return values


def _read_optional_packet_counter(packet: Any) -> int | None:
    """Return PacketCounter iff containsPacketCounter() is True; otherwise None."""
    try:
        if packet.containsPacketCounter():
            return int(packet.packetCounter())
    except Exception:
        pass
    return None


def _read_optional_sample_time_fine(packet: Any) -> int | None:
    """Return SampleTimeFine iff containsSampleTimeFine() is True; otherwise None."""
    try:
        if packet.containsSampleTimeFine():
            return int(packet.sampleTimeFine())
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
#  Adapter
# ──────────────────────────────────────────────────────────────

_COUNTER_WRAP_MOD = 65536  # Xsens MTw packet counter is uint16


class XsensAwindaImuAdapter(QueuedHardwareAdapter):
    """Adapter with callback→packet-queue→consumer pipeline.

    Callback thread: copy XsDataPacket, push (device_id, packet, host_ns)
    to bounded ``_packet_queue`` (fault on full).

    Consumer thread: parse 12 fields, align by PacketCounter
    (preferred) or per-device arrival index (fallback when
    containsPacketCounter() is false), publish via raw queue.
    """

    def __init__(
        self,
        config: XsensAwindaConfig | Mapping[str, Any] | None = None,
        *,
        backend: AwindaBackend | None = None,
    ) -> None:
        self._config = _coerce_config(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or XdaAwindaBackend(self._config)
        self._sensor_ids: tuple[str, ...] = self._config.sensor_ids
        self._use_device_ids: tuple[str, ...] = ()

        self._packet_queue: Queue[tuple[str, Any, int | None]] = Queue(
            maxsize=self._config.queue_capacity
        )
        self._consumer_thread: Thread | None = None
        self._consumer_stop = Event()

        self._pending: OrderedDict[int, _PendingGroup] = OrderedDict()
        self._pending_lock = Lock()
        self._fallback_counters: dict[str, int] = {}

        self._sample_index = 0
        self._batch_sequence = 0
        self._incomplete_samples = 0
        self._malformed_packets = 0
        self._duplicate_packets = 0
        self._counter_gaps = 0
        self._max_arrival_spread_ns = 0
        self._max_device_time_spread = 0
        self._counter_has_data = False
        self._time_has_data = False
        self._accepting_packets = False
        self._alignment_mode = "packet_counter_or_per_device_arrival_index"
        self._last_common_counter: dict[str, int] = {}
        self._last_device_counter: dict[str, int] = {}

    # ── descriptor / snapshot ──────────────────────────────

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        ids = self._use_device_ids or self._sensor_ids or tuple(
            f"unassigned_{index + 1}" for index in range(3)
        )
        backend_meta = dict(getattr(self._backend, "metadata", {}))
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="imu",
            display_name="Xsens Awinda 3-MTw array",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=IMU_CHANNELS,
            units=IMU_UNITS,
            nominal_rate_hz=float(
                getattr(self._backend, "actual_rate_hz", cfg.sample_rate_hz)
            ),
            sample_shape=(3, len(IMU_CHANNELS)),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": False,
                "manufacturer": "Xsens",
                "system": "Awinda WirelessMaster/MTw",
                "device_ids": list(ids),
                "physical_location_mapping": "configured" if cfg.sensor_ids else "unassigned",
                "alignment_mode": self._alignment_mode,
                "expected_device_count": 3,
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
            "actual_rate_hz": getattr(self._backend, "actual_rate_hz", None),
            **{k: v for k, v in backend_meta.items()
               if k not in {"pending_group_limit", "queue_capacity"}},
            "alignment_mode": self._alignment_mode,
        }

    # ── lifecycle hooks ────────────────────────────────────

    def _connect_hardware(self) -> None:
        self._backend.connect(self._on_packet)
        ids = tuple(self._backend.device_ids)
        if len(ids) != 3 or len(set(ids)) != 3:
            raise AdapterError(f"Awinda 后端必须提供 3 个唯一 MTw ID，实际为 {ids}")
        self._use_device_ids = ids

    def _reset_trial_state(self) -> None:
        with self._pending_lock:
            self._pending.clear()
            self._fallback_counters = {
                device_id: 0 for device_id in self._use_device_ids
            }
        self._sample_index = 0
        self._batch_sequence = 0
        self._incomplete_samples = 0
        self._malformed_packets = 0
        self._duplicate_packets = 0
        self._counter_gaps = 0
        self._max_arrival_spread_ns = 0
        self._max_device_time_spread = 0
        self._counter_has_data = False
        self._time_has_data = False
        self._last_common_counter = {}
        self._last_device_counter = {}
        while not self._packet_queue.empty():
            try:
                self._packet_queue.get_nowait()
            except Empty:
                break

    def _start_hardware(self) -> None:
        self._accepting_packets = True
        self._consumer_stop.clear()
        for did in self._use_device_ids:
            self._last_device_counter[did] = -1
        self._consumer_thread = Thread(
            target=self._consumer_loop,
            name="xsens-awinda-consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        self._backend.start()

    def _stop_hardware(self) -> None:
        # 1. stop accepting new packets from callback
        self._accepting_packets = False

        # 2. transition hardware to config mode
        try:
            self._backend.stop()
        except BaseException:
            pass

        # 3. remove callback handlers to prevent late callback races
        try:
            self._backend.remove_callbacks()
        except BaseException:
            pass

        # 4. signal consumer thread to stop and drain remaining
        self._consumer_stop.set()
        try:
            # Drain remaining packets from packet_queue
            while True:
                try:
                    item = self._packet_queue.get_nowait()
                    self._process_one_packet(*item)
                except Empty:
                    break
        except BaseException:
            pass

        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=3.0)
            if self._consumer_thread.is_alive():
                self._set_fault(
                    AdapterError("Awinda 消费线程未能在 3 秒内停止")
                )
        self._consumer_thread = None

        # 5. flush any incomplete pending groups
        with self._pending_lock:
            for group in self._pending.values():
                self._incomplete_samples += max(0, 3 - len(group.rows))
            self._pending.clear()

    def _close_hardware(self) -> None:
        self._backend.close()

    # ── callback entry point ───────────────────────────────

    def _on_packet(
        self, device_id: str, packet: Any, host_ns: int | None = None
    ) -> None:
        """Called from XDA callback thread. Copy and enqueue immediately."""
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
                f"Awinda packet queue overflow (capacity={self._config.queue_capacity})"
            )
            self._set_fault(error)

    # ── consumer loop ──────────────────────────────────────

    def _consumer_loop(self) -> None:
        """Thread: read packet_queue, parse, align, publish."""
        while not self._consumer_stop.is_set():
            try:
                item = self._packet_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                self._process_one_packet(*item)
            except BaseException as exc:
                self._set_fault(exc)

    def _process_one_packet(
        self, device_id: str, packet: Any, host_ns: int
    ) -> None:
        """Parse and align a single copied packet."""
        try:
            row = parse_xsens_packet(packet)
        except BaseException:
            self._malformed_packets += 1
            return

        # Read counter / sampleTime with contains* guards
        counter = _read_optional_packet_counter(packet)
        sample_time = _read_optional_sample_time_fine(packet)

        if counter is not None:
            self._counter_has_data = True
        if sample_time is not None:
            self._time_has_data = True

        with self._pending_lock:
            if counter is not None:
                # Use PacketCounter for alignment
                key = counter
                self._last_device_counter[device_id] = counter
            else:
                # Fallback: per-device arrival index
                fallback = self._fallback_counters[device_id]
                self._fallback_counters[device_id] = fallback + 1
                key = fallback

            group = self._pending.setdefault(
                key,
                _PendingGroup(
                    rows={},
                    host_times={},
                    device_times={},
                    device_counters={},
                ),
            )

            if device_id in group.rows:
                # Duplicate device in same counter group
                self._duplicate_packets += 1
                return

            group.rows[device_id] = row
            group.host_times[device_id] = host_ns
            group.device_times[device_id] = sample_time if sample_time is not None else counter
            group.device_counters[device_id] = counter

            complete = len(group.rows) == 3
            if complete:
                self._pending.pop(key)
                # Detect common counter gap
                self._record_counter_gap(key, group)

            # Evict stale groups beyond limit
            while len(self._pending) > self._config.pending_group_limit:
                _old_key, old_group = self._pending.popitem(last=False)
                self._incomplete_samples += max(0, 3 - len(old_group.rows))

        if complete:
            self._emit_group(group)

    def _record_counter_gap(self, key: int, group: _PendingGroup) -> None:
        """Record gap if common counter jumped more than 1."""
        if not self._counter_has_data:
            return

        # Check all three devices have counters
        all_counters = [group.device_counters[did] for did in self._use_device_ids]
        if any(c is None for c in all_counters):
            return

        # All three should share the same counter key (key itself)
        for did in self._use_device_ids:
            prev = self._last_device_counter.get(did)
            cur = group.device_counters[did]
            if prev is not None and prev >= 0 and cur is not None:
                expected = (prev + 1) % _COUNTER_WRAP_MOD
                if cur != expected and cur != prev:
                    self._counter_gaps += 1
                    break
                self._last_device_counter[did] = cur

        # Check consistent counter across all three devices
        counters = [c for c in all_counters if c is not None]
        if len(counters) == 3:
            last = self._last_common_counter.get("value")
            if last is not None and len(set(counters)) == 1:
                cur_val = counters[0]
                expected = (last + 1) % _COUNTER_WRAP_MOD
                if cur_val != expected and cur_val != last:
                    self._counter_gaps += 1
            if len(set(counters)) == 1:
                self._last_common_counter["value"] = counters[0]

    def _emit_group(self, group: _PendingGroup) -> None:
        """Emit a complete (N,3,12) SampleBatch for one aligned trio."""
        data = np.ascontiguousarray(
            np.stack(
                [group.rows[device_id] for device_id in self._use_device_ids],
                axis=0,
            )[None, ...],
            dtype=np.float32,
        )

        host_times_list = [group.host_times[device_id] for device_id in self._use_device_ids]
        host_ns = min(host_times_list)

        # Track spreads for health
        arrival_spread = max(host_times_list) - min(host_times_list)
        if arrival_spread > self._max_arrival_spread_ns:
            self._max_arrival_spread_ns = arrival_spread

        device_times_list = [
            group.device_times[device_id] for device_id in self._use_device_ids
        ]
        non_null_times = [t for t in device_times_list if t is not None]
        device_time: int | float | None = non_null_times[0] if non_null_times else None

        if len(non_null_times) == 3:
            time_spread = abs(max(non_null_times) - min(non_null_times))
            self._max_device_time_spread = max(self._max_device_time_spread, time_spread)

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
            device_timestamp=device_time,
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
            pending_groups = len(self._pending)
        return {
            "pending_alignment_groups": pending_groups,
            "incomplete_sensor_samples": self._incomplete_samples,
            "malformed_packets": self._malformed_packets,
            "duplicate_packets": self._duplicate_packets,
            "counter_gaps": self._counter_gaps,
            "max_arrival_spread_ns": self._max_arrival_spread_ns,
            "max_device_time_spread": self._max_device_time_spread,
            "counter_source_available": self._counter_has_data,
            "sample_time_fine_available": self._time_has_data,
            "alignment_mode": self._alignment_mode,
            "resolved_sensor_ids": ",".join(
                self._use_device_ids or self._sensor_ids
            ),
            "packet_queue_size": self._packet_queue.qsize(),
        }


__all__ = [
    "AwindaBackend",
    "XdaAwindaBackend",
    "XsensAwindaConfig",
    "XsensAwindaImuAdapter",
    "parse_xsens_packet",
]
