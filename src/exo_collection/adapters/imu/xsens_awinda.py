"""Xsens Awinda WirelessMaster and three-MTw hardware adapter."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from threading import Lock
from time import monotonic, perf_counter_ns, sleep, time_ns
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.domain.events import SampleBatch


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


class AwindaBackend(Protocol):
    device_ids: tuple[str, ...]
    actual_rate_hz: float

    def connect(
        self,
        on_packet: Callable[[str, Any, int], None],
    ) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class XdaAwindaBackend:
    """Own the vendor XDA objects and expose only copied live packets."""

    def __init__(self, config: XsensAwindaConfig) -> None:
        self.config = config
        self.device_ids: tuple[str, ...] = ()
        self.actual_rate_hz = config.sample_rate_hz
        self._xda: Any = None
        self._control: Any = None
        self._master: Any = None
        self._master_port: Any = None
        self._devices: list[Any] = []
        self._callback: Any = None
        self._measurement_started = False
        self._radio_enabled = False

    def connect(self, on_packet: Callable[[str, Any, int], None]) -> None:
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
        if not control.openPort(master_port.portName(), master_port.baudrate()):
            raise AdapterError(f"无法打开 Awinda 端口 {master_port.portName()}")
        master = control.device(master_port.deviceId())
        if not master:
            raise AdapterError("无法取得 Awinda WirelessMaster 设备对象")
        self._master = master
        if not master.gotoConfig():
            raise AdapterError("Awinda 无法进入配置模式")
        if not master.enableRadio(self.config.radio_channel):
            raise AdapterError(f"Awinda 无法开启无线信道 {self.config.radio_channel}")
        self._radio_enabled = True

        deadline = monotonic() + self.config.wait_timeout_s
        first_seen_at: float | None = None
        devices: list[Any] = []
        while monotonic() < deadline:
            devices = list(master.children())
            if devices and first_seen_at is None:
                first_seen_at = monotonic()
            if (
                first_seen_at is not None
                and monotonic() - first_seen_at >= self.config.stable_wait_s
            ):
                break
            sleep(self.config.poll_interval_s)
        if len(devices) != self.config.expected_device_count:
            raise AdapterError(
                f"Awinda 需要 3 台 MTw，实际发现 {len(devices)} 台。"
            )

        discovered = {
            str(device.deviceId().toXsString()): device for device in devices
        }
        if self.config.sensor_ids:
            missing = set(self.config.sensor_ids) - set(discovered)
            unexpected = set(discovered) - set(self.config.sensor_ids)
            if missing or unexpected:
                raise AdapterError(
                    f"MTw ID 与配置不一致：missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            ordered_ids = self.config.sensor_ids
        else:
            ordered_ids = tuple(sorted(discovered))
        self.device_ids = tuple(ordered_ids)
        self._devices = [discovered[device_id] for device_id in self.device_ids]

        supported = master.supportedUpdateRates()
        supported_rates = [float(supported[index]) for index in range(supported.size())]
        requested = float(self.config.sample_rate_hz)
        matching = next(
            (rate for rate in supported_rates if abs(rate - requested) < 1e-9),
            None,
        )
        if matching is None:
            raise AdapterError(
                f"Awinda 不支持 {requested:g} Hz；支持值为 {supported_rates}"
            )
        if not master.setUpdateRate(int(matching) if matching.is_integer() else matching):
            raise AdapterError(f"Awinda 设置采样率 {matching:g} Hz 失败")
        self.actual_rate_hz = matching

        output = xda.XsOutputConfigurationArray()
        output.push_back(xda.XsOutputConfiguration(xda.XDI_EulerAngles, matching))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_Acceleration, matching))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_RateOfTurn, matching))
        output.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, matching))

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
        for device in self._devices:
            device.addCallbackHandler(callback)
            if not device.setOutputConfiguration(output):
                raise AdapterError(
                    f"MTw {device.deviceId().toXsString()} 配置输出失败"
                )

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
        try:
            self.stop()
        except BaseException:
            pass
        if self._callback is not None:
            for device in self._devices:
                try:
                    device.removeCallbackHandler(self._callback)
                except BaseException:
                    pass
        if self._master is not None and self._radio_enabled:
            try:
                self._master.disableRadio()
            finally:
                self._radio_enabled = False
        if self._control is not None and self._master_port is not None:
            try:
                if not self._master_port.empty():
                    self._control.closePort(self._master_port.portName())
            finally:
                self._control.close()
        self._callback = None
        self._devices = []
        self._master = None
        self._master_port = None
        self._control = None


@dataclass(slots=True)
class _PendingGroup:
    rows: dict[str, np.ndarray]
    host_times: dict[str, int]
    device_times: dict[str, int | float | None]


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


def _optional_packet_number(packet: Any) -> int | None:
    method = getattr(packet, "packetCounter", None)
    if callable(method):
        try:
            return int(method())
        except (TypeError, ValueError):
            return None
    return None


def _optional_sample_time(packet: Any) -> int | None:
    method = getattr(packet, "sampleTimeFine", None)
    if callable(method):
        try:
            return int(method())
        except (TypeError, ValueError):
            return None
    return None


class XsensAwindaImuAdapter(QueuedHardwareAdapter):
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
        self._pending: OrderedDict[tuple[str, int], _PendingGroup] = OrderedDict()
        self._pending_lock = Lock()
        self._fallback_counters: dict[str, int] = {}
        self._sample_index = 0
        self._batch_sequence = 0
        self._incomplete_samples = 0
        self._malformed_packets = 0
        self._alignment_mode = "packet_counter_or_per_device_arrival_index"

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        ids = self._sensor_ids or tuple(f"unassigned_{index + 1}" for index in range(3))
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="imu",
            display_name="Xsens Awinda 3-MTw array",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=IMU_CHANNELS,
            units=IMU_UNITS,
            nominal_rate_hz=float(getattr(self._backend, "actual_rate_hz", cfg.sample_rate_hz)),
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
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {
            **asdict(self._config),
            "resolved_sensor_ids": list(self._sensor_ids),
            "actual_rate_hz": getattr(self._backend, "actual_rate_hz", None),
        }

    def _connect_hardware(self) -> None:
        self._backend.connect(self._on_packet)
        ids = tuple(self._backend.device_ids)
        if len(ids) != 3 or len(set(ids)) != 3:
            raise AdapterError(f"Awinda 后端必须提供 3 个唯一 MTw ID，实际为 {ids}")
        self._sensor_ids = ids

    def _reset_trial_state(self) -> None:
        with self._pending_lock:
            self._pending.clear()
            self._fallback_counters = {device_id: 0 for device_id in self._sensor_ids}
        self._sample_index = 0
        self._batch_sequence = 0
        self._incomplete_samples = 0
        self._malformed_packets = 0

    def _start_hardware(self) -> None:
        self._backend.start()

    def _stop_hardware(self) -> None:
        self._backend.stop()
        with self._pending_lock:
            for group in self._pending.values():
                self._incomplete_samples += max(0, 3 - len(group.rows))
            self._pending.clear()

    def _close_hardware(self) -> None:
        self._backend.close()

    def _on_packet(self, device_id: str, packet: Any, host_ns: int | None = None) -> None:
        if device_id not in self._sensor_ids:
            self._set_fault(AdapterError(f"收到未配置 MTw {device_id} 的数据"))
            return
        received_ns = perf_counter_ns() if host_ns is None else int(host_ns)
        try:
            row = parse_xsens_packet(packet)
            packet_number = _optional_packet_number(packet)
            sample_time = _optional_sample_time(packet)
            with self._pending_lock:
                if packet_number is None:
                    fallback = self._fallback_counters[device_id]
                    self._fallback_counters[device_id] = fallback + 1
                    key = ("arrival_index", fallback)
                else:
                    key = ("packet_counter", packet_number)
                group = self._pending.setdefault(
                    key,
                    _PendingGroup(rows={}, host_times={}, device_times={}),
                )
                if device_id in group.rows:
                    raise AdapterError(
                        f"MTw {device_id} 在对齐键 {key} 上出现重复数据包"
                    )
                group.rows[device_id] = row
                group.host_times[device_id] = received_ns
                group.device_times[device_id] = sample_time if sample_time is not None else packet_number
                complete = len(group.rows) == 3
                if complete:
                    self._pending.pop(key)
                while len(self._pending) > self._config.pending_group_limit:
                    _old_key, old_group = self._pending.popitem(last=False)
                    self._incomplete_samples += max(0, 3 - len(old_group.rows))
            if complete:
                self._emit_group(group)
        except BaseException as exc:
            self._malformed_packets += 1
            self._set_fault(exc)

    def _emit_group(self, group: _PendingGroup) -> None:
        data = np.ascontiguousarray(
            np.stack([group.rows[device_id] for device_id in self._sensor_ids], axis=0)[None, ...],
            dtype=np.float32,
        )
        host_ns = min(group.host_times.values())
        device_times = [group.device_times[device_id] for device_id in self._sensor_ids]
        non_null_times = [value for value in device_times if value is not None]
        device_time: int | float | None = non_null_times[0] if non_null_times else None
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

    def _dropped_packets(self) -> int:
        return self._incomplete_samples

    def _sequence_gaps(self) -> int:
        return self._incomplete_samples

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        with self._pending_lock:
            pending_groups = len(self._pending)
        return {
            "pending_alignment_groups": pending_groups,
            "incomplete_sensor_samples": self._incomplete_samples,
            "malformed_packets": self._malformed_packets,
            "alignment_mode": self._alignment_mode,
            "resolved_sensor_ids": ",".join(self._sensor_ids),
        }


__all__ = [
    "AwindaBackend",
    "XdaAwindaBackend",
    "XsensAwindaConfig",
    "XsensAwindaImuAdapter",
    "parse_xsens_packet",
]
