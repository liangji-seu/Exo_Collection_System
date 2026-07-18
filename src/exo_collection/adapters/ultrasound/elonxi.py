"""Elonxi four-channel A-mode ultrasound adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from threading import Event
from time import perf_counter_ns, sleep, time_ns
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import FrameBatch


@dataclass(frozen=True, slots=True)
class ElonxiUltrasoundConfig:
    device_id: str = "ultrasound_elonxi"
    clock_domain: str = "ultrasound_elonxi_clock"
    sdk_path: str | None = None
    device_ip: str | None = None
    port: int = 1430
    channels: tuple[int, ...] = (1, 2, 3, 4)
    samples_per_channel: int = 1000
    nominal_rate_hz: float = 20.0
    queue_capacity: int = 64
    discovery_timeout_s: float = 10.0
    device_on_delay_s: float = 2.0
    configuration_delay_s: float = 1.0
    collection_delay_s: float = 0.5
    stop_delay_s: float = 0.3
    device_off_delay_s: float = 0.5

    def __post_init__(self) -> None:
        channels = tuple(int(value) for value in self.channels)
        object.__setattr__(self, "channels", channels)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if channels != (1, 2, 3, 4):
            raise ValueError("Elonxi collection requires channels 1,2,3,4")
        if self.samples_per_channel <= 0 or self.nominal_rate_hz <= 0:
            raise ValueError("sample count and nominal rate must be positive")
        if self.queue_capacity <= 0 or self.port <= 0 or self.port > 65535:
            raise ValueError("invalid queue capacity or UDP port")
        for value in (
            self.discovery_timeout_s,
            self.device_on_delay_s,
            self.configuration_delay_s,
            self.collection_delay_s,
            self.stop_delay_s,
            self.device_off_delay_s,
        ):
            if value < 0:
                raise ValueError("timeouts and delays must be non-negative")


def _coerce_config(
    value: ElonxiUltrasoundConfig | Mapping[str, Any] | None,
) -> ElonxiUltrasoundConfig:
    if value is None:
        return ElonxiUltrasoundConfig()
    if isinstance(value, ElonxiUltrasoundConfig):
        return value
    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw.pop("id")
    if "sdk_directory" in raw and "sdk_path" not in raw:
        raw["sdk_path"] = raw.pop("sdk_directory")
    if isinstance(raw.get("channels"), str):
        raw["channels"] = tuple(
            int(part.strip()) for part in str(raw["channels"]).split(",") if part.strip()
        )
    allowed = ElonxiUltrasoundConfig.__dataclass_fields__
    return ElonxiUltrasoundConfig(
        **{key: item for key, item in raw.items() if key in allowed}
    )


class ElonxiBackend(Protocol):
    resolved_device_ip: str | None

    def connect(
        self,
        on_ultrasound: Callable[[Any], None],
        on_rel_data: Callable[[Any, Any], None],
        on_notification: Callable[[Any, Any], None],
    ) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def discover_elonxi_ip(timeout_s: float) -> str | None:
    """Discover the first `_http._udp.local.` endpoint without global state."""

    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except ImportError as exc:
        raise AdapterError(
            "未安装 zeroconf，且未在界面指定 Elonxi 设备 IP。"
        ) from exc

    found = Event()
    addresses: list[str] = []

    class Listener(ServiceListener):
        def add_service(self, zeroconf: Any, service_type: str, name: str) -> None:
            info = zeroconf.get_service_info(service_type, name)
            if info is not None:
                addresses.extend(info.parsed_addresses())
                if addresses:
                    found.set()

        def update_service(self, zeroconf: Any, service_type: str, name: str) -> None:
            self.add_service(zeroconf, service_type, name)

        def remove_service(self, zeroconf: Any, service_type: str, name: str) -> None:
            return None

    zeroconf = Zeroconf()
    browser = ServiceBrowser(zeroconf, "_http._udp.local.", Listener())
    try:
        found.wait(timeout=timeout_s)
        return addresses[0] if addresses else None
    finally:
        browser.cancel()
        zeroconf.close()


class PythonNetElonxiBackend:
    """Thin owner of the exact .NET API used by the validated first system."""

    def __init__(self, config: ElonxiUltrasoundConfig) -> None:
        self.config = config
        self.resolved_device_ip: str | None = None
        self._newsletter: Any = None
        self._global_events: Any = None
        self._callbacks: tuple[Any, Any, Any] | None = None
        self._collecting = False
        self._device_on = False

    def connect(
        self,
        on_ultrasound: Callable[[Any], None],
        on_rel_data: Callable[[Any, Any], None],
        on_notification: Callable[[Any, Any], None],
    ) -> None:
        cfg = self.config
        if not cfg.sdk_path:
            raise AdapterError("请先在真实设备设置中选择 Elonxi SDK 目录。")
        sdk_path = Path(cfg.sdk_path).expanduser().resolve()
        dll_path = sdk_path / "Elonxi_SDK.dll" if sdk_path.is_dir() else sdk_path
        if not dll_path.is_file():
            raise AdapterError(f"未找到 Elonxi SDK DLL: {dll_path}")
        try:
            from pythonnet import load

            load("coreclr")
            import clr

            clr.AddReference("System.Collections")
            clr.AddReference(str(dll_path.with_suffix("")))
            sdk = importlib.import_module("Elonxi_SDK")
        except BaseException as exc:
            raise AdapterError(
                f"加载 Elonxi .NET SDK 失败（{dll_path}）: {exc}"
            ) from exc

        device_ip = cfg.device_ip or discover_elonxi_ip(cfg.discovery_timeout_s)
        if not device_ip:
            raise AdapterError("未发现 Elonxi 超声设备；请检查网络或在界面指定 IP。")
        self.resolved_device_ip = device_ip
        global_events = sdk.GlobalEvents
        try:
            global_events.NotificationReceived += on_notification
            global_events.RealRealUltrDataReceived += on_ultrasound
            global_events.RealRealRelDataReceived += on_rel_data
            self._callbacks = (on_notification, on_ultrasound, on_rel_data)
            self._global_events = global_events
            self._newsletter = sdk.Newsletter(cfg.port, device_ip, cfg.port)
            self._newsletter.deviceSwitch(True)
            self._device_on = True
            sleep(cfg.device_on_delay_s)
            self._newsletter.configParam(
                ",".join(str(channel) for channel in cfg.channels),
                "",
                "",
                0,
                0,
                False,
            )
            sleep(cfg.configuration_delay_s)
        except BaseException as exc:
            self.close()
            raise AdapterError(f"初始化 Elonxi 超声设备失败: {exc}") from exc

    def start(self) -> None:
        if self._newsletter is None:
            raise AdapterError("Elonxi Newsletter 尚未初始化")
        self._newsletter.collectionSwitch(True)
        self._collecting = True
        sleep(self.config.collection_delay_s)

    def stop(self) -> None:
        if self._newsletter is not None and self._collecting:
            self._newsletter.collectionSwitch(False)
            self._collecting = False
            sleep(self.config.stop_delay_s)

    def close(self) -> None:
        try:
            self.stop()
        except BaseException:
            pass
        if self._newsletter is not None and self._device_on:
            try:
                self._newsletter.deviceSwitch(False)
                sleep(self.config.device_off_delay_s)
            finally:
                self._device_on = False
        events, callbacks = self._global_events, self._callbacks
        if events is not None and callbacks is not None:
            notification, ultrasound, rel_data = callbacks
            try:
                events.NotificationReceived -= notification
                events.RealRealUltrDataReceived -= ultrasound
                events.RealRealRelDataReceived -= rel_data
            except BaseException:
                pass
        self._callbacks = None
        self._global_events = None
        self._newsletter = None


class ElonxiUltrasoundAdapter(QueuedHardwareAdapter):
    def __init__(
        self,
        config: ElonxiUltrasoundConfig | Mapping[str, Any] | None = None,
        *,
        backend: ElonxiBackend | None = None,
    ) -> None:
        self._config = _coerce_config(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or PythonNetElonxiBackend(self._config)
        self._sequence = 0
        self._frame_index = 0
        self._current_packet_number: int | None = None
        self._malformed_callbacks = 0

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="ultrasound",
            display_name="Elonxi four-channel A-mode ultrasound",
            clock_domain=cfg.clock_domain,
            event_kind="frame_batch",
            channels=tuple(f"ch_{channel}" for channel in cfg.channels),
            units=("a.u.",) * len(cfg.channels),
            nominal_rate_hz=cfg.nominal_rate_hz,
            sample_shape=(len(cfg.channels), cfg.samples_per_channel),
            dtype=np.dtype(np.uint16).str,
            metadata={
                "simulated": False,
                "manufacturer": "Elonxi",
                "geometry": "a_line",
                "channels": list(cfg.channels),
                "frame_shape": [len(cfg.channels), cfg.samples_per_channel],
                "resolved_device_ip": self._backend.resolved_device_ip,
                "device_timestamp": "latest RealRealRelDataReceived pack number when available",
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        snapshot = asdict(self._config)
        snapshot["resolved_device_ip"] = self._backend.resolved_device_ip
        return snapshot

    def _connect_hardware(self) -> None:
        self._backend.connect(
            self._on_ultrasound,
            self._on_rel_data,
            self._on_notification,
        )

    def _reset_trial_state(self) -> None:
        self._sequence = 0
        self._frame_index = 0
        self._current_packet_number = None
        self._malformed_callbacks = 0

    def _start_hardware(self) -> None:
        self._backend.start()

    def _stop_hardware(self) -> None:
        self._backend.stop()

    def _close_hardware(self) -> None:
        self._backend.close()

    def _on_notification(self, packet_type: Any, message: Any) -> None:
        # Notifications are vendor control telemetry.  Do not write or log
        # opaque content from the SDK on its callback thread.
        return None

    def _on_rel_data(self, is_ultrasound: Any, packet_number: Any) -> None:
        if bool(is_ultrasound):
            try:
                self._current_packet_number = int(packet_number)
            except (TypeError, ValueError):
                self._current_packet_number = None

    def _on_ultrasound(self, data_by_channel: Any) -> None:
        host_ns = perf_counter_ns()
        try:
            mapping = dict(data_by_channel.items())
            normalized: dict[int, list[Any]] = {
                int(channel): list(waveforms)
                for channel, waveforms in mapping.items()
            }
            expected = set(self._config.channels)
            if set(normalized) != expected:
                raise AdapterError(
                    f"Elonxi 回调通道不完整：expected={sorted(expected)}, "
                    f"received={sorted(normalized)}"
                )
            frame_counts = {len(normalized[channel]) for channel in self._config.channels}
            if len(frame_counts) != 1 or not frame_counts or next(iter(frame_counts)) <= 0:
                raise AdapterError("Elonxi 四通道回调的帧数不一致或为空")
            frame_count = next(iter(frame_counts))
            frames = np.empty(
                (frame_count, len(self._config.channels), self._config.samples_per_channel),
                dtype=np.uint16,
            )
            for channel_index, channel in enumerate(self._config.channels):
                for frame_offset, waveform in enumerate(normalized[channel]):
                    values = np.asarray(list(waveform))
                    if values.shape != (self._config.samples_per_channel,):
                        raise AdapterError(
                            f"Elonxi 通道 {channel} 波形长度 {values.size}，"
                            f"应为 {self._config.samples_per_channel}"
                        )
                    if not np.issubdtype(values.dtype, np.number):
                        raise AdapterError(f"Elonxi 通道 {channel} 包含非数值采样")
                    numeric = values.astype(np.float64, copy=False)
                    if (
                        not np.isfinite(numeric).all()
                        or np.any(numeric < 0)
                        or np.any(numeric > np.iinfo(np.uint16).max)
                        or np.any(numeric != np.floor(numeric))
                    ):
                        raise AdapterError(f"Elonxi 通道 {channel} 采样无法无损转换为 uint16")
                    frames[frame_offset, channel_index] = numeric.astype(np.uint16)
            frames = np.ascontiguousarray(frames)
            event = FrameBatch(
                session_uuid=(
                    str(self._trial.session_uuid)
                    if self._trial is not None and self._trial.session_uuid is not None
                    else None
                ),
                trial_uuid=str(self._trial.trial_uuid) if self._trial is not None else None,
                device_id=self._config.device_id,
                modality="ultrasound",
                clock_domain=self._config.clock_domain,
                host_monotonic_ns=host_ns,
                host_utc_ns=time_ns(),
                first_frame_index=self._frame_index,
                frame_count=frame_count,
                sequence_number=self._sequence,
                device_timestamp=self._current_packet_number,
                frame_rate_hz=self._config.nominal_rate_hz,
                data=frames,
            )
            self._publish_raw(event, item_count=frame_count, host_monotonic_ns=host_ns)
            self._frame_index += frame_count
            self._sequence += 1
        except BaseException as exc:
            self._malformed_callbacks += 1
            self._set_fault(exc)

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "malformed_callbacks": self._malformed_callbacks,
            "resolved_device_ip": self._backend.resolved_device_ip,
        }


__all__ = [
    "ElonxiBackend",
    "ElonxiUltrasoundAdapter",
    "ElonxiUltrasoundConfig",
    "PythonNetElonxiBackend",
    "discover_elonxi_ip",
]
