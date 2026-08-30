"""Noraxon Ultium EMG adapter (AcquireCom SDK).

The Noraxon ``Easy2.AcquireCom`` SDK is an in-process COM object model whose
``ThreadingModel`` is ``Apartment``.  Every COM call must therefore happen on a
single apartment-threaded (STA) thread.  This adapter hosts the SDK on one
dedicated sampler thread and publishes ``SampleBatch`` events through the shared
loss-intolerant raw queue, exactly like the other ``QueuedHardwareAdapter``
implementations.

The Noraxon SDK is imported lazily inside that thread so the rest of the
project (including unit tests) never requires ``comtypes`` or the vendor DLL.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, fields, is_dataclass
from time import perf_counter_ns, time_ns
from typing import Any, Iterable, Mapping

import numpy as np

from exo_collection.adapters.base import AdapterError, AdapterState, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import DeviceStatus, HealthStatus, SampleBatch

_log = logging.getLogger(__name__)

# The Noraxon AcquireCom type library.  ``GetModule`` is only called inside the
# STA sampler thread so it never runs at import time.
_NORAXON_TYPELIB = ("{089FD02C-0456-4A18-BB0A-C34D001D93BD}", 1, 0)

# A real Ultium EMG sensor is tagged ``line.noraxon_g3_<serial>`` in addition to
# ``type.input.analog.emg`` and ``device.noraxon.ultium...``.  Replay "player"
# channels carry ``device.player.player.record`` and must be excluded.
_SERIAL_TAG_PREFIX = "line.noraxon_g3_"
_ULTIUM_TAG_PREFIX = "device.noraxon.ultium"
_EMG_FILTER_TAG = "type.input.analog.emg"

_POLL_INTERVAL_S = 0.025
_CONNECT_TIMEOUT_S = 30.0
_JOIN_TIMEOUT_S = 5.0


def _normalise_unit_id(value: Any) -> str:
    """Return the bare sensor serial for a configured unit ID.

    Configurations may name a sensor by its full tag (``line.noraxon_g3_234fc``
    or ``noraxon_g3_234fc``) or by the bare serial (``234fc``).  An empty string
    marks a deliberately unassigned slot.
    """

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    for prefix in ("line.noraxon_g3_", "noraxon_g3_", "line."):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _serial_from_tags(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith(_SERIAL_TAG_PREFIX):
            return tag[len(_SERIAL_TAG_PREFIX):]
    return None


def _ultium_serials_from_components(
    tags_per_component: Iterable[list[str]],
) -> list[str]:
    """Return sorted, deduplicated Ultium serials across component tag lists."""
    serials: list[str] = []
    for tags in tags_per_component:
        if not any(tag.startswith(_ULTIUM_TAG_PREFIX) for tag in tags):
            continue
        serial = _serial_from_tags(tags)
        if serial is not None:
            serials.append(serial)
    return sorted(set(serials))


@dataclass(frozen=True, slots=True)
class NoraxonEmgChannel:
    """One configured muscle/sensor pair."""

    name: str
    unit_id: str


def _coerce_channel(value: Any) -> NoraxonEmgChannel:
    if isinstance(value, NoraxonEmgChannel):
        return value
    if isinstance(value, Mapping):
        return NoraxonEmgChannel(
            name=str(value.get("name", "")).strip(),
            unit_id=str(value.get("unit_id", "")).strip(),
        )
    raise TypeError(
        f"channel must be a mapping or NoraxonEmgChannel, got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class NoraxonEmgConfig:
    device_id: str = "emg_noraxon"
    clock_domain: str = "emg_noraxon_clock"
    sample_rate_hz: float = 4000.0
    unit: str = "µV"
    channels: tuple[NoraxonEmgChannel, ...] = ()
    queue_capacity: int = 512

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if not self.unit.strip():
            raise ValueError("unit must not be empty")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        channels = tuple(_coerce_channel(item) for item in self.channels)
        if not channels:
            raise ValueError("channels must not be empty")
        names = [channel.name for channel in channels]
        if any(not name for name in names):
            raise ValueError("channel muscle names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("channel muscle names must be unique")
        object.__setattr__(self, "channels", channels)


def _coerce_config(value: Any) -> NoraxonEmgConfig:
    if value is None:
        return NoraxonEmgConfig()
    if isinstance(value, NoraxonEmgConfig):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("expected NoraxonEmgConfig or a mapping")
    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw.pop("id")
    allowed = {item.name for item in fields(NoraxonEmgConfig)}
    return NoraxonEmgConfig(**{key: val for key, val in raw.items() if key in allowed})


class _NoraxonSampler(threading.Thread):
    """STA thread that owns the Noraxon COM objects end to end."""

    def __init__(
        self,
        config: NoraxonEmgConfig,
        on_batch: Any,
        on_fault: Any,
    ) -> None:
        super().__init__(name="noraxon-emg-sampler", daemon=True)
        self._config = config
        self._on_batch = on_batch
        self._on_fault = on_fault
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._error: BaseException | None = None
        self.detected_unit_ids: list[str] = []
        self.missing: list[tuple[str, str]] = []  # (muscle_name, unit_id)
        self.matched_indices: list[int] = []
        self.actual_rate_hz: float = float(config.sample_rate_hz)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self._error = exc
            self._connected_event.set()
            try:
                self._on_fault(exc)
            except BaseException:
                pass
        finally:
            self._teardown_com()

    def _run(self) -> None:
        import comtypes
        import comtypes.client
        from comtypes.automation import VARIANT
        from comtypes.safearray import safearray_as_ndarray

        comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
        try:
            try:
                import comtypes.gen.Easy2AcquireCom as sdk
            except ImportError:
                comtypes.client.GetModule(_NORAXON_TYPELIB)
                import comtypes.gen.Easy2AcquireCom as sdk

            dm = comtypes.client.CreateObject(sdk.DeviceManager._reg_clsid_)
            dm.Initialize("")
            dm.ClearLastErrorText()

            def _sdk_error() -> str:
                try:
                    return str(dm.GetLastErrorText()).strip()
                except Exception:
                    return ""

            device = dm.GetCurrentDevice()
            if device is None:
                raise AdapterError(
                    "Noraxon 未找到当前设备：请先在 myoRESEARCH 中选定 Ultium 接收器"
                )

            device.SetComponentFilterTags(_EMG_FILTER_TAG)
            component_count = device.GetComponentCount()

            detected: dict[str, Any] = {}
            with safearray_as_ndarray:
                for index in range(component_count):
                    component = device.GetComponent(index)
                    tags = list(component.GetTags())
                    if not any(tag.startswith(_ULTIUM_TAG_PREFIX) for tag in tags):
                        continue
                    serial = _serial_from_tags(tags)
                    if serial is None:
                        continue
                    detected[serial] = component.QueryInterface(sdk.IAnalogInput)

            self.detected_unit_ids = sorted(detected)

            matched_indices: list[int] = []
            matched_ains: list[Any] = []
            missing: list[tuple[str, str]] = []
            for index, channel in enumerate(self._config.channels):
                serial = _normalise_unit_id(channel.unit_id)
                if serial and serial in detected:
                    matched_indices.append(index)
                    matched_ains.append(detected[serial])
                else:
                    missing.append((channel.name, channel.unit_id))

            self.missing = missing
            self.matched_indices = matched_indices

            if not matched_ains:
                raise AdapterError(
                    "Noraxon EMG：未匹配到任何配置的传感器，无法采集（"
                    f"检测到 {len(detected)} 个 Ultium 传感器："
                    + ", ".join(self.detected_unit_ids)
                    + "）"
                )

            for ain in matched_ains:
                try:
                    ain.Enable()
                except Exception as exc:
                    detail = _sdk_error()
                    raise AdapterError(
                        f"Noraxon EMG 启用传感器失败（Enable）：{exc}"
                        + (f"；SDK 错误：{detail}" if detail else "")
                    ) from exc
                try:
                    ain.RecoveryEnable()
                except Exception:
                    # 数据丢失恢复是可选项（仅对丢包恢复有意义）；不支持的
                    # 传感器不应因此导致整个连接失败。
                    pass

            try:
                device.Activate()
            except Exception as exc:
                detail = _sdk_error()
                raise AdapterError(
                    f"Noraxon EMG 激活设备失败（Activate）：{exc}"
                    + (f"；SDK 错误：{detail}" if detail else "")
                ) from exc

            for ain in matched_ains:
                try:
                    self.actual_rate_hz = float(ain.GetFrequency())
                except Exception:
                    pass
                break

            self._connected_event.set()

            total_channels = len(self._config.channels)
            while not self._stop_event.is_set():
                state = device.Transfer()
                if not (state & sdk.TransferDataReady):
                    self._stop_event.wait(_POLL_INTERVAL_S)
                    continue

                counts: list[int] = []
                with safearray_as_ndarray:
                    for ain in matched_ains:
                        counts.append(int(ain.GetQuantCount()))
                row_count = max(counts) if counts else 0
                if row_count <= 0:
                    continue

                data = np.full((row_count, total_channels), np.nan, dtype=np.float32)
                with safearray_as_ndarray:
                    for local_index, (ain, count) in enumerate(zip(matched_ains, counts)):
                        if count <= 0:
                            continue
                        buffer = VARIANT(np.zeros(count, dtype=np.float64))
                        ain.GetQuants(0, count, buffer, 0)
                        global_column = matched_indices[local_index]
                        data[:count, global_column] = np.asarray(
                            list(buffer.value), dtype=np.float32
                        )
                self._on_batch(data, perf_counter_ns())
        finally:
            try:
                device.Stop()
                device.Deactivate()
            except Exception:
                pass
            comtypes.CoUninitialize()

    def _teardown_com(self) -> None:
        # The balanced ``CoUninitialize`` lives in ``_run``'s ``finally``; this
        # hook only exists so the thread never leaks the apartment on a fatal
        # import error before ``CoInitializeEx``.
        pass


def scan_ultium_units(timeout_s: float = 10.0) -> list[str]:
    """Enumerate the online Noraxon Ultium sensors as bare serials.

    Runs the AcquireCom COM enumeration on a dedicated apartment (STA) thread
    and blocks until it completes or ``timeout_s`` elapses.  Returns the sorted
    serials (e.g. ``["234f5", "234fc"]``); raises :class:`AdapterError` if the
    receiver is unavailable or no Ultium sensor is present.
    """

    outcome: dict[str, Any] = {}

    def _scan() -> None:
        try:
            import comtypes
            import comtypes.client
            from comtypes.safearray import safearray_as_ndarray

            comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
            try:
                try:
                    import comtypes.gen.Easy2AcquireCom as sdk
                except ImportError:
                    comtypes.client.GetModule(_NORAXON_TYPELIB)
                    import comtypes.gen.Easy2AcquireCom as sdk

                dm = comtypes.client.CreateObject(sdk.DeviceManager._reg_clsid_)
                dm.Initialize("")
                dm.ClearLastErrorText()
                device = dm.GetCurrentDevice()
                if device is None:
                    raise AdapterError(
                        "Noraxon 未找到当前设备：请先在 myoRESEARCH 中选定 Ultium 接收器"
                    )
                device.SetComponentFilterTags(_EMG_FILTER_TAG)
                tags_per_component: list[list[str]] = []
                with safearray_as_ndarray:
                    for index in range(device.GetComponentCount()):
                        component = device.GetComponent(index)
                        tags_per_component.append(list(component.GetTags()))
                outcome["serials"] = _ultium_serials_from_components(
                    tags_per_component
                )
            finally:
                comtypes.CoUninitialize()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_scan, name="noraxon-emg-scan", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise AdapterError("Noraxon EMG 传感器扫描超时")
    if "error" in outcome:
        raise AdapterError(f"Noraxon EMG 传感器扫描失败：{outcome['error']}")
    return list(outcome.get("serials", []))


class NoraxonEmgAdapter(QueuedHardwareAdapter):
    """Record four configured muscles from Noraxon Ultium EMG sensors.

    The configured channels form a fixed column layout.  A missing sensor leaves
    its column as NaN for the whole Trial (never a fatal fault) and is reported
    through descriptor metadata, health metrics, and a DEGRADED health status.
    """

    def __init__(
        self,
        config: NoraxonEmgConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._config = _coerce_config(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._sampler: _NoraxonSampler | None = None
        self._sample_index = 0
        self._sequence = 0
        self._detected_unit_ids: tuple[str, ...] = ()
        self._missing_unit_ids: tuple[str, ...] = ()
        self._missing_muscles: tuple[str, ...] = ()
        self._matched_indices: tuple[int, ...] = ()
        self._actual_rate_hz = float(self._config.sample_rate_hz)

    # -- descriptor / configuration ------------------------------------------

    def descriptor(self) -> ModalityDescriptor:
        names = tuple(channel.name for channel in self._config.channels)
        detected = set(self._detected_unit_ids)
        channel_mapping = [
            {
                "channel": index,
                "muscle": channel.name,
                "unit_id": channel.unit_id,
                "connected": bool(
                    _normalise_unit_id(channel.unit_id)
                    and _normalise_unit_id(channel.unit_id) in detected
                ),
            }
            for index, channel in enumerate(self._config.channels)
        ]
        return ModalityDescriptor(
            device_id=self._config.device_id,
            modality="emg",
            display_name="Noraxon Ultium EMG",
            clock_domain=self._config.clock_domain,
            event_kind="sample_batch",
            channels=names,
            units=tuple(self._config.unit for _ in names),
            nominal_rate_hz=self._actual_rate_hz,
            sample_shape=(len(names),),
            dtype=np.dtype(np.float32).str,
            metadata={
                "manufacturer": "Noraxon",
                "simulated": False,
                "storage_format": "block_binary",
                "channel_names": list(names),
                "unit_ids": [channel.unit_id for channel in self._config.channels],
                "muscle_to_channel": {
                    channel.name: index
                    for index, channel in enumerate(self._config.channels)
                },
                "channel_mapping": channel_mapping,
                "detected_unit_ids": list(self._detected_unit_ids),
                "missing_unit_ids": list(self._missing_unit_ids),
                "missing_muscles": list(self._missing_muscles),
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {
            **asdict(self._config),
            "channels": [
                {"name": channel.name, "unit_id": channel.unit_id}
                for channel in self._config.channels
            ],
            "detected_unit_ids": list(self._detected_unit_ids),
            "missing_unit_ids": list(self._missing_unit_ids),
        }

    # -- lifecycle ------------------------------------------------------------

    def _connect_hardware(self) -> None:
        self._sampler = _NoraxonSampler(self._config, self._on_batch, self._on_sampler_fault)
        self._sampler.start()
        if not self._sampler._connected_event.wait(_CONNECT_TIMEOUT_S):
            raise AdapterError("Noraxon EMG 连接超时（30 秒内未就绪）")
        if self._sampler._error is not None:
            raise AdapterError(f"Noraxon EMG 连接失败：{self._sampler._error}")
        self._detected_unit_ids = tuple(self._sampler.detected_unit_ids)
        self._missing_unit_ids = tuple(unit_id for _, unit_id in self._sampler.missing)
        self._missing_muscles = tuple(name for name, _ in self._sampler.missing)
        self._matched_indices = tuple(self._sampler.matched_indices)
        self._actual_rate_hz = self._sampler.actual_rate_hz
        if self._missing_muscles:
            # A missing sensor is a warning, not a fatal fault: the affected
            # channel is recorded as NaN and the shortfall is surfaced through
            # health() and descriptor metadata.
            _log.warning(
                "Noraxon EMG: %d 个传感器未连接: %s",
                len(self._missing_muscles),
                ", ".join(self._missing_muscles),
            )
            self._emit_status(
                DeviceStatus.CONNECTED,
                "Noraxon EMG 传感器缺失告警：缺少 "
                + ", ".join(self._missing_muscles),
            )

    def _reset_trial_state(self) -> None:
        self._sample_index = 0
        self._sequence = 0

    def _start_hardware(self) -> None:
        # The sampler drains the device continuously; publishing is gated on the
        # RUNNING lifecycle state by ``_publish_raw``.
        return None

    def _stop_hardware(self) -> None:
        return None

    def _close_hardware(self) -> None:
        sampler = self._sampler
        if sampler is None:
            return
        sampler.stop()
        sampler.join(timeout=_JOIN_TIMEOUT_S)
        if sampler.is_alive():
            _log.warning("Noraxon EMG sampler thread did not stop within %.0f s", _JOIN_TIMEOUT_S)
        self._sampler = None

    # -- callbacks ------------------------------------------------------------

    def _on_sampler_fault(self, exc: BaseException) -> None:
        self._set_fault(exc)

    def _on_batch(self, data: np.ndarray, host_ns: int) -> None:
        if self.state is not AdapterState.RUNNING:
            return
        try:
            sample_count = int(data.shape[0])
            event = SampleBatch(
                session_uuid=str(self._trial.session_uuid)
                if self._trial and self._trial.session_uuid is not None
                else None,
                trial_uuid=str(self._trial.trial_uuid) if self._trial else None,
                device_id=self._config.device_id,
                modality="emg",
                clock_domain=self._config.clock_domain,
                host_monotonic_ns=host_ns,
                host_utc_ns=time_ns(),
                first_sample_index=self._sample_index,
                sample_count=sample_count,
                sequence_number=self._sequence,
                device_timestamp=None,
                sample_rate_hz=self._actual_rate_hz,
                data=np.ascontiguousarray(data, dtype=np.float32),
            )
            self._publish_raw(event, item_count=sample_count, host_monotonic_ns=host_ns)
            self._sample_index += sample_count
            self._sequence += 1
        except BaseException as exc:
            self._set_fault(AdapterError(f"Noraxon EMG 数据发布失败：{exc}"))

    # -- health ---------------------------------------------------------------

    def health(self) -> Any:
        snapshot = super().health()
        if self._missing_unit_ids and snapshot.status is HealthStatus.HEALTHY:
            snapshot = snapshot.model_copy(update={"status": HealthStatus.DEGRADED})
        return snapshot

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "channel_count": len(self._config.channels),
            "matched_channel_count": len(self._matched_indices),
            "detected_sensor_count": len(self._detected_unit_ids),
            "missing_sensor_count": len(self._missing_unit_ids),
        }


__all__ = [
    "NoraxonEmgAdapter",
    "NoraxonEmgChannel",
    "NoraxonEmgConfig",
    "scan_ultium_units",
]
