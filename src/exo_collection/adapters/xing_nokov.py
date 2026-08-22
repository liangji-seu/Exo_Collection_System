"""XING/Nokov SDK backends and hardware adapters.

The vendor client receives UDP data internally.  Its callback memory is copied
immediately into NumPy/Python-owned objects before returning to the SDK.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter_ns, time_ns
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import SampleBatch

# The vendor ``nokovsdk`` wrapper keeps each registered callback in a single
# module-global ``data_func`` / ``analog_func``.  Registering a second client of
# the same kind overwrites that global and would otherwise let the first thunk
# be garbage collected, leaving a dangling function pointer inside the C++
# client.  Retain every thunk for the lifetime of the process.
_SDK_CALLBACK_LOCK = Lock()
_SDK_CALLBACK_KEEPALIVE: list[Any] = []


def _coerce_dataclass(config_type: type[Any], value: Any) -> Any:
    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw.pop("id")
    allowed = config_type.__dataclass_fields__
    return config_type(**{key: item for key, item in raw.items() if key in allowed})


class XingStreamBackend(Protocol):
    metadata: Mapping[str, Any]

    def connect(self, callback: Callable[[Mapping[str, Any], int], None]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def _decode_name(value: Any, fallback: str) -> str:
    try:
        raw = bytes(value).split(b"\0", 1)[0]
        return raw.decode("utf-8", errors="replace").strip() or fallback
    except Exception:
        return fallback


class NokovSdkBackend:
    """Small ownership wrapper around the vendor ``nokov.nokovsdk`` module."""

    def __init__(
        self,
        *,
        server_ip: str,
        stream_kind: str,
        marker_count_fallback: int = 0,
        frame_rate_fallback_hz: float = 100.0,
        unlabeled_marker_capacity: int = 16,
    ) -> None:
        if stream_kind not in {"mocap", "emg", "force_plate"}:
            raise ValueError("stream_kind must be mocap, emg, or force_plate")
        self.server_ip = server_ip
        self.stream_kind = stream_kind
        self.marker_count_fallback = int(marker_count_fallback)
        self.frame_rate_fallback_hz = float(frame_rate_fallback_hz)
        self.unlabeled_marker_capacity = int(unlabeled_marker_capacity)
        self.metadata: dict[str, Any] = {}
        self._sdk: Any = None
        self._client: Any = None
        self._callback: Callable[[Mapping[str, Any], int], None] | None = None
        self._accepting = False
        self._sdk_callback: Any = None

    def connect(self, callback: Callable[[Mapping[str, Any], int], None]) -> None:
        try:
            from nokov import nokovsdk
        except (ImportError, OSError) as exc:
            raise AdapterError(
                "未能加载 XING/Nokov Python SDK。请先安装现场资料 dist 目录中的 "
                "nokovpy-3.0.1-py3-none-any.whl，并确认 Microsoft VC++ x64 "
                "运行库可用。"
            ) from exc
        self._sdk = nokovsdk
        self._callback = callback
        client = nokovsdk.PySDKClient()
        self._client = client
        # Mirror the vendor example ordering (Nokov_SDK_Client.py): Initialize
        # first, then register the data/message/notify callbacks, then read the
        # server description before the data descriptions.
        result = int(client.Initialize(self.server_ip.encode("utf-8")))
        if result != 0:
            self.close()
            raise AdapterError(
                f"XING/Nokov SDK 连接 {self.server_ip} 失败，错误码 {result}"
            )
        if self.stream_kind == "emg":
            # Analog channels with per-mocap-frame subframes (high-rate devices).
            self._sdk_callback = self._on_analog_frame
            with _SDK_CALLBACK_LOCK:
                client.PySetAnalogChFunc(self._sdk_callback, None)
                _SDK_CALLBACK_KEEPALIVE.append(nokovsdk.analog_func)
        else:
            # Marker and 6-DOF force-plate data both ride the same broadcast
            # frame (``FrameOfMocapData``); only the extracted section differs.
            self._sdk_callback = (
                self._on_mocap_frame
                if self.stream_kind == "mocap"
                else self._on_force_frame
            )
            with _SDK_CALLBACK_LOCK:
                client.PySetDataCallback(self._sdk_callback, None)
                _SDK_CALLBACK_KEEPALIVE.append(nokovsdk.data_func)
        client.PySetVerbosityLevel(0)
        client.PySetMessageCallback(self._on_sdk_message)
        client.PySetNotifyMsgCallback(self._on_sdk_notify, None)
        version = tuple(int(value) for value in client.PyNokovVersion())
        self.metadata = {
            "manufacturer": "Nokov/XING",
            "sdk_version": ".".join(str(value) for value in version),
            "server_ip": self.server_ip,
            "transport": "vendor_sdk_udp",
        }
        if self.stream_kind == "mocap":
            server_description = nokovsdk.ServerDescription()
            client.PyGetServerDescription(server_description)
            self.metadata.update(self._read_mocap_description())

    def _read_mocap_description(self) -> dict[str, Any]:
        assert self._sdk is not None and self._client is not None
        sdk = self._sdk
        marker_sets: list[dict[str, Any]] = []
        frame_rate = self.frame_rate_fallback_hz
        try:
            descriptions = sdk.POINTER(sdk.DataDescriptions)()
            result = int(self._client.PyGetDataDescriptions(descriptions))
            if result == 0 and bool(descriptions):
                data = descriptions.contents
                for index in range(int(data.nDataDescriptions)):
                    definition = data.arrDataDescriptions[index]
                    if definition.type == sdk.DataDescriptors.Descriptor_MarkerSet.value:
                        item = definition.Data.MarkerSetDescription.contents
                        set_name = _decode_name(item.szName, f"markerset_{index + 1}")
                        names = [
                            _decode_name(
                                item.szMarkerNames[marker_index],
                                f"{set_name}_{marker_index + 1:02d}",
                            )
                            for marker_index in range(int(item.nMarkers))
                        ]
                        marker_sets.append({"name": set_name, "marker_names": names})
                    elif definition.type == sdk.DataDescriptors.Descriptor_Param.value:
                        frame_rate = float(definition.Data.DataParam.contents.nFrameRate)
        except Exception:
            marker_sets = []
        marker_names = [
            f"{marker_set['name']}/{name}"
            for marker_set in marker_sets
            for name in marker_set["marker_names"]
        ]
        if marker_names:
            marker_source = "labeled"
        elif self.marker_count_fallback > 0:
            marker_names = [
                f"marker_{index + 1:02d}" for index in range(self.marker_count_fallback)
            ]
            marker_sets = [{"name": "configured", "marker_names": marker_names.copy()}]
            marker_source = "fallback"
        else:
            # No labelled MarkerSet is defined: collect the Seeker's unlabelled
            # (OtherMarkers) points so the acquisition pipeline still has data
            # to flow.  The per-frame count varies, so the stream is padded to a
            # fixed capacity with NaN where a marker is absent.
            marker_names = [
                f"marker_{index + 1:02d}"
                for index in range(self.unlabeled_marker_capacity)
            ]
            marker_sets = []
            marker_source = "unlabeled"
        if not marker_names:
            raise AdapterError(
                "XING/Nokov 未返回 MarkerSet 定义，且未配置后备 Marker 数量；"
                "无法确定 marker 流。"
            )
        return {
            "frame_rate_hz": frame_rate,
            "marker_count": len(marker_names),
            "marker_names": marker_names,
            "marker_sets": marker_sets,
            "marker_source": marker_source,
        }

    def start(self) -> None:
        self._accepting = True

    def stop(self) -> None:
        self._accepting = False

    def close(self) -> None:
        self._accepting = False
        self._callback = None
        self._sdk_callback = None
        # The Python wrapper exposes cleanup only through its destructor.
        self._client = None
        self._sdk = None
        gc.collect()

    def _on_sdk_message(self, _level: Any, _message: Any) -> None:
        # Vendor log telemetry; deliberately ignored.  Never unwind through the
        # ctypes callback boundary, and never log opaque SDK content here.
        return None

    def _on_sdk_notify(self, _notify: Any, _user_data: Any) -> None:
        # Vendor state-change notifications; deliberately ignored.
        return None

    def _on_mocap_frame(self, pointer: Any, _user_data: Any) -> None:
        if not self._accepting or not pointer or self._callback is None:
            return
        try:
            frame = pointer.contents
            marker_sets: list[dict[str, Any]] = []
            for set_index in range(int(frame.nMarkerSets)):
                source = frame.MocapData[set_index]
                values = np.empty((int(source.nMarkers), 3), dtype=np.float32)
                for marker_index in range(int(source.nMarkers)):
                    values[marker_index] = tuple(
                        float(source.Markers[marker_index][axis]) for axis in range(3)
                    )
                marker_sets.append(
                    {
                        "name": _decode_name(source.szName, f"markerset_{set_index + 1}"),
                        "values": values,
                    }
                )
            other_markers = np.empty((int(frame.nOtherMarkers), 3), dtype=np.float32)
            for marker_index in range(int(frame.nOtherMarkers)):
                other_markers[marker_index] = tuple(
                    float(frame.OtherMarkers[marker_index][axis]) for axis in range(3)
                )
            self._callback(
                {
                    "frame_number": int(frame.iFrame),
                    "device_timestamp": int(frame.iTimeStamp),
                    "marker_sets": marker_sets,
                    "other_markers": other_markers,
                },
                perf_counter_ns(),
            )
        except BaseException:
            # Never unwind through a ctypes callback boundary.
            return

    def _on_force_frame(self, pointer: Any, _user_data: Any) -> None:
        if not self._accepting or not pointer or self._callback is None:
            return
        try:
            frame = pointer.contents
            # The force-plate's six axes (Fx/Fy/Fz/Mx/My/Mz) are exposed as the
            # frame's analog channels: a flat ``Analogdata`` array, one value
            # per channel per mocap frame (no subframes).
            channel_count = int(frame.nAnalogdatas)
            values = np.empty((1, channel_count), dtype=np.float32)
            for channel in range(channel_count):
                values[0, channel] = float(frame.Analogdata[channel])
            self._callback(
                {
                    "frame_number": int(frame.iFrame),
                    "device_timestamp": int(frame.iTimeStamp),
                    "values": values,
                },
                perf_counter_ns(),
            )
        except BaseException:
            # Never unwind through a ctypes callback boundary.
            return

    def _on_analog_frame(self, pointer: Any, _user_data: Any) -> None:
        if not self._accepting or not pointer or self._callback is None:
            return
        try:
            frame = pointer.contents
            channel_count = int(frame.nAnalogdatas)
            subframes = int(frame.nSubFrame)
            values = np.empty((subframes, channel_count), dtype=np.float32)
            for channel in range(channel_count):
                for subframe in range(subframes):
                    values[subframe, channel] = float(frame.Analogdata[channel][subframe])
            self._callback(
                {
                    "frame_number": int(frame.iFrame),
                    "device_timestamp": int(frame.iTimeStamp),
                    "values": values,
                },
                perf_counter_ns(),
            )
        except BaseException:
            return


@dataclass(frozen=True, slots=True)
class XingNokovMocapConfig:
    device_id: str = "mocap_xing_nokov"
    clock_domain: str = "mocap_xing_nokov_clock"
    server_ip: str = "10.1.1.198"
    nominal_rate_hz: float = 100.0
    marker_count_fallback: int = 0
    unlabeled_marker_capacity: int = 16
    queue_capacity: int = 256

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.clock_domain.strip() or not self.server_ip.strip():
            raise ValueError("device_id, clock_domain, and server_ip must not be empty")
        if self.nominal_rate_hz <= 0 or self.marker_count_fallback < 0:
            raise ValueError("invalid mocap rate or marker_count_fallback")
        if self.unlabeled_marker_capacity < 0:
            raise ValueError("unlabeled_marker_capacity must be non-negative")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")


class XingNokovMocapAdapter(QueuedHardwareAdapter):
    def __init__(
        self,
        config: XingNokovMocapConfig | Mapping[str, Any] | None = None,
        *,
        backend: XingStreamBackend | None = None,
    ) -> None:
        self._config = _coerce_dataclass(XingNokovMocapConfig, config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or NokovSdkBackend(
            server_ip=self._config.server_ip,
            stream_kind="mocap",
            marker_count_fallback=self._config.marker_count_fallback,
            frame_rate_fallback_hz=self._config.nominal_rate_hz,
            unlabeled_marker_capacity=self._config.unlabeled_marker_capacity,
        )
        self._marker_names: tuple[str, ...] = ()
        self._marker_sets: tuple[Mapping[str, Any], ...] = ()
        self._unlabeled = False
        self._last_unlabeled_marker_count = 0
        self._sample_index = 0
        self._sequence = 0
        self._last_frame_number: int | None = None
        self._sequence_gaps_count = 0
        self._malformed_frames = 0

    def descriptor(self) -> ModalityDescriptor:
        marker_count = len(self._marker_names) or self._config.marker_count_fallback or 1
        rate = float(self._backend.metadata.get("frame_rate_hz", self._config.nominal_rate_hz))
        return ModalityDescriptor(
            device_id=self._config.device_id,
            modality="mocap",
            display_name="XING/Nokov motion-capture markers",
            clock_domain=self._config.clock_domain,
            event_kind="sample_batch",
            channels=("x", "y", "z"),
            units=("mm", "mm", "mm"),
            nominal_rate_hz=rate,
            sample_shape=(marker_count, 3),
            dtype=np.dtype(np.float32).str,
            metadata={
                **dict(self._backend.metadata),
                "simulated": False,
                "coordinate_unit": "mm",
                "marker_names": list(self._marker_names),
                "marker_sets": list(self._marker_sets),
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {**asdict(self._config), **dict(self._backend.metadata)}

    def _connect_hardware(self) -> None:
        self._backend.connect(self._on_frame)
        metadata = dict(self._backend.metadata)
        self._marker_names = tuple(str(item) for item in metadata.get("marker_names", ()))
        self._marker_sets = tuple(metadata.get("marker_sets", ()))
        self._unlabeled = metadata.get("marker_source") == "unlabeled" or not self._marker_sets
        if not self._marker_names:
            raise AdapterError("XING/Nokov Marker 定义为空")

    def _reset_trial_state(self) -> None:
        self._sample_index = 0
        self._sequence = 0
        self._last_frame_number = None
        self._sequence_gaps_count = 0
        self._malformed_frames = 0

    def _start_hardware(self) -> None:
        self._backend.start()

    def _stop_hardware(self) -> None:
        self._backend.stop()

    def _close_hardware(self) -> None:
        self._backend.close()

    def _on_frame(self, payload: Mapping[str, Any], host_ns: int) -> None:
        try:
            if self._unlabeled:
                capacity = len(self._marker_names)
                raw = payload.get("other_markers")
                other = (
                    np.asarray(raw, dtype=np.float32)
                    if raw is not None
                    else np.empty((0, 3), dtype=np.float32)
                )
                if other.ndim != 2 or other.shape[1] != 3:
                    other = np.empty((0, 3), dtype=np.float32)
                count = min(int(other.shape[0]), capacity)
                data = np.full((1, capacity, 3), np.nan, dtype=np.float32)
                if count:
                    data[0, :count, :] = other[:count, :]
                data = np.ascontiguousarray(data)
                self._last_unlabeled_marker_count = count
            else:
                by_name = {
                    str(item["name"]): np.asarray(item["values"], dtype=np.float32)
                    for item in payload["marker_sets"]
                }
                rows: list[np.ndarray] = []
                for marker_set in self._marker_sets:
                    values = by_name.get(str(marker_set["name"]))
                    expected = len(marker_set["marker_names"])
                    if values is None or values.shape != (expected, 3):
                        raise ValueError(
                            f"MarkerSet {marker_set['name']} shape mismatch: "
                            f"expected {(expected, 3)}, got {None if values is None else values.shape}"
                        )
                    rows.append(values)
                data = np.ascontiguousarray(np.concatenate(rows, axis=0)[None, ...])
            frame_number = int(payload["frame_number"])
            if self._last_frame_number is not None and frame_number > self._last_frame_number + 1:
                self._sequence_gaps_count += frame_number - self._last_frame_number - 1
            self._last_frame_number = frame_number
            event = SampleBatch(
                session_uuid=str(self._trial.session_uuid) if self._trial and self._trial.session_uuid else None,
                trial_uuid=str(self._trial.trial_uuid) if self._trial else None,
                device_id=self._config.device_id,
                modality="mocap",
                clock_domain=self._config.clock_domain,
                host_monotonic_ns=host_ns,
                host_utc_ns=time_ns(),
                first_sample_index=self._sample_index,
                sample_count=1,
                sequence_number=self._sequence,
                device_timestamp=payload.get("device_timestamp"),
                sample_rate_hz=self.descriptor().nominal_rate_hz,
                data=data,
            )
            self._publish_raw(event, item_count=1, host_monotonic_ns=host_ns)
            self._sample_index += 1
            self._sequence += 1
        except BaseException as exc:
            self._malformed_frames += 1
            self._set_fault(AdapterError(f"XING/Nokov Marker 帧解析失败：{exc}"))

    def _sequence_gaps(self) -> int:
        return self._sequence_gaps_count

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        metrics: dict[str, int | float | str | bool | None] = {
            "marker_count": len(self._marker_names),
            "malformed_frames": self._malformed_frames,
            "last_frame_number": self._last_frame_number,
        }
        if self._unlabeled:
            metrics["marker_source"] = "unlabeled"
            metrics["last_unlabeled_marker_count"] = self._last_unlabeled_marker_count
        return metrics


@dataclass(frozen=True, slots=True)
class XingNokovEmgConfig:
    device_id: str = "emg_xing_nokov"
    clock_domain: str = "emg_xing_nokov_clock"
    server_ip: str = "10.1.1.198"
    sample_rate_hz: float = 1000.0
    channel_count: int = 8
    channel_names: tuple[str, ...] = ()
    unit: str = "mV"
    queue_capacity: int = 512

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.clock_domain.strip() or not self.server_ip.strip():
            raise ValueError("device_id, clock_domain, and server_ip must not be empty")
        if self.sample_rate_hz <= 0 or self.channel_count <= 0 or self.queue_capacity <= 0:
            raise ValueError("EMG rate, channel_count, and queue_capacity must be positive")
        names = tuple(str(item).strip() for item in self.channel_names if str(item).strip())
        if names and len(names) != self.channel_count:
            raise ValueError("channel_names length must equal channel_count")
        if not self.unit.strip():
            raise ValueError("unit must not be empty")
        object.__setattr__(self, "channel_names", names)


class XingNokovEmgAdapter(QueuedHardwareAdapter):
    def __init__(
        self,
        config: XingNokovEmgConfig | Mapping[str, Any] | None = None,
        *,
        backend: XingStreamBackend | None = None,
    ) -> None:
        self._config = _coerce_dataclass(XingNokovEmgConfig, config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or NokovSdkBackend(
            server_ip=self._config.server_ip,
            stream_kind="emg",
            frame_rate_fallback_hz=self._config.sample_rate_hz,
        )
        self._sample_index = 0
        self._sequence = 0
        self._last_frame_number: int | None = None
        self._sequence_gaps_count = 0
        self._malformed_frames = 0

    @property
    def _channel_names(self) -> tuple[str, ...]:
        return self._config.channel_names or tuple(
            f"emg_{index + 1:02d}" for index in range(self._config.channel_count)
        )

    def descriptor(self) -> ModalityDescriptor:
        names = self._channel_names
        return ModalityDescriptor(
            device_id=self._config.device_id,
            modality="emg",
            display_name="XING/Nokov analog EMG",
            clock_domain=self._config.clock_domain,
            event_kind="sample_batch",
            channels=names,
            units=tuple(self._config.unit for _ in names),
            nominal_rate_hz=self._config.sample_rate_hz,
            sample_shape=(self._config.channel_count,),
            dtype=np.dtype(np.float32).str,
            metadata={
                **dict(self._backend.metadata),
                "simulated": False,
                "channel_names": list(names),
                "analog_layout": "sdk_channel_by_subframe_transposed_to_sample_by_channel",
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {**asdict(self._config), **dict(self._backend.metadata)}

    def _connect_hardware(self) -> None:
        self._backend.connect(self._on_frame)

    def _reset_trial_state(self) -> None:
        self._sample_index = 0
        self._sequence = 0
        self._last_frame_number = None
        self._sequence_gaps_count = 0
        self._malformed_frames = 0

    def _start_hardware(self) -> None:
        self._backend.start()

    def _stop_hardware(self) -> None:
        self._backend.stop()

    def _close_hardware(self) -> None:
        self._backend.close()

    def _on_frame(self, payload: Mapping[str, Any], host_ns: int) -> None:
        try:
            data = np.ascontiguousarray(payload["values"], dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != self._config.channel_count or data.shape[0] < 1:
                raise ValueError(
                    f"expected (*, {self._config.channel_count}), got {data.shape}"
                )
            frame_number = int(payload["frame_number"])
            if self._last_frame_number is not None and frame_number > self._last_frame_number + 1:
                self._sequence_gaps_count += frame_number - self._last_frame_number - 1
            self._last_frame_number = frame_number
            sample_count = int(data.shape[0])
            event = SampleBatch(
                session_uuid=str(self._trial.session_uuid) if self._trial and self._trial.session_uuid else None,
                trial_uuid=str(self._trial.trial_uuid) if self._trial else None,
                device_id=self._config.device_id,
                modality="emg",
                clock_domain=self._config.clock_domain,
                host_monotonic_ns=host_ns,
                host_utc_ns=time_ns(),
                first_sample_index=self._sample_index,
                sample_count=sample_count,
                sequence_number=self._sequence,
                device_timestamp=payload.get("device_timestamp"),
                sample_rate_hz=self._config.sample_rate_hz,
                data=data,
            )
            self._publish_raw(event, item_count=sample_count, host_monotonic_ns=host_ns)
            self._sample_index += sample_count
            self._sequence += 1
        except BaseException as exc:
            self._malformed_frames += 1
            self._set_fault(AdapterError(f"XING/Nokov EMG 帧解析失败：{exc}"))

    def _sequence_gaps(self) -> int:
        return self._sequence_gaps_count

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "channel_count": self._config.channel_count,
            "malformed_frames": self._malformed_frames,
            "last_frame_number": self._last_frame_number,
        }


@dataclass(frozen=True, slots=True)
class XingNokovForcePlateConfig:
    device_id: str = "force_plate_xing_nokov"
    clock_domain: str = "force_plate_xing_nokov_clock"
    server_ip: str = "10.1.1.198"
    sample_rate_hz: float = 100.0
    channel_count: int = 6
    channel_names: tuple[str, ...] = ("fx", "fy", "fz", "mx", "my", "mz")
    units: tuple[str, ...] = ("N", "N", "N", "N*m", "N*m", "N*m")
    queue_capacity: int = 512

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.clock_domain.strip() or not self.server_ip.strip():
            raise ValueError("device_id, clock_domain, and server_ip must not be empty")
        if self.sample_rate_hz <= 0 or self.channel_count <= 0 or self.queue_capacity <= 0:
            raise ValueError(
                "force plate rate, channel_count, and queue_capacity must be positive"
            )
        names = tuple(str(item).strip() for item in self.channel_names if str(item).strip())
        units = tuple(str(item).strip() for item in self.units if str(item).strip())
        if names and len(names) != self.channel_count:
            raise ValueError("channel_names length must equal channel_count")
        if units and len(units) != self.channel_count:
            raise ValueError("units length must equal channel_count")
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "units", units)


class XingNokovForcePlateAdapter(QueuedHardwareAdapter):
    def __init__(
        self,
        config: XingNokovForcePlateConfig | Mapping[str, Any] | None = None,
        *,
        backend: XingStreamBackend | None = None,
    ) -> None:
        self._config = _coerce_dataclass(XingNokovForcePlateConfig, config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or NokovSdkBackend(
            server_ip=self._config.server_ip,
            stream_kind="force_plate",
            frame_rate_fallback_hz=self._config.sample_rate_hz,
        )
        self._sample_index = 0
        self._sequence = 0
        self._last_frame_number: int | None = None
        self._sequence_gaps_count = 0
        self._malformed_frames = 0

    @property
    def _channel_names(self) -> tuple[str, ...]:
        return self._config.channel_names or tuple(
            f"force_{index + 1:02d}" for index in range(self._config.channel_count)
        )

    @property
    def _units(self) -> tuple[str, ...]:
        if self._config.units:
            return self._config.units
        return tuple("" for _ in self._channel_names)

    def descriptor(self) -> ModalityDescriptor:
        names = self._channel_names
        return ModalityDescriptor(
            device_id=self._config.device_id,
            modality="force_plate",
            display_name="XING/Nokov 六维力 (analog)",
            clock_domain=self._config.clock_domain,
            event_kind="sample_batch",
            channels=names,
            units=self._units,
            nominal_rate_hz=self._config.sample_rate_hz,
            sample_shape=(self._config.channel_count,),
            dtype=np.dtype(np.float32).str,
            metadata={
                **dict(self._backend.metadata),
                "simulated": False,
                "channel_names": list(names),
                "units": list(self._units),
                "analog_layout": "mocap_frame_analog_channel",
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {**asdict(self._config), **dict(self._backend.metadata)}

    def _connect_hardware(self) -> None:
        self._backend.connect(self._on_frame)

    def _reset_trial_state(self) -> None:
        self._sample_index = 0
        self._sequence = 0
        self._last_frame_number = None
        self._sequence_gaps_count = 0
        self._malformed_frames = 0

    def _start_hardware(self) -> None:
        self._backend.start()

    def _stop_hardware(self) -> None:
        self._backend.stop()

    def _close_hardware(self) -> None:
        self._backend.close()

    def _on_frame(self, payload: Mapping[str, Any], host_ns: int) -> None:
        try:
            data = np.ascontiguousarray(payload["values"], dtype=np.float32)
            if (
                data.ndim != 2
                or data.shape[1] != self._config.channel_count
                or data.shape[0] < 1
            ):
                raise ValueError(
                    f"expected (*, {self._config.channel_count}), got {data.shape}"
                )
            frame_number = int(payload["frame_number"])
            if self._last_frame_number is not None and frame_number > self._last_frame_number + 1:
                self._sequence_gaps_count += frame_number - self._last_frame_number - 1
            self._last_frame_number = frame_number
            sample_count = int(data.shape[0])
            event = SampleBatch(
                session_uuid=str(self._trial.session_uuid) if self._trial and self._trial.session_uuid else None,
                trial_uuid=str(self._trial.trial_uuid) if self._trial else None,
                device_id=self._config.device_id,
                modality="force_plate",
                clock_domain=self._config.clock_domain,
                host_monotonic_ns=host_ns,
                host_utc_ns=time_ns(),
                first_sample_index=self._sample_index,
                sample_count=sample_count,
                sequence_number=self._sequence,
                device_timestamp=payload.get("device_timestamp"),
                sample_rate_hz=self._config.sample_rate_hz,
                data=data,
            )
            self._publish_raw(event, item_count=sample_count, host_monotonic_ns=host_ns)
            self._sample_index += sample_count
            self._sequence += 1
        except BaseException as exc:
            self._malformed_frames += 1
            self._set_fault(AdapterError(f"XING/Nokov 测力台帧解析失败：{exc}"))

    def _sequence_gaps(self) -> int:
        return self._sequence_gaps_count

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "channel_count": self._config.channel_count,
            "malformed_frames": self._malformed_frames,
            "last_frame_number": self._last_frame_number,
        }


__all__ = [
    "NokovSdkBackend",
    "XingNokovEmgAdapter",
    "XingNokovEmgConfig",
    "XingNokovForcePlateAdapter",
    "XingNokovForcePlateConfig",
    "XingNokovMocapAdapter",
    "XingNokovMocapConfig",
    "XingStreamBackend",
]
