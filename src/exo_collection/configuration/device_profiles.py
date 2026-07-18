"""Typed loader for the built-in simulated device profile.

The ``adapter`` field is a validated identifier only.  Loading this document
never imports a module or resolves an arbitrary class name; the orchestration
layer owns a fixed in-process adapter registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SimulatedModality: TypeAlias = Literal[
    "ultrasound", "imu", "encoder", "sync_pulse"
]

ULTRASOUND_ADAPTER = "exo_collection.adapters.simulated.SimulatedUltrasoundAdapter"
IMU_ADAPTER = "exo_collection.adapters.simulated.SimulatedImuAdapter"
ENCODER_ADAPTER = "exo_collection.adapters.simulated.SimulatedEncoderAdapter"
SYNC_PULSE_ADAPTER = "exo_collection.adapters.simulated.SimulatedSyncPulseAdapter"
ELONXI_ULTRASOUND_ADAPTER = (
    "exo_collection.adapters.ultrasound.ElonxiUltrasoundAdapter"
)
XSENS_AWINDA_ADAPTER = "exo_collection.adapters.imu.XsensAwindaImuAdapter"
TEENSY_ENCODER_ADAPTER = (
    "exo_collection.adapters.encoder.TeensySerialEncoderAdapter"
)
RAW_ETHERNET_ULTRASOUND_ADAPTER = (
    "exo_collection.adapters.ultrasound.RawEthernetUltrasoundAdapter"
)


class ProfileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_default=True,
    )


class CommonSimulationParameters(ProfileModel):
    queue_capacity: int | None = Field(default=None, gt=0)
    seed: int | None = None
    clock_drift_ppm: float | None = None
    timestamp_jitter_ns: int | None = Field(default=None, ge=0)
    drop_every_n_batches: int | None = Field(default=None, ge=0)
    drop_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    disconnect_after_batches: int | None = Field(default=None, ge=0)
    realtime: bool | None = None


class UltrasoundSimulationParameters(CommonSimulationParameters):
    frame_rate_hz: float | None = Field(default=None, gt=0)
    frames_per_batch: int | None = Field(default=None, gt=0)
    channel_count: int | None = Field(default=None, gt=0)
    samples_per_channel: int | None = Field(default=None, gt=0)
    frame_shape: tuple[int, ...] | None = None
    dtype: NonEmptyStr | None = None
    baseline: float | None = None
    echo_amplitude: float | None = None
    noise_std: float | None = Field(default=None, ge=0)

    @field_validator("frame_shape")
    @classmethod
    def validate_frame_shape(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and (not value or any(dimension <= 0 for dimension in value)):
            raise ValueError("frame_shape must contain positive dimensions")
        return value


class ImuSimulationParameters(CommonSimulationParameters):
    device_ids: tuple[NonEmptyStr, ...] | None = None
    sample_rate_hz: float | None = Field(default=None, gt=0)
    samples_per_batch: int | None = Field(default=None, gt=0)
    noise_fraction: float | None = Field(default=None, ge=0)

    @field_validator("device_ids")
    @classmethod
    def validate_device_ids(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is not None and (not value or len(value) != len(set(value))):
            raise ValueError("device_ids must be non-empty and unique")
        return value


class EncoderSimulationParameters(CommonSimulationParameters):
    sample_rate_hz: float | None = Field(default=None, gt=0)
    samples_per_batch: int | None = Field(default=None, gt=0)
    hardware_tick_hz: float | None = Field(default=None, gt=0)
    hardware_tick_modulus: int | None = Field(default=None, gt=1)
    sequence_modulus: int | None = Field(default=None, gt=1)
    gait_frequency_hz: float | None = Field(default=None, gt=0)
    noise_std: float | None = Field(default=None, ge=0)


class SyncPulseSimulationParameters(CommonSimulationParameters):
    sample_rate_hz: float | None = Field(default=None, gt=0)
    samples_per_batch: int | None = Field(default=None, gt=0)
    baseline_voltage: float | None = None
    pulse_voltage: float | None = None
    noise_std_voltage: float | None = Field(default=None, ge=0)
    pulse_interval_s: float | None = Field(default=None, gt=0)
    pulse_width_s: float | None = Field(default=None, gt=0)
    first_pulse_s: float | None = Field(default=None, ge=0)
    high_threshold: float | None = None
    low_threshold: float | None = None
    min_pulse_width_ns: int | None = Field(default=None, ge=0)
    debounce_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_pulse_width(self) -> SyncPulseSimulationParameters:
        if (
            self.pulse_interval_s is not None
            and self.pulse_width_s is not None
            and self.pulse_width_s >= self.pulse_interval_s
        ):
            raise ValueError("pulse_width_s must be less than pulse_interval_s")
        return self


class DeviceProfileBase(ProfileModel):
    device_id: NonEmptyStr = Field(alias="id")
    required: bool
    clock_domain: NonEmptyStr
    parameters: CommonSimulationParameters

    def adapter_configuration(self) -> dict[str, Any]:
        parameters = self.parameters.model_dump(exclude_none=True)
        return {
            "device_id": self.device_id,
            "clock_domain": self.clock_domain,
            **parameters,
        }


class UltrasoundDeviceProfile(DeviceProfileBase):
    modality: Literal["ultrasound"]
    adapter: Literal[ULTRASOUND_ADAPTER]
    writer: Literal["block_binary"]
    parameters: UltrasoundSimulationParameters


class ImuDeviceProfile(DeviceProfileBase):
    modality: Literal["imu"]
    adapter: Literal[IMU_ADAPTER]
    writer: Literal["hdf5_signal"]
    parameters: ImuSimulationParameters


class EncoderDeviceProfile(DeviceProfileBase):
    modality: Literal["encoder"]
    adapter: Literal[ENCODER_ADAPTER]
    writer: Literal["hdf5_signal"]
    parameters: EncoderSimulationParameters


class SyncPulseDeviceProfile(DeviceProfileBase):
    modality: Literal["sync_pulse"]
    adapter: Literal[SYNC_PULSE_ADAPTER]
    writer: Literal["hdf5_signal"]
    parameters: SyncPulseSimulationParameters


SimulatedDeviceProfile: TypeAlias = Annotated[
    UltrasoundDeviceProfile
    | ImuDeviceProfile
    | EncoderDeviceProfile
    | SyncPulseDeviceProfile,
    Field(discriminator="modality"),
]


class SimulatedDeviceProfileDocument(ProfileModel):
    profile_kind: Literal["simulated"] = "simulated"
    display_name: NonEmptyStr = "内置模拟设备"
    schema_version: Literal["1.0.0"]
    devices: tuple[SimulatedDeviceProfile, ...]

    @model_validator(mode="after")
    def validate_complete_profile(self) -> SimulatedDeviceProfileDocument:
        modalities = [device.modality for device in self.devices]
        expected = {"ultrasound", "imu", "encoder", "sync_pulse"}
        if set(modalities) != expected or len(modalities) != len(expected):
            raise ValueError(
                "simulated profile must define ultrasound, imu, encoder, and sync_pulse exactly once"
            )
        device_ids = [device.device_id for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("simulated device ids must be unique")
        clock_domains = [device.clock_domain for device in self.devices]
        if len(clock_domains) != len(set(clock_domains)):
            raise ValueError("simulated clock domains must be unique")
        return self

    def by_modality(self) -> dict[str, SimulatedDeviceProfile]:
        return {device.modality: device for device in self.devices}


class RawEthernetUltrasoundParameters(ProfileModel):
    interface_name: NonEmptyStr | None = None
    channels: tuple[int, int, int, int] = (1, 2, 3, 4)
    samples_per_channel: int = Field(default=1000, gt=0)
    nominal_rate_hz: float = Field(default=20.0, gt=0)
    queue_capacity: int = Field(default=64, gt=0)
    inbound_queue_capacity: int = Field(default=256, gt=0)
    scan_timeout_s: float = Field(default=1.5, gt=0)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if tuple(value) != (1, 2, 3, 4):
            raise ValueError("ultrasound channels must be exactly [1, 2, 3, 4]")
        return value


class HardwareUltrasoundParameters(ProfileModel):
    sdk_path: NonEmptyStr | None = None
    device_ip: NonEmptyStr | None = None
    port: int = Field(default=1430, ge=1, le=65535)
    channels: tuple[int, int, int, int] = (1, 2, 3, 4)
    samples_per_channel: int = Field(default=1000, gt=0)
    nominal_rate_hz: float = Field(default=20.0, gt=0)
    queue_capacity: int = Field(default=64, gt=0)
    discovery_timeout_s: float = Field(default=10.0, ge=0)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if tuple(value) != (1, 2, 3, 4):
            raise ValueError("ultrasound channels must be exactly [1, 2, 3, 4]")
        return value


class HardwareImuParameters(ProfileModel):
    radio_channel: int = Field(default=25, ge=11, le=25)
    sample_rate_hz: float = Field(default=120.0, gt=0)
    expected_device_count: Literal[3] = 3
    sensor_ids: tuple[NonEmptyStr, ...] = ()
    wait_timeout_s: float = Field(default=15.0, gt=0)
    stable_wait_s: float = Field(default=3.0, ge=0)
    poll_interval_s: float = Field(default=0.25, gt=0)
    pending_group_limit: int = Field(default=128, gt=0)
    queue_capacity: int = Field(default=256, gt=0)

    @field_validator("sensor_ids")
    @classmethod
    def validate_sensor_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value and (len(value) != 3 or len(set(value)) != 3):
            raise ValueError("sensor_ids must be empty or contain three unique IDs")
        return value


class HardwareEncoderParameters(ProfileModel):
    port: NonEmptyStr | None = None
    baudrate: int = Field(default=1_000_000, gt=0)
    vid: int = Field(default=0x16C0, ge=0, le=0xFFFF)
    pid: int = Field(default=0x0483, ge=0, le=0xFFFF)
    nominal_rate_hz: float = Field(default=200.0, gt=0)
    batch_size: int = Field(default=20, gt=0)
    queue_capacity: int = Field(default=256, gt=0)
    read_size: int = Field(default=128, gt=0)
    read_timeout_s: float = Field(default=0.05, gt=0)


class HardwareDeviceProfileBase(ProfileModel):
    device_id: NonEmptyStr = Field(alias="id")
    required: bool
    clock_domain: NonEmptyStr
    simulated: bool

    def adapter_configuration(self) -> dict[str, Any]:
        parameters = self.parameters.model_dump(exclude_none=True)  # type: ignore[attr-defined]
        return {
            "device_id": self.device_id,
            "clock_domain": self.clock_domain,
            **parameters,
        }


class HardwareUltrasoundDeviceProfile(HardwareDeviceProfileBase):
    modality: Literal["ultrasound"]
    adapter: Literal[ELONXI_ULTRASOUND_ADAPTER, RAW_ETHERNET_ULTRASOUND_ADAPTER]
    writer: Literal["block_binary"]
    simulated: Literal[False]
    parameters: RawEthernetUltrasoundParameters | HardwareUltrasoundParameters

    @model_validator(mode="before")
    @classmethod
    def dispatch_parameters(cls, data: Any) -> Any:
        if isinstance(data, dict):
            adapter = data.get("adapter", "")
            params = data.get("parameters")
            if isinstance(params, dict):
                if adapter == RAW_ETHERNET_ULTRASOUND_ADAPTER:
                    data["parameters"] = RawEthernetUltrasoundParameters(**dict(params))
                elif adapter == ELONXI_ULTRASOUND_ADAPTER:
                    data["parameters"] = HardwareUltrasoundParameters(**dict(params))
        return data


class HardwareImuDeviceProfile(HardwareDeviceProfileBase):
    modality: Literal["imu"]
    adapter: Literal[XSENS_AWINDA_ADAPTER]
    writer: Literal["hdf5_signal"]
    simulated: Literal[False]
    parameters: HardwareImuParameters


class HardwareEncoderDeviceProfile(HardwareDeviceProfileBase):
    modality: Literal["encoder"]
    adapter: Literal[TEENSY_ENCODER_ADAPTER]
    writer: Literal["hdf5_signal"]
    simulated: Literal[False]
    parameters: HardwareEncoderParameters


class HardwareSyncPulseDeviceProfile(DeviceProfileBase):
    modality: Literal["sync_pulse"]
    adapter: Literal[SYNC_PULSE_ADAPTER]
    writer: Literal["hdf5_signal"]
    simulated: Literal[True]
    parameters: SyncPulseSimulationParameters


HardwareDeviceProfile: TypeAlias = Annotated[
    HardwareUltrasoundDeviceProfile
    | HardwareImuDeviceProfile
    | HardwareEncoderDeviceProfile
    | HardwareSyncPulseDeviceProfile,
    Field(discriminator="modality"),
]


class HardwareDeviceProfileDocument(ProfileModel):
    profile_kind: Literal["hardware"]
    display_name: NonEmptyStr = "真实三设备 + 模拟同步（台架验证）"
    schema_version: Literal["1.0.0"]
    laboratory_sync_ready: Literal[False]
    devices: tuple[HardwareDeviceProfile, ...]

    @model_validator(mode="after")
    def validate_complete_profile(self) -> HardwareDeviceProfileDocument:
        modalities = [device.modality for device in self.devices]
        expected = {"ultrasound", "imu", "encoder", "sync_pulse"}
        if set(modalities) != expected or len(modalities) != len(expected):
            raise ValueError(
                "hardware profile must define ultrasound, imu, encoder, and sync_pulse exactly once"
            )
        device_ids = [device.device_id for device in self.devices]
        clock_domains = [device.clock_domain for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("hardware device ids must be unique")
        if len(clock_domains) != len(set(clock_domains)):
            raise ValueError("hardware clock domains must be unique")
        return self

    def by_modality(self) -> dict[str, HardwareDeviceProfile]:
        return {device.modality: device for device in self.devices}


DeviceProfileDocument: TypeAlias = (
    SimulatedDeviceProfileDocument | HardwareDeviceProfileDocument
)


def _default_profile_candidates() -> list[Path]:
    candidates: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root) / "config" / "devices" / "simulated.json")
    candidates.extend(
        [
            Path(sys.executable).resolve().parent / "config" / "devices" / "simulated.json",
            Path(__file__).resolve().parents[3] / "config" / "devices" / "simulated.json",
            Path.cwd() / "config" / "devices" / "simulated.json",
        ]
    )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def default_simulated_device_profile_path() -> Path:
    """Locate the source-tree or PyInstaller-bundled simulated profile."""

    candidates = _default_profile_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"simulated device profile not found; searched:\n{searched}")


def load_simulated_device_profile(
    path: str | Path | None = None,
) -> SimulatedDeviceProfileDocument:
    """Read and strictly validate ``config/devices/simulated.json``."""

    source = (
        default_simulated_device_profile_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid simulated device profile JSON at {source}: {exc}") from exc
    return SimulatedDeviceProfileDocument.model_validate(payload)


def default_device_profile_path(key: Literal["simulated", "hardware"]) -> Path:
    filename = f"{key}.json"
    candidates = [candidate.with_name(filename) for candidate in _default_profile_candidates()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"{key} device profile not found; searched:\n{searched}")


def load_device_profile(
    key_or_path: Literal["simulated", "hardware"] | str | Path = "simulated",
) -> DeviceProfileDocument:
    text = str(key_or_path)
    source = (
        default_device_profile_path(text)  # type: ignore[arg-type]
        if text in {"simulated", "hardware"}
        else Path(key_or_path).expanduser().resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid device profile JSON at {source}: {exc}") from exc
    profile_kind = payload.get("profile_kind", "simulated")
    if profile_kind == "simulated":
        return SimulatedDeviceProfileDocument.model_validate(payload)
    if profile_kind == "hardware":
        return HardwareDeviceProfileDocument.model_validate(payload)
    raise ValueError(f"unsupported device profile kind: {profile_kind!r}")
