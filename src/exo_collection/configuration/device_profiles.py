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
