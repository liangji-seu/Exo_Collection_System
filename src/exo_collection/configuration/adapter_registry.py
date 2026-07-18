"""Fixed in-process adapter registry; JSON never controls Python imports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from exo_collection.adapters.base import ModalityAdapter
from exo_collection.adapters.encoder.simulated import SimulatedEncoderAdapter
from exo_collection.adapters.encoder.teensy_serial import TeensySerialEncoderAdapter
from exo_collection.adapters.imu.simulated import SimulatedImuAdapter
from exo_collection.adapters.imu.xsens_awinda import XsensAwindaImuAdapter
from exo_collection.adapters.sync_pulse.simulated import SimulatedSyncPulseAdapter
from exo_collection.adapters.ultrasound.elonxi import ElonxiUltrasoundAdapter
from exo_collection.adapters.ultrasound.simulated import SimulatedUltrasoundAdapter

from .device_profiles import (
    DeviceProfileDocument,
    ELONXI_ULTRASOUND_ADAPTER,
    ENCODER_ADAPTER,
    IMU_ADAPTER,
    SYNC_PULSE_ADAPTER,
    TEENSY_ENCODER_ADAPTER,
    ULTRASOUND_ADAPTER,
    XSENS_AWINDA_ADAPTER,
)


ADAPTER_REGISTRY: dict[str, type[Any]] = {
    ULTRASOUND_ADAPTER: SimulatedUltrasoundAdapter,
    IMU_ADAPTER: SimulatedImuAdapter,
    ENCODER_ADAPTER: SimulatedEncoderAdapter,
    SYNC_PULSE_ADAPTER: SimulatedSyncPulseAdapter,
    ELONXI_ULTRASOUND_ADAPTER: ElonxiUltrasoundAdapter,
    XSENS_AWINDA_ADAPTER: XsensAwindaImuAdapter,
    TEENSY_ENCODER_ADAPTER: TeensySerialEncoderAdapter,
}


def build_adapters(
    profile: DeviceProfileDocument,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, ModalityAdapter]:
    """Validate per-modality overrides and instantiate only approved classes."""

    requested = dict(overrides or {})
    devices = profile.by_modality()
    unknown_modalities = set(requested) - set(devices)
    if unknown_modalities:
        display = ", ".join(sorted(unknown_modalities))
        raise ValueError(f"unknown device override modality: {display}")

    adapters: dict[str, ModalityAdapter] = {}
    for modality, device in devices.items():
        adapter_type = ADAPTER_REGISTRY.get(device.adapter)
        if adapter_type is None:
            raise ValueError(f"adapter identifier is not registered: {device.adapter}")
        override = requested.get(modality, {})
        if not isinstance(override, Mapping):
            raise TypeError(f"override for {modality} must be a mapping")
        base_parameters = device.parameters.model_dump(exclude_none=True)
        parameter_type = type(device.parameters)
        validated = parameter_type.model_validate({**base_parameters, **dict(override)})
        configuration = {
            "device_id": device.device_id,
            "clock_domain": device.clock_domain,
            **validated.model_dump(exclude_none=True),
        }
        adapters[modality] = adapter_type(configuration)
    return adapters


__all__ = ["ADAPTER_REGISTRY", "build_adapters"]
