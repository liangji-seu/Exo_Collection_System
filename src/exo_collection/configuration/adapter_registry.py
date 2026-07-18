"""Fixed in-process adapter registry; JSON never controls Python imports."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from exo_collection.adapters.base import ModalityAdapter
from exo_collection.adapters.encoder.simulated import SimulatedEncoderAdapter
from exo_collection.adapters.encoder.teensy_serial import TeensySerialEncoderAdapter
from exo_collection.adapters.imu.simulated import SimulatedImuAdapter
from exo_collection.adapters.imu.xsens_awinda import XsensAwindaImuAdapter
from exo_collection.adapters.sync_pulse.simulated import SimulatedSyncPulseAdapter
from exo_collection.adapters.ultrasound.elonxi import ElonxiUltrasoundAdapter
from exo_collection.adapters.ultrasound.raw_ethernet import RawEthernetUltrasoundAdapter
from exo_collection.adapters.ultrasound.simulated import SimulatedUltrasoundAdapter

_log = logging.getLogger(__name__)

from .device_profiles import (
    DeviceProfileDocument,
    ELONXI_ULTRASOUND_ADAPTER,
    ENCODER_ADAPTER,
    IMU_ADAPTER,
    RAW_ETHERNET_ULTRASOUND_ADAPTER,
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
    RAW_ETHERNET_ULTRASOUND_ADAPTER: RawEthernetUltrasoundAdapter,
    XSENS_AWINDA_ADAPTER: XsensAwindaImuAdapter,
    TEENSY_ENCODER_ADAPTER: TeensySerialEncoderAdapter,
}


def build_adapter(
    profile: DeviceProfileDocument,
    modality: str,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> ModalityAdapter:
    """Instantiate one validated, registry-approved modality adapter."""

    requested = dict(overrides or {})
    devices = profile.by_modality()
    unknown_modalities = set(requested) - set(devices)
    if unknown_modalities:
        display = ", ".join(sorted(unknown_modalities))
        raise ValueError(f"unknown device override modality: {display}")
    try:
        device = devices[modality]
    except KeyError as exc:
        raise ValueError(f"device profile has no modality: {modality}") from exc
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
    _log.debug(
        "building %s adapter type=%s config_keys=%s",
        modality,
        adapter_type.__name__,
        sorted(configuration),
    )
    return adapter_type(configuration)


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

    return {
        modality: build_adapter(profile, modality, requested)
        for modality in devices
    }


__all__ = ["ADAPTER_REGISTRY", "build_adapter", "build_adapters"]
