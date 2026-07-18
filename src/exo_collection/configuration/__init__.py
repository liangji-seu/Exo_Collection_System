"""Validated local configuration contracts."""

from .app_settings import SharedAppSettings, default_data_root
from .adapter_registry import ADAPTER_REGISTRY, build_adapters
from .device_profiles import (
    DeviceProfileDocument,
    HardwareDeviceProfileDocument,
    SimulatedDeviceProfile,
    SimulatedDeviceProfileDocument,
    default_device_profile_path,
    default_simulated_device_profile_path,
    load_device_profile,
    load_simulated_device_profile,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "DeviceProfileDocument",
    "HardwareDeviceProfileDocument",
    "SharedAppSettings",
    "SimulatedDeviceProfile",
    "SimulatedDeviceProfileDocument",
    "build_adapters",
    "default_data_root",
    "default_device_profile_path",
    "default_simulated_device_profile_path",
    "load_device_profile",
    "load_simulated_device_profile",
]
