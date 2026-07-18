"""Unified modality adapter API and built-in simulators."""

from .base import (
    AdapterError,
    AdapterLifecycleError,
    AdapterState,
    DeviceConfig,
    ModalityAdapter,
    ModalityDescriptor,
    PreparedInfo,
    RawQueueOverflowError,
    SimulationConfig,
    StartToken,
    StopReport,
    TrialContext,
)
from .encoder import (
    ENCODER_CHANNELS,
    ENCODER_UNITS,
    SimulatedEncoderAdapter,
    SimulatedEncoderConfig,
    TeensyEncoderConfig,
    TeensySerialEncoderAdapter,
)
from .imu import (
    IMU_CHANNELS,
    IMU_UNITS,
    SimulatedImuAdapter,
    SimulatedImuConfig,
    XsensAwindaConfig,
    XsensAwindaImuAdapter,
)
from .hardware_base import QueuedHardwareAdapter
from .simulated import (
    SimulatedIMUAdapter,
    SimulatedIMUConfig,
    SimulatedSyncPulseAdapter,
    SimulatedSyncPulseConfig,
    SimulatedUltrasoundAdapter,
    SimulatedUltrasoundConfig,
)
from .ultrasound import ElonxiUltrasoundAdapter, ElonxiUltrasoundConfig

__all__ = [
    "AdapterError",
    "AdapterLifecycleError",
    "AdapterState",
    "DeviceConfig",
    "ENCODER_CHANNELS",
    "ENCODER_UNITS",
    "IMU_CHANNELS",
    "IMU_UNITS",
    "ModalityAdapter",
    "ModalityDescriptor",
    "QueuedHardwareAdapter",
    "PreparedInfo",
    "RawQueueOverflowError",
    "SimulatedEncoderAdapter",
    "SimulatedEncoderConfig",
    "SimulatedIMUAdapter",
    "SimulatedIMUConfig",
    "SimulatedImuAdapter",
    "SimulatedImuConfig",
    "SimulatedSyncPulseAdapter",
    "SimulatedSyncPulseConfig",
    "SimulatedUltrasoundAdapter",
    "SimulatedUltrasoundConfig",
    "TeensyEncoderConfig",
    "TeensySerialEncoderAdapter",
    "SimulationConfig",
    "StartToken",
    "StopReport",
    "TrialContext",
    "ElonxiUltrasoundAdapter",
    "ElonxiUltrasoundConfig",
    "XsensAwindaConfig",
    "XsensAwindaImuAdapter",
]
