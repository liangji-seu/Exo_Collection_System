"""Convenient imports for all built-in simulated devices."""

from .encoder import SimulatedEncoderAdapter, SimulatedEncoderConfig
from .imu import SimulatedImuAdapter, SimulatedImuConfig
from .sync_pulse import SimulatedSyncPulseAdapter, SimulatedSyncPulseConfig
from .ultrasound import SimulatedUltrasoundAdapter, SimulatedUltrasoundConfig

# Common acronym spelling retained as a compatibility alias.
SimulatedIMUAdapter = SimulatedImuAdapter
SimulatedIMUConfig = SimulatedImuConfig

__all__ = [
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
]
