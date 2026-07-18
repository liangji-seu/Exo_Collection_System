"""IMU modality adapters."""

from .simulated import IMU_CHANNELS, IMU_UNITS, SimulatedImuAdapter, SimulatedImuConfig
from .xsens_awinda import (
    XdaAwindaBackend,
    XsensAwindaConfig,
    XsensAwindaImuAdapter,
    parse_xsens_packet,
)

__all__ = [
    "IMU_CHANNELS",
    "IMU_UNITS",
    "SimulatedImuAdapter",
    "SimulatedImuConfig",
    "XdaAwindaBackend",
    "XsensAwindaConfig",
    "XsensAwindaImuAdapter",
    "parse_xsens_packet",
]
