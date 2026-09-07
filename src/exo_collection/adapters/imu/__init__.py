"""IMU modality adapters."""

from .simulated import IMU_CHANNELS, IMU_UNITS, SimulatedImuAdapter, SimulatedImuConfig
from .xsens_awinda import (
    XdaAwindaBackend,
    XsensAwindaConfig,
    XsensAwindaImuAdapter,
    parse_xsens_packet,
)
from .xsens_mtw_usb import (
    MtwUsbBackend,
    XdaMtwUsbBackend,
    XsensMtwUsbConfig,
    XsensMtwUsbImuAdapter,
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
    "MtwUsbBackend",
    "XdaMtwUsbBackend",
    "XsensMtwUsbConfig",
    "XsensMtwUsbImuAdapter",
]
