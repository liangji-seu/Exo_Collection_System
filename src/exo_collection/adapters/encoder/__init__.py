"""Encoder modality adapters."""

from .simulated import (
    ENCODER_CHANNELS,
    ENCODER_UNITS,
    SimulatedEncoderAdapter,
    SimulatedEncoderConfig,
)
from .teensy_serial import (
    MotorStatusFrame,
    MotorStatusStreamParser,
    TeensyEncoderConfig,
    TeensySerialEncoderAdapter,
    calc_crc8,
    parse_status_frame,
)

__all__ = [
    "ENCODER_CHANNELS",
    "ENCODER_UNITS",
    "SimulatedEncoderAdapter",
    "SimulatedEncoderConfig",
    "MotorStatusFrame",
    "MotorStatusStreamParser",
    "TeensyEncoderConfig",
    "TeensySerialEncoderAdapter",
    "calc_crc8",
    "parse_status_frame",
]
