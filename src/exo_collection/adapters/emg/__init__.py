"""Electromyography adapters."""

from .simulated import SimulatedEmgAdapter, SimulatedEmgConfig
from .xing_nokov import XingNokovEmgAdapter, XingNokovEmgConfig

__all__ = [
    "SimulatedEmgAdapter",
    "SimulatedEmgConfig",
    "XingNokovEmgAdapter",
    "XingNokovEmgConfig",
]
