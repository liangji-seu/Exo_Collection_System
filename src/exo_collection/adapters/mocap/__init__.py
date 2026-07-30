"""Motion-capture adapters."""

from .simulated import SimulatedMocapAdapter, SimulatedMocapConfig
from .xing_nokov import XingNokovMocapAdapter, XingNokovMocapConfig

__all__ = [
    "SimulatedMocapAdapter",
    "SimulatedMocapConfig",
    "XingNokovMocapAdapter",
    "XingNokovMocapConfig",
]
