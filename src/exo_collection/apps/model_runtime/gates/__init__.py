"""Frozen online suitability gates for the torque runtime."""

from .contract import GateDecision, GateEngine
from .loader import load_gate

__all__ = ["GateDecision", "GateEngine", "load_gate"]
