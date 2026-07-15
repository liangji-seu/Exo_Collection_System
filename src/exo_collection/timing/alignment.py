"""Shared-pulse pairing for external artifact clock alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .clock_model import AffineClockModel, fit_affine_clock


@dataclass(frozen=True, slots=True)
class PulseAnchor:
    pulse_id: str
    source_time: int | float
    host_monotonic_ns: int


def align_shared_pulses(
    external_pulses: Iterable[tuple[str, int | float]],
    host_pulses: Iterable[tuple[str, int]],
) -> tuple[AffineClockModel, tuple[str, ...]]:
    """Pair unique pulse IDs and fit an external→host monotonic mapping."""

    external = dict(external_pulses)
    host = dict(host_pulses)
    if len(external) == 0 or len(host) == 0:
        raise ValueError("both clock domains need at least one pulse")
    shared = tuple(sorted(set(external) & set(host), key=lambda key: external[key]))
    if not shared:
        raise ValueError("no shared pulse IDs were found")
    model = fit_affine_clock(
        (external[pulse_id] for pulse_id in shared),
        (host[pulse_id] for pulse_id in shared),
    )
    return model, shared

