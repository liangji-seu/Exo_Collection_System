"""Persistable affine mappings from device/external clocks to Trial time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ClockResiduals:
    count: int
    mean_ns: float
    rms_ns: float
    standard_deviation_ns: float
    p95_absolute_ns: float
    max_absolute_ns: float


class DeviceClockMapper:
    """Streaming device-counter → host-monotonic-ns mapper.

    Reconstructs a uniform per-sample host timestamp from a device clock
    (e.g. an Xsens PacketCounter that increments by one per sample and wraps
    at ``wrap_mod``) instead of the bursty host packet-arrival time.  The
    scale is fixed to the nominal period ``period_ns``; the offset is a
    running mean of ``arrival_ns - period_ns * unwrapped_counter`` that is
    frozen once ``warmup`` anchors have been seen.

    - The first sample's offset is seeded from its own residual, so the first
      ``map`` call returns ``arrival_ns`` exactly (no discontinuity).
    - After ``warmup`` samples the offset is frozen, so every subsequent
      mapped value is spaced exactly ``period_ns`` apart regardless of bursty
      arrival jitter (the jitter is averaged away during warm-up).
    - Fixed scale (rather than a streaming least-squares slope) avoids wild
      early fits dominated by jitter; crystal drift (~20 ppm) is negligible
      over a single trial.  Extend here for drift tracking if ever needed.
    """

    def __init__(
        self, period_ns: float, wrap_mod: int = 65536, warmup: int = 16
    ) -> None:
        if period_ns <= 0 or not np.isfinite(period_ns):
            raise ValueError("period_ns must be positive and finite")
        if wrap_mod <= 0:
            raise ValueError("wrap_mod must be positive")
        if warmup <= 0:
            raise ValueError("warmup must be positive")
        self._period_ns = float(period_ns)
        self._wrap_mod = int(wrap_mod)
        self._warmup = int(warmup)
        self._n = 0
        self._last_raw: int | None = None
        self._unwrapped: int | None = None
        self._offset = 0.0

    def map(self, raw_counter: int, arrival_ns: int) -> int:
        """Record one anchor and return the reconstructed host ns for it."""
        self._n += 1
        if self._unwrapped is None:
            self._unwrapped = int(raw_counter)
        else:
            delta = int(raw_counter) - int(self._last_raw)
            if delta < -self._wrap_mod // 2:
                delta += self._wrap_mod
            elif delta > self._wrap_mod // 2:
                delta -= self._wrap_mod
            self._unwrapped += delta
        self._last_raw = int(raw_counter)

        if self._n <= self._warmup:
            residual = float(arrival_ns) - self._period_ns * float(self._unwrapped)
            self._offset += (residual - self._offset) / self._n
        return int(round(self._period_ns * float(self._unwrapped) + self._offset))


@dataclass(frozen=True, slots=True)
class AffineClockModel:
    """Mapping ``t_global_ns = scale_a * t_source + offset_b_ns``."""

    scale_a: float
    offset_b_ns: float
    anchor_count: int
    source_start: float
    source_end: float
    residuals: ClockResiduals
    algorithm_version: str = "affine-least-squares-1.0.0"

    def map(self, source_time: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(source_time, dtype=np.float64)
        return values * self.scale_a + self.offset_b_ns

    def map_one(self, source_time: int | float) -> float:
        return float(source_time) * self.scale_a + self.offset_b_ns


def fit_affine_clock(
    source_times: Iterable[int | float],
    global_times_ns: Iterable[int | float],
) -> AffineClockModel:
    """Fit drift and offset; one shared pulse intentionally estimates offset only."""

    source = np.asarray(list(source_times), dtype=np.float64)
    target = np.asarray(list(global_times_ns), dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or source.size != target.size:
        raise ValueError("source and global anchors must be equally sized vectors")
    if source.size == 0:
        raise ValueError("at least one clock anchor is required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("clock anchors must be finite")
    if source.size > 1 and np.any(np.diff(source) <= 0):
        raise ValueError("source clock anchors must be strictly increasing")
    if source.size > 1 and np.any(np.diff(target) <= 0):
        raise ValueError("global clock anchors must be strictly increasing")

    if source.size == 1:
        scale = 1.0
        offset = float(target[0] - source[0])
        algorithm = "single-anchor-offset-1.0.0"
    else:
        centered_source = source - source.mean()
        denominator = float(np.dot(centered_source, centered_source))
        if denominator == 0:
            raise ValueError("clock anchors do not span a source interval")
        scale = float(np.dot(centered_source, target - target.mean()) / denominator)
        if scale <= 0:
            raise ValueError("fitted clock scale is not positive")
        offset = float(target.mean() - scale * source.mean())
        algorithm = "affine-least-squares-1.0.0"

    residual = target - (scale * source + offset)
    absolute = np.abs(residual)
    stats = ClockResiduals(
        count=int(source.size),
        mean_ns=float(residual.mean()),
        rms_ns=float(np.sqrt(np.mean(np.square(residual)))),
        standard_deviation_ns=float(residual.std()),
        p95_absolute_ns=float(np.percentile(absolute, 95)),
        max_absolute_ns=float(absolute.max()),
    )
    return AffineClockModel(
        scale_a=scale,
        offset_b_ns=offset,
        anchor_count=int(source.size),
        source_start=float(source[0]),
        source_end=float(source[-1]),
        residuals=stats,
        algorithm_version=algorithm,
    )

