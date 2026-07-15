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

