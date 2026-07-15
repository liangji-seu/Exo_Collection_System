"""Deterministic configurable ultrasound simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from exo_collection.adapters.base import (
    ModalityDescriptor,
    QueuedSimulatedAdapter,
    SimulationConfig,
)
from exo_collection.domain.events import FrameBatch


@dataclass(frozen=True, slots=True)
class SimulatedUltrasoundConfig(SimulationConfig):
    """Ultrasound shape/rate plus shared fault-injection controls.

    The default mirrors the first hardware prototype (4 A-line channels, 1000
    samples, approximately 100 frames/s).  ``frame_shape=(64, 64)`` and
    ``frame_rate_hz=30`` selects the milestone B-mode-like configuration.
    """

    device_id: str = "ultrasound_sim"
    clock_domain: str = "ultrasound_sim_clock"
    frame_rate_hz: float = 100.0
    frames_per_batch: int = 1
    channel_count: int = 4
    samples_per_channel: int = 1000
    frame_shape: tuple[int, ...] | None = None
    dtype: str = "uint16"
    baseline: float = 80.0
    echo_amplitude: float = 1800.0
    noise_std: float = 12.0

    def __post_init__(self) -> None:
        SimulationConfig.__post_init__(self)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if self.frame_rate_hz <= 0 or not np.isfinite(self.frame_rate_hz):
            raise ValueError("frame_rate_hz must be positive and finite")
        if self.frames_per_batch <= 0:
            raise ValueError("frames_per_batch must be positive")
        if self.channel_count <= 0 or self.samples_per_channel <= 0:
            raise ValueError("channel_count and samples_per_channel must be positive")
        if self.frame_shape is not None:
            shape = tuple(int(v) for v in self.frame_shape)
            if not shape or any(v <= 0 for v in shape):
                raise ValueError("frame_shape dimensions must be positive")
            object.__setattr__(self, "frame_shape", shape)
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "uif":
            raise ValueError("ultrasound dtype must be unsigned, signed, or floating point")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

    @property
    def resolved_frame_shape(self) -> tuple[int, ...]:
        return self.frame_shape or (self.channel_count, self.samples_per_channel)


class SimulatedUltrasoundAdapter(QueuedSimulatedAdapter[SimulatedUltrasoundConfig]):
    config_type = SimulatedUltrasoundConfig

    def __init__(
        self,
        config: SimulatedUltrasoundConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config)

    @property
    def _rate_hz(self) -> float:
        return self._config.frame_rate_hz

    @property
    def _items_per_batch(self) -> int:
        return self._config.frames_per_batch

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        if cfg.frame_shape is None:
            channels = tuple(f"ch_{index + 1}" for index in range(cfg.channel_count))
            units = ("a.u.",) * cfg.channel_count
            geometry = "a_line"
        else:
            channels = ("amplitude",)
            units = ("a.u.",)
            geometry = "frame"
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="ultrasound",
            display_name="Simulated ultrasound",
            clock_domain=cfg.clock_domain,
            event_kind="frame_batch",
            channels=channels,
            units=units,
            nominal_rate_hz=cfg.frame_rate_hz,
            sample_shape=cfg.resolved_frame_shape,
            dtype=np.dtype(cfg.dtype).str,
            metadata={
                "simulated": True,
                "geometry": geometry,
                "frame_shape": list(cfg.resolved_frame_shape),
                "frames_per_batch": cfg.frames_per_batch,
            },
        )

    def _make_events(
        self,
        *,
        sequence: int,
        first_item_index: int,
        host_monotonic_ns: int,
    ) -> list[FrameBatch]:
        cfg = self._config
        shape = (cfg.frames_per_batch, *cfg.resolved_frame_shape)
        grid = np.indices(cfg.resolved_frame_shape, dtype=np.float64)
        spatial = np.zeros(cfg.resolved_frame_shape, dtype=np.float64)
        for axis, coords in enumerate(grid):
            denom = max(1, cfg.resolved_frame_shape[axis] - 1)
            spatial += 0.08 * np.sin(2.0 * np.pi * (axis + 1) * coords / denom)

        generated = np.empty(shape, dtype=np.float64)
        depth_axis = np.linspace(0.0, 1.0, cfg.resolved_frame_shape[-1])
        depth_shape = (1,) * (len(cfg.resolved_frame_shape) - 1) + (depth_axis.size,)
        for offset in range(cfg.frames_per_batch):
            frame_index = first_item_index + offset
            moving_center = 0.28 + 0.12 * np.sin(frame_index * 0.037)
            echo = np.exp(-0.5 * ((depth_axis - moving_center) / 0.035) ** 2).reshape(depth_shape)
            harmonic = 0.35 * np.exp(
                -0.5 * ((depth_axis - min(0.92, moving_center + 0.24)) / 0.055) ** 2
            ).reshape(depth_shape)
            noise = self._rng_values.normal(0.0, cfg.noise_std, cfg.resolved_frame_shape)
            generated[offset] = (
                cfg.baseline
                + cfg.echo_amplitude * (echo + harmonic) * (1.0 + spatial)
                + noise
            )

        dtype = np.dtype(cfg.dtype)
        if dtype.kind in "ui":
            info = np.iinfo(dtype)
            generated = np.clip(np.rint(generated), info.min, info.max)
        data = np.ascontiguousarray(generated.astype(dtype, copy=False))
        event = FrameBatch(
            **self._event_common(host_monotonic_ns),
            first_frame_index=first_item_index,
            frame_count=cfg.frames_per_batch,
            sequence_number=sequence,
            device_timestamp=self.device_time_ns(first_item_index, cfg.frame_rate_hz),
            frame_rate_hz=cfg.frame_rate_hz,
            data=data,
        )
        return [event]


__all__ = ["SimulatedUltrasoundAdapter", "SimulatedUltrasoundConfig"]
