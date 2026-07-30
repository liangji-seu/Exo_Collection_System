"""Deterministic marker trajectories for UI and recording tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from exo_collection.adapters.base import (
    ModalityDescriptor,
    QueuedSimulatedAdapter,
    SimulationConfig,
)
from exo_collection.domain.events import SampleBatch


MOCAP_AXES = ("x", "y", "z")
MOCAP_UNITS = ("mm", "mm", "mm")


@dataclass(frozen=True, slots=True)
class SimulatedMocapConfig(SimulationConfig):
    device_id: str = "mocap_sim"
    clock_domain: str = "mocap_sim_clock"
    sample_rate_hz: float = 100.0
    marker_count: int = 12
    samples_per_batch: int = 5

    def __post_init__(self) -> None:
        SimulationConfig.__post_init__(self)
        if self.sample_rate_hz <= 0 or self.marker_count <= 0:
            raise ValueError("sample_rate_hz and marker_count must be positive")
        if self.samples_per_batch <= 0:
            raise ValueError("samples_per_batch must be positive")


class SimulatedMocapAdapter(QueuedSimulatedAdapter[SimulatedMocapConfig]):
    config_type = SimulatedMocapConfig

    @property
    def _rate_hz(self) -> float:
        return self._config.sample_rate_hz

    @property
    def _items_per_batch(self) -> int:
        return self._config.samples_per_batch

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        marker_names = [f"marker_{index + 1:02d}" for index in range(cfg.marker_count)]
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="mocap",
            display_name="Simulated XING marker stream",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=MOCAP_AXES,
            units=MOCAP_UNITS,
            nominal_rate_hz=cfg.sample_rate_hz,
            sample_shape=(cfg.marker_count, 3),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": True,
                "coordinate_unit": "mm",
                "marker_names": marker_names,
                "marker_sets": [{"name": "simulated", "marker_names": marker_names}],
            },
        )

    def _make_events(
        self,
        *,
        sequence: int,
        first_item_index: int,
        host_monotonic_ns: int,
    ) -> list[SampleBatch]:
        cfg = self._config
        indices = first_item_index + np.arange(cfg.samples_per_batch)
        t = indices.astype(np.float64) / cfg.sample_rate_hz
        phase = 2.0 * np.pi * 0.8 * t[:, None]
        marker_phase = np.linspace(0.0, 2.0 * np.pi, cfg.marker_count, endpoint=False)[None, :]
        data = np.empty((cfg.samples_per_batch, cfg.marker_count, 3), dtype=np.float32)
        data[..., 0] = 500.0 * np.sin(phase + marker_phase)
        data[..., 1] = 200.0 * np.cos(phase + marker_phase)
        data[..., 2] = 900.0 + 80.0 * np.sin(phase * 2.0 + marker_phase)
        return [
            SampleBatch(
                **self._event_common(host_monotonic_ns),
                first_sample_index=first_item_index,
                sample_count=cfg.samples_per_batch,
                sequence_number=sequence,
                device_timestamp=first_item_index,
                sample_rate_hz=cfg.sample_rate_hz,
                data=np.ascontiguousarray(data),
            )
        ]


__all__ = [
    "MOCAP_AXES",
    "MOCAP_UNITS",
    "SimulatedMocapAdapter",
    "SimulatedMocapConfig",
]
