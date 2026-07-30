"""Deterministic multi-channel EMG source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exo_collection.adapters.base import (
    ModalityDescriptor,
    QueuedSimulatedAdapter,
    SimulationConfig,
)
from exo_collection.domain.events import SampleBatch


@dataclass(frozen=True, slots=True)
class SimulatedEmgConfig(SimulationConfig):
    device_id: str = "emg_sim"
    clock_domain: str = "emg_sim_clock"
    sample_rate_hz: float = 1000.0
    channel_count: int = 8
    samples_per_batch: int = 20
    amplitude_mv: float = 1.0

    def __post_init__(self) -> None:
        SimulationConfig.__post_init__(self)
        if self.sample_rate_hz <= 0 or self.channel_count <= 0:
            raise ValueError("sample_rate_hz and channel_count must be positive")
        if self.samples_per_batch <= 0 or self.amplitude_mv <= 0:
            raise ValueError("samples_per_batch and amplitude_mv must be positive")


class SimulatedEmgAdapter(QueuedSimulatedAdapter[SimulatedEmgConfig]):
    config_type = SimulatedEmgConfig

    @property
    def _rate_hz(self) -> float:
        return self._config.sample_rate_hz

    @property
    def _items_per_batch(self) -> int:
        return self._config.samples_per_batch

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        names = tuple(f"emg_{index + 1:02d}" for index in range(cfg.channel_count))
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="emg",
            display_name="Simulated XING analog/EMG stream",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=names,
            units=tuple("mV" for _ in names),
            nominal_rate_hz=cfg.sample_rate_hz,
            sample_shape=(cfg.channel_count,),
            dtype=np.dtype(np.float32).str,
            metadata={"simulated": True, "channel_names": list(names)},
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
        channel_phase = np.arange(cfg.channel_count, dtype=np.float64)[None, :] * 0.73
        carrier = np.sin(2.0 * np.pi * 70.0 * t[:, None] + channel_phase)
        envelope = 0.15 + 0.85 * np.square(
            np.sin(2.0 * np.pi * 0.8 * t[:, None] + channel_phase)
        )
        noise = self._rng_values.normal(0.0, 0.04, carrier.shape)
        data = cfg.amplitude_mv * (carrier * envelope + noise)
        return [
            SampleBatch(
                **self._event_common(host_monotonic_ns),
                first_sample_index=first_item_index,
                sample_count=cfg.samples_per_batch,
                sequence_number=sequence,
                device_timestamp=first_item_index,
                sample_rate_hz=cfg.sample_rate_hz,
                data=np.ascontiguousarray(data.astype(np.float32)),
            )
        ]


__all__ = ["SimulatedEmgAdapter", "SimulatedEmgConfig"]
