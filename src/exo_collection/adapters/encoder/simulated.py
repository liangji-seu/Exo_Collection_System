"""Deterministic bilateral encoder/motor-feedback simulator."""

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


ENCODER_CHANNELS = (
    "left_position",
    "left_velocity",
    "left_torque",
    "right_position",
    "right_velocity",
    "right_torque",
)
ENCODER_UNITS = ("rad", "rad/s", "N*m", "rad", "rad/s", "N*m")


@dataclass(frozen=True, slots=True)
class SimulatedEncoderConfig(SimulationConfig):
    device_id: str = "encoder_sim"
    clock_domain: str = "encoder_sim_clock"
    sample_rate_hz: float = 200.0
    samples_per_batch: int = 20
    hardware_tick_hz: float = 1000.0
    hardware_tick_modulus: int = 2**32
    sequence_modulus: int = 2**16
    gait_frequency_hz: float = 0.9
    noise_std: float = 0.001

    def __post_init__(self) -> None:
        SimulationConfig.__post_init__(self)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        for name, value in (
            ("sample_rate_hz", self.sample_rate_hz),
            ("hardware_tick_hz", self.hardware_tick_hz),
            ("gait_frequency_hz", self.gait_frequency_hz),
        ):
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
        if self.samples_per_batch <= 0:
            raise ValueError("samples_per_batch must be positive")
        if self.hardware_tick_modulus <= 1 or self.sequence_modulus <= 1:
            raise ValueError("tick and sequence moduli must exceed one")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


class SimulatedEncoderAdapter(QueuedSimulatedAdapter[SimulatedEncoderConfig]):
    """Produces six physical channels plus sequence/tick event metadata.

    ``first_sample_index`` is the unwrapped sample/sequence counter and
    ``device_timestamp`` is the first sample's wrapped hardware tick.  Thus the
    signal array stays numeric and the HDF5 writer can place those counters in
    ``/samples/sample_index`` and ``/samples/device_time`` without duplicating
    them as floating-point channels.
    """

    config_type = SimulatedEncoderConfig

    def __init__(
        self, config: SimulatedEncoderConfig | Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(config)

    @property
    def _rate_hz(self) -> float:
        return self._config.sample_rate_hz

    @property
    def _items_per_batch(self) -> int:
        return self._config.samples_per_batch

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="encoder",
            display_name="Simulated bilateral motor encoder feedback",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=ENCODER_CHANNELS,
            units=ENCODER_UNITS,
            nominal_rate_hz=cfg.sample_rate_hz,
            sample_shape=(len(ENCODER_CHANNELS),),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": True,
                "sides": ["left", "right"],
                "sequence_counter": "first_sample_index",
                "sequence_modulus": cfg.sequence_modulus,
                "device_timestamp": "hardware_tick",
                "device_time_unit": "tick",
                "hardware_tick_hz": cfg.hardware_tick_hz,
                "hardware_tick_modulus": cfg.hardware_tick_modulus,
                "samples_per_batch": cfg.samples_per_batch,
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
        indices = first_item_index + np.arange(cfg.samples_per_batch, dtype=np.float64)
        t = indices / cfg.sample_rate_hz
        omega = 2.0 * np.pi * cfg.gait_frequency_hz
        left_phase = omega * t
        right_phase = left_phase + np.pi

        data = np.column_stack(
            (
                0.45 * np.sin(left_phase),
                0.45 * omega * np.cos(left_phase),
                2.2 * np.sin(left_phase - 0.35),
                0.45 * np.sin(right_phase),
                0.45 * omega * np.cos(right_phase),
                2.2 * np.sin(right_phase - 0.35),
            )
        )
        if cfg.noise_std:
            scales = np.asarray([1.0, 3.0, 2.5, 1.0, 3.0, 2.5])
            data += self._rng_values.normal(0.0, cfg.noise_std, data.shape) * scales

        tick_scale = 1.0 + cfg.clock_drift_ppm * 1e-6
        first_tick = int(
            round(first_item_index * cfg.hardware_tick_hz / cfg.sample_rate_hz * tick_scale)
        ) % cfg.hardware_tick_modulus
        event = SampleBatch(
            **self._event_common(host_monotonic_ns),
            first_sample_index=first_item_index,
            sample_count=cfg.samples_per_batch,
            sequence_number=sequence,
            device_timestamp=first_tick,
            sample_rate_hz=cfg.sample_rate_hz,
            data=np.ascontiguousarray(data.astype(np.float32)),
        )
        return [event]


__all__ = [
    "ENCODER_CHANNELS",
    "ENCODER_UNITS",
    "SimulatedEncoderAdapter",
    "SimulatedEncoderConfig",
]
