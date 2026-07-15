"""Deterministic multi-device IMU simulator."""

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


IMU_CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "roll",
    "pitch",
    "yaw",
)
IMU_UNITS = (
    "m/s2",
    "m/s2",
    "m/s2",
    "rad/s",
    "rad/s",
    "rad/s",
    "a.u.",
    "a.u.",
    "a.u.",
    "deg",
    "deg",
    "deg",
)


@dataclass(frozen=True, slots=True)
class SimulatedImuConfig(SimulationConfig):
    device_id: str = "imu_sim"
    clock_domain: str = "imu_sim_clock"
    device_ids: tuple[str, ...] = ("imu_left", "imu_right")
    sample_rate_hz: float = 200.0
    samples_per_batch: int = 20
    noise_fraction: float = 0.002

    def __post_init__(self) -> None:
        SimulationConfig.__post_init__(self)
        ids = tuple(str(value) for value in self.device_ids)
        object.__setattr__(self, "device_ids", ids)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if not ids or any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("device_ids must be non-empty and unique")
        if self.sample_rate_hz <= 0 or not np.isfinite(self.sample_rate_hz):
            raise ValueError("sample_rate_hz must be positive and finite")
        if self.samples_per_batch <= 0:
            raise ValueError("samples_per_batch must be positive")
        if self.noise_fraction < 0:
            raise ValueError("noise_fraction must be non-negative")


class SimulatedImuAdapter(QueuedSimulatedAdapter[SimulatedImuConfig]):
    config_type = SimulatedImuConfig

    def __init__(self, config: SimulatedImuConfig | Mapping[str, Any] | None = None) -> None:
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
            modality="imu",
            display_name="Simulated Xsens-compatible IMU array",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=IMU_CHANNELS,
            units=IMU_UNITS,
            nominal_rate_hz=cfg.sample_rate_hz,
            sample_shape=(len(cfg.device_ids), len(IMU_CHANNELS)),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": True,
                "device_ids": list(cfg.device_ids),
                "coordinate_convention": "right-handed simulator frame",
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
        data = np.empty(
            (cfg.samples_per_batch, len(cfg.device_ids), len(IMU_CHANNELS)),
            dtype=np.float64,
        )
        scales = np.asarray(
            [2.0, 1.0, 9.81, 1.2, 0.8, 2.5, 0.5, 0.5, 0.5, 25.0, 15.0, 45.0]
        )

        for device_index, _device_id in enumerate(cfg.device_ids):
            phase = 2.0 * np.pi * (0.85 + 0.04 * device_index) * t + device_index * 0.73
            data[:, device_index, 0] = 0.9 * np.sin(phase)
            data[:, device_index, 1] = 0.35 * np.cos(phase * 0.5)
            data[:, device_index, 2] = 9.81 + 0.55 * np.sin(phase * 2.0)
            data[:, device_index, 3] = 0.45 * np.cos(phase)
            data[:, device_index, 4] = 0.22 * np.sin(phase + 0.4)
            data[:, device_index, 5] = 1.15 * np.sin(phase - 0.2)
            data[:, device_index, 6] = 0.32 + 0.025 * np.sin(phase * 0.17)
            data[:, device_index, 7] = -0.08 + 0.018 * np.cos(phase * 0.21)
            data[:, device_index, 8] = 0.41 + 0.02 * np.sin(phase * 0.13)
            data[:, device_index, 9] = 7.0 * np.sin(phase * 0.5)
            data[:, device_index, 10] = 14.0 * np.sin(phase)
            data[:, device_index, 11] = np.mod(np.degrees(phase), 360.0) - 180.0

        if cfg.noise_fraction:
            noise = self._rng_values.normal(0.0, cfg.noise_fraction, data.shape)
            data += noise * scales.reshape(1, 1, -1)
        data_f32 = np.ascontiguousarray(data.astype(np.float32))
        event = SampleBatch(
            **self._event_common(host_monotonic_ns),
            first_sample_index=first_item_index,
            sample_count=cfg.samples_per_batch,
            sequence_number=sequence,
            device_timestamp=self.device_time_ns(first_item_index, cfg.sample_rate_hz),
            sample_rate_hz=cfg.sample_rate_hz,
            data=data_f32,
        )
        return [event]


__all__ = ["IMU_CHANNELS", "IMU_UNITS", "SimulatedImuAdapter", "SimulatedImuConfig"]
