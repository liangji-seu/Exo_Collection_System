"""Raw analog sync-pulse simulator with online hysteresis detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from exo_collection.adapters.base import (
    ModalityDescriptor,
    PreparedInfo,
    QueuedSimulatedAdapter,
    SimulationConfig,
    TrialContext,
)
from exo_collection.domain.events import SampleBatch
from exo_collection.timing.pulse_detector import PulseDetector, PulseDetectorConfig


@dataclass(frozen=True, slots=True)
class SimulatedSyncPulseConfig(SimulationConfig):
    device_id: str = "sync_pulse_sim"
    clock_domain: str = "sync_pulse_sim_clock"
    sample_rate_hz: float = 2000.0
    samples_per_batch: int = 100
    baseline_voltage: float = 0.0
    pulse_voltage: float = 5.0
    noise_std_voltage: float = 0.03
    pulse_interval_s: float = 1.0
    pulse_width_s: float = 0.02
    first_pulse_s: float = 0.25
    high_threshold: float = 2.5
    low_threshold: float = 1.0
    min_pulse_width_ns: int = 1_000_000
    debounce_ns: int = 500_000

    def __post_init__(self) -> None:
        SimulationConfig.__post_init__(self)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if self.sample_rate_hz <= 0 or not np.isfinite(self.sample_rate_hz):
            raise ValueError("sample_rate_hz must be positive and finite")
        if self.samples_per_batch <= 0:
            raise ValueError("samples_per_batch must be positive")
        if self.pulse_interval_s <= 0 or self.pulse_width_s <= 0:
            raise ValueError("pulse interval and width must be positive")
        if self.pulse_width_s >= self.pulse_interval_s:
            raise ValueError("pulse_width_s must be less than pulse_interval_s")
        if self.first_pulse_s < 0:
            raise ValueError("first_pulse_s must be non-negative")
        if self.noise_std_voltage < 0:
            raise ValueError("noise_std_voltage must be non-negative")
        PulseDetectorConfig(
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
            min_pulse_width_ns=self.min_pulse_width_ns,
            debounce_ns=self.debounce_ns,
        )


class SimulatedSyncPulseAdapter(QueuedSimulatedAdapter[SimulatedSyncPulseConfig]):
    config_type = SimulatedSyncPulseConfig

    def __init__(
        self,
        config: SimulatedSyncPulseConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._detector: PulseDetector | None = None
        self._detector_last_host_ns: int | None = None

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
            modality="sync_pulse",
            display_name="Simulated analog synchronization pulse",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch+sync_pulse",
            channels=("voltage",),
            units=("V",),
            nominal_rate_hz=cfg.sample_rate_hz,
            sample_shape=(1,),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": True,
                "pulse_interval_s": cfg.pulse_interval_s,
                "pulse_width_s": cfg.pulse_width_s,
                "high_threshold": cfg.high_threshold,
                "low_threshold": cfg.low_threshold,
                "samples_per_batch": cfg.samples_per_batch,
                "preserves_raw_waveform": True,
            },
        )

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        info = super().prepare(trial)
        cfg = self._config
        self._detector = PulseDetector(
            PulseDetectorConfig(
                high_threshold=cfg.high_threshold,
                low_threshold=cfg.low_threshold,
                min_pulse_width_ns=cfg.min_pulse_width_ns,
                debounce_ns=cfg.debounce_ns,
            ),
            source_device=cfg.device_id,
            clock_domain=cfg.clock_domain,
            session_uuid=trial.session_uuid,
            trial_uuid=trial.trial_uuid,
        )
        self._detector_last_host_ns = None
        return info

    def _make_events(
        self,
        *,
        sequence: int,
        first_item_index: int,
        host_monotonic_ns: int,
    ) -> list[Any]:
        cfg = self._config
        indices = first_item_index + np.arange(cfg.samples_per_batch, dtype=np.float64)
        t = indices / cfg.sample_rate_hz
        after_start = t >= cfg.first_pulse_s
        phase = np.mod(np.maximum(0.0, t - cfg.first_pulse_s), cfg.pulse_interval_s)
        active = after_start & (phase < cfg.pulse_width_s)
        voltage = np.where(active, cfg.pulse_voltage, cfg.baseline_voltage)
        if cfg.noise_std_voltage:
            voltage = voltage + self._rng_values.normal(
                0.0, cfg.noise_std_voltage, cfg.samples_per_batch
            )
        data = np.ascontiguousarray(voltage.astype(np.float32).reshape(-1, 1))

        sample_period_ns = max(1, round(1_000_000_000 / cfg.sample_rate_hz))
        if self._detector_last_host_ns is not None:
            host_monotonic_ns = max(
                host_monotonic_ns, self._detector_last_host_ns + sample_period_ns
            )
        self._detector_last_host_ns = (
            host_monotonic_ns + (cfg.samples_per_batch - 1) * sample_period_ns
        )
        batch = SampleBatch(
            **self._event_common(host_monotonic_ns),
            first_sample_index=first_item_index,
            sample_count=cfg.samples_per_batch,
            sequence_number=sequence,
            device_timestamp=self.device_time_ns(first_item_index, cfg.sample_rate_hz),
            sample_rate_hz=cfg.sample_rate_hz,
            data=data,
        )
        assert self._detector is not None
        pulse_events = self._detector.process(
            data[:, 0],
            first_sample_index=first_item_index,
            host_monotonic_ns=host_monotonic_ns,
            sample_rate_hz=cfg.sample_rate_hz,
        )
        return [batch, *pulse_events]


__all__ = ["SimulatedSyncPulseAdapter", "SimulatedSyncPulseConfig"]
