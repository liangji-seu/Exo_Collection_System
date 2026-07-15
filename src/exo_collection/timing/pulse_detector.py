"""Streaming hysteresis detector for analog synchronization pulses."""

from __future__ import annotations

from dataclasses import dataclass
from time import time_ns
from typing import Sequence
from uuid import UUID

import numpy as np

from exo_collection.domain.events import EdgeType, SyncPulseEvent


DETECTOR_VERSION = "hysteresis-1.0.0"


@dataclass(frozen=True, slots=True)
class PulseDetectorConfig:
    """Detection thresholds and temporal rejection rules.

    A pulse starts above ``high_threshold`` and can only end below
    ``low_threshold``.  It must remain within the high hysteresis state for at
    least ``min_pulse_width_ns``.  A prospective falling edge must remain low
    for ``debounce_ns`` before it is accepted.
    """

    high_threshold: float = 2.5
    low_threshold: float = 1.0
    min_pulse_width_ns: int = 1_000_000
    debounce_ns: int = 500_000
    detector_version: str = DETECTOR_VERSION

    def __post_init__(self) -> None:
        if not np.isfinite(self.high_threshold) or not np.isfinite(self.low_threshold):
            raise ValueError("pulse thresholds must be finite")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be less than high_threshold")
        if self.min_pulse_width_ns < 0:
            raise ValueError("min_pulse_width_ns must be non-negative")
        if self.debounce_ns < 0:
            raise ValueError("debounce_ns must be non-negative")
        if not self.detector_version.strip():
            raise ValueError("detector_version must not be empty")


class PulseDetector:
    """Incremental pulse detector that preserves state across input chunks."""

    def __init__(
        self,
        config: PulseDetectorConfig | None = None,
        *,
        source_device: str = "sync_pulse_sim",
        clock_domain: str = "sync_pulse_device_clock",
        session_uuid: UUID | str | None = None,
        trial_uuid: UUID | str | None = None,
    ) -> None:
        self.config = config or PulseDetectorConfig()
        if not source_device.strip() or not clock_domain.strip():
            raise ValueError("source_device and clock_domain must not be empty")
        self.source_device = source_device
        self.clock_domain = clock_domain
        self.session_uuid = session_uuid
        self.trial_uuid = trial_uuid
        self.reset()

    def reset(self) -> None:
        self._state = "low"
        self._candidate_start_index: int | None = None
        self._candidate_start_ns: int | None = None
        self._candidate_amplitude: float | None = None
        self._active_start_index: int | None = None
        self._active_start_ns: int | None = None
        self._active_pulse_id: str | None = None
        self._peak_amplitude = -np.inf
        self._fall_index: int | None = None
        self._fall_ns: int | None = None
        self._fall_amplitude: float | None = None
        self._pulse_counter = 0
        self._last_processed_index: int | None = None
        self._last_processed_ns: int | None = None

    @property
    def pulse_count(self) -> int:
        return self._pulse_counter

    @property
    def active(self) -> bool:
        return self._state in ("high", "fall_candidate")

    def process(
        self,
        samples: Sequence[float] | np.ndarray,
        *,
        first_sample_index: int,
        host_monotonic_ns: int | Sequence[int] | np.ndarray,
        sample_rate_hz: float | None = None,
    ) -> list[SyncPulseEvent]:
        """Process one contiguous chunk and return newly confirmed edges.

        ``host_monotonic_ns`` may be a timestamp for the first sample or one
        timestamp per sample.  A scalar requires ``sample_rate_hz`` so that the
        remaining timestamps can be reconstructed.
        """

        if first_sample_index < 0:
            raise ValueError("first_sample_index must be non-negative")
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("sync pulse samples must be one-dimensional")
        if values.size == 0:
            return []
        if not np.all(np.isfinite(values)):
            raise ValueError("sync pulse samples must be finite")

        times = self._timestamps(
            count=values.size,
            host_monotonic_ns=host_monotonic_ns,
            sample_rate_hz=sample_rate_hz,
        )
        if self._last_processed_index is not None and first_sample_index <= self._last_processed_index:
            raise ValueError("sample indices must increase across chunks")
        if self._last_processed_ns is not None and int(times[0]) <= self._last_processed_ns:
            raise ValueError("host monotonic timestamps must increase across chunks")

        events: list[SyncPulseEvent] = []
        for offset, (raw_value, raw_time) in enumerate(zip(values, times, strict=True)):
            index = first_sample_index + offset
            timestamp = int(raw_time)
            value = float(raw_value)
            events.extend(self._consume(value=value, index=index, timestamp_ns=timestamp))

        self._last_processed_index = first_sample_index + values.size - 1
        self._last_processed_ns = int(times[-1])
        return events

    def _consume(self, *, value: float, index: int, timestamp_ns: int) -> list[SyncPulseEvent]:
        cfg = self.config
        result: list[SyncPulseEvent] = []

        if self._state == "low":
            if value >= cfg.high_threshold:
                self._candidate_start_index = index
                self._candidate_start_ns = timestamp_ns
                self._candidate_amplitude = value
                self._peak_amplitude = value
                self._state = "rise_candidate"
                if cfg.min_pulse_width_ns == 0:
                    result.append(self._confirm_rising())
            return result

        if self._state == "rise_candidate":
            self._peak_amplitude = max(self._peak_amplitude, value)
            if value <= cfg.low_threshold:
                width_ns = timestamp_ns - int(self._candidate_start_ns)
                if width_ns >= cfg.min_pulse_width_ns:
                    result.append(self._confirm_rising())
                    self._fall_index = index
                    self._fall_ns = timestamp_ns
                    self._fall_amplitude = value
                    self._state = "fall_candidate"
                    if cfg.debounce_ns == 0:
                        result.append(self._confirm_falling())
                else:
                    self._clear_candidate()
                    self._state = "low"
            elif timestamp_ns - int(self._candidate_start_ns) >= cfg.min_pulse_width_ns:
                result.append(self._confirm_rising())
            return result

        if self._state == "high":
            self._peak_amplitude = max(self._peak_amplitude, value)
            if value <= cfg.low_threshold:
                self._fall_index = index
                self._fall_ns = timestamp_ns
                self._fall_amplitude = value
                self._state = "fall_candidate"
                if cfg.debounce_ns == 0:
                    result.append(self._confirm_falling())
            return result

        if self._state == "fall_candidate":
            self._peak_amplitude = max(self._peak_amplitude, value)
            if value >= cfg.high_threshold:
                # A short low excursion is noise; remain inside the same pulse.
                self._fall_index = None
                self._fall_ns = None
                self._fall_amplitude = None
                self._state = "high"
            elif timestamp_ns - int(self._fall_ns) >= cfg.debounce_ns:
                result.append(self._confirm_falling())
            return result

        raise RuntimeError(f"unknown detector state {self._state!r}")

    def _confirm_rising(self) -> SyncPulseEvent:
        assert self._candidate_start_index is not None
        assert self._candidate_start_ns is not None
        assert self._candidate_amplitude is not None
        self._pulse_counter += 1
        self._active_pulse_id = f"{self.source_device}:{self._pulse_counter:06d}"
        self._active_start_index = self._candidate_start_index
        self._active_start_ns = self._candidate_start_ns
        event = self._event(
            pulse_id=self._active_pulse_id,
            edge_type=EdgeType.RISING,
            sample_index=self._active_start_index,
            host_monotonic_ns=self._active_start_ns,
            amplitude=self._candidate_amplitude,
            pulse_width_ns=None,
            threshold=self.config.high_threshold,
            confidence=self._confidence(
                amplitude=self._peak_amplitude,
                width_ns=self.config.min_pulse_width_ns,
            ),
        )
        peak_amplitude = self._peak_amplitude
        self._clear_candidate()
        self._peak_amplitude = peak_amplitude
        self._state = "high"
        return event

    def _confirm_falling(self) -> SyncPulseEvent:
        assert self._active_pulse_id is not None
        assert self._active_start_ns is not None
        assert self._fall_index is not None
        assert self._fall_ns is not None
        assert self._fall_amplitude is not None
        pulse_width_ns = max(0, self._fall_ns - self._active_start_ns)
        event = self._event(
            pulse_id=self._active_pulse_id,
            edge_type=EdgeType.FALLING,
            sample_index=self._fall_index,
            host_monotonic_ns=self._fall_ns,
            amplitude=float(self._peak_amplitude),
            pulse_width_ns=pulse_width_ns,
            threshold=self.config.low_threshold,
            confidence=self._confidence(
                amplitude=self._peak_amplitude,
                width_ns=pulse_width_ns,
            ),
        )
        self._active_start_index = None
        self._active_start_ns = None
        self._active_pulse_id = None
        self._fall_index = None
        self._fall_ns = None
        self._fall_amplitude = None
        self._peak_amplitude = -np.inf
        self._state = "low"
        return event

    def _event(
        self,
        *,
        pulse_id: str,
        edge_type: EdgeType,
        sample_index: int,
        host_monotonic_ns: int,
        amplitude: float,
        pulse_width_ns: int | None,
        threshold: float,
        confidence: float,
    ) -> SyncPulseEvent:
        return SyncPulseEvent(
            session_uuid=self.session_uuid,
            trial_uuid=self.trial_uuid,
            device_id=self.source_device,
            modality="sync_pulse",
            clock_domain=self.clock_domain,
            host_monotonic_ns=host_monotonic_ns,
            host_utc_ns=time_ns(),
            pulse_id=pulse_id,
            source_device=self.source_device,
            edge_type=edge_type,
            sample_index=sample_index,
            amplitude=amplitude,
            pulse_width_ns=pulse_width_ns,
            detection_threshold=threshold,
            confidence=confidence,
            detector_version=self.config.detector_version,
        )

    def _confidence(self, *, amplitude: float, width_ns: int) -> float:
        span = max(abs(self.config.high_threshold - self.config.low_threshold), 1e-12)
        amplitude_score = np.clip((amplitude - self.config.low_threshold) / span, 0.0, 2.0) / 2.0
        if self.config.min_pulse_width_ns == 0:
            width_score = 1.0
        else:
            width_score = min(1.0, width_ns / self.config.min_pulse_width_ns)
        return float(np.clip(0.5 * amplitude_score + 0.5 * width_score, 0.0, 1.0))

    def _clear_candidate(self) -> None:
        self._candidate_start_index = None
        self._candidate_start_ns = None
        self._candidate_amplitude = None
        if self._state != "high":
            self._peak_amplitude = -np.inf

    @staticmethod
    def _timestamps(
        *,
        count: int,
        host_monotonic_ns: int | Sequence[int] | np.ndarray,
        sample_rate_hz: float | None,
    ) -> np.ndarray:
        if np.isscalar(host_monotonic_ns):
            if sample_rate_hz is None or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
                raise ValueError("positive sample_rate_hz is required for a scalar timestamp")
            period_ns = 1_000_000_000 / float(sample_rate_hz)
            times = int(host_monotonic_ns) + np.rint(np.arange(count) * period_ns).astype(np.int64)
        else:
            times = np.asarray(host_monotonic_ns, dtype=np.int64)
            if times.shape != (count,):
                raise ValueError("timestamp array length must match sample count")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("host monotonic timestamps must be strictly increasing")
        return times


__all__ = ["DETECTOR_VERSION", "PulseDetector", "PulseDetectorConfig"]
