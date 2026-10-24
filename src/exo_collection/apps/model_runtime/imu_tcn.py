"""Strict causal input assembly for the deployed right-leg/pelvis IMU TCN."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable

import numpy as np

from exo_collection.apps.model_runtime.spec import FeatureSpec, ImuTcnSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class ImuWindowResult:
    """A source-grid window and the physical sample times used to build it."""

    features: np.ndarray
    grid_end_ns: int
    right_source_times_ns: np.ndarray
    pelvis_source_times_ns: np.ndarray
    grid_times_ns: np.ndarray | None = None
    raw_values: np.ndarray | None = None


class ImuTcnFeatureExtractor:
    """Build ``[12, N]`` BTC-compatible windows from two independent streams.

    The hub may expose both MTw units through one umbrella ``imu`` stream.  A
    grid point selects only a sample at or before that point, and every source
    sample must be no more than 50 ms old.  No interpolation, zero padding,
    duplicated first frame, normalization, or future sample is introduced.
    """

    def __init__(
        self, config: ImuTcnSpec, *, clock_ns: Callable[[], int] | None = None
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns or perf_counter_ns
        self.last_reason: str | None = None
        self.last_grid_end_ns: int | None = None
        self.last_result: ImuWindowResult | None = None
        self.last_diagnostics: dict[str, object] = {}
        self.waiting_for_valid_window = False
        self._resample_issue = None

    @property
    def needs_rewarm(self) -> bool:
        return self.last_reason is not None and self.last_reason != "warmup"

    def reset(self) -> None:
        self.last_reason = None
        self.last_grid_end_ns = None
        self.last_result = None
        self.waiting_for_valid_window = False

    def preflight(self, hub: SubscriptionHub, spec: FeatureSpec) -> dict[str, object]:
        errors: list[str] = []
        if spec.control_rate_hz <= 0:
            errors.append("control_rate_hz must be positive")
        window_samples = self._window_samples(spec)
        if window_samples is None:
            errors.append(
                "IMU TCN requires an integer positive window_s * control_rate_hz"
            )
        if self.config.right_device_id is None or self.config.pelvis_device_id is None:
            errors.append("right_device_id and pelvis_device_id are required")
        try:
            descriptor = hub.descriptor(self.config.modality)
        except KeyError:
            descriptor = {}
            errors.append(f"missing modality {self.config.modality!r}")
        metadata = descriptor.get("metadata", {}) or {}
        source_rate_hz = self._declared_source_rate_hz(hub, spec, descriptor, metadata)
        if source_rate_hz is not None and source_rate_hz + 1e-6 < spec.control_rate_hz:
            errors.append(
                f"IMU source rate {source_rate_hz:g} Hz is below model control rate "
                f"{spec.control_rate_hz:g} Hz"
            )
        ids = {str(value) for value in (descriptor.get("device_ids", ()) or metadata.get("device_ids", ()))}
        for name, value in (
            ("right_device_id", self.config.right_device_id),
            ("pelvis_device_id", self.config.pelvis_device_id),
        ):
            if value and ids and str(value) not in ids:
                errors.append(f"{name} {value!r} is not in descriptor device_ids")
        channels = tuple(descriptor.get("channels", ()) or metadata.get("channels", ()) or ())
        units = tuple(descriptor.get("units", ()) or metadata.get("units", ()) or ())
        shape = tuple(descriptor.get("sample_shape", ()) or metadata.get("sample_shape", ()) or ())
        if not channels or len(channels) < 6:
            errors.append("IMU descriptor must expose raw acc/gyr channel names")
        if len(shape) != 2 or shape[1] < 6:
            errors.append("IMU descriptor sample_shape must be (device_count, >=6)")
        expected_channels = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
        if channels and tuple(str(value) for value in channels[:6]) != expected_channels:
            errors.append("IMU channels must start with acc_x/y/z,gyr_x/y/z in raw order")
        if len(units) < 6:
            errors.append("IMU descriptor must expose SI units for acc/gyr channels")
        else:
            allowed_acc = {"m/s2", "m/s^2", "m/s²"}
            if any(str(value) not in allowed_acc for value in units[:3]):
                errors.append("accelerometer units must be m/s2")
            if any(str(value) != "rad/s" for value in units[3:6]):
                errors.append("gyroscope units must be rad/s")
        latest_right = latest_pelvis = None
        if self.config.right_device_id:
            latest_right = hub.latest_for_device(self.config.modality, self.config.right_device_id)
        if self.config.pelvis_device_id:
            latest_pelvis = hub.latest_for_device(self.config.modality, self.config.pelvis_device_id)
        ages: dict[str, float | None] = {"right_age_ms": None, "pelvis_age_ms": None}
        if latest_right is None or latest_pelvis is None:
            errors.append("waiting for both selected IMU streams")
        else:
            now_ns = int(self._clock_ns())
            ages["right_age_ms"] = max(0.0, (now_ns - latest_right[0]) / 1_000_000.0)
            ages["pelvis_age_ms"] = max(0.0, (now_ns - latest_pelvis[0]) / 1_000_000.0)
            max_age = int(self.config.max_sample_age_s * _NS_PER_S)
            if now_ns - latest_right[0] > max_age or now_ns - latest_pelvis[0] > max_age:
                errors.append("selected IMU streams are older than max_sample_age_s")
            if not errors:
                # Probe a fresh grid endpoint without consuming it.  This
                # Verify all causal frames before arming is offered.
                saved = (self.last_reason, self.last_grid_end_ns, self.last_result,
                         self.waiting_for_valid_window)
                self.last_grid_end_ns = None
                probe = self.extract(hub, spec)
                probe_reason = self.last_reason
                (self.last_reason, self.last_grid_end_ns, self.last_result,
                 self.waiting_for_valid_window) = saved
                if probe.size == 0 and probe_reason not in {None, "no new source grid"}:
                    errors.append(str(probe_reason))
        return {"ready": not errors, "errors": errors, "device_ids": sorted(ids), **ages}

    def extract(self, hub: SubscriptionHub, spec: FeatureSpec) -> np.ndarray:
        self.last_reason = None
        self.waiting_for_valid_window = False
        count = self._window_samples(spec)
        if count is None:
            return self._empty(
                "IMU TCN requires an integer positive window_s * control_rate_hz"
            )
        right_id = self.config.right_device_id
        pelvis_id = self.config.pelvis_device_id
        if not right_id or not pelvis_id:
            return self._empty("device IDs are not configured")
        right = hub.window_for_device(self.config.modality, right_id, max(3.0, spec.window_s + 0.2))
        pelvis = hub.window_for_device(self.config.modality, pelvis_id, max(3.0, spec.window_s + 0.2))
        if right[0].size == 0 or pelvis[0].size == 0:
            return self._empty("warmup: waiting for both IMUs", warmup=True)
        source_rate_hz = self._declared_source_rate_hz(
            hub, spec, hub.descriptor(self.config.modality),
            hub.descriptor(self.config.modality).get("metadata", {}) or {},
        )
        if source_rate_hz is not None and source_rate_hz + 1e-6 < spec.control_rate_hz:
            return self._empty(
                f"IMU source rate {source_rate_hz:g} Hz is below model control rate "
                f"{spec.control_rate_hz:g} Hz"
            )
        now_ns = int(self._clock_ns())
        max_age_ns = int(self.config.max_sample_age_s * _NS_PER_S)
        self.last_diagnostics = {
            "checked_at_ns": now_ns, "max_sample_age_ms": max_age_ns / 1e6,
            "declared_source_rate_hz": source_rate_hz,
            "right_device_id": right_id, "pelvis_device_id": pelvis_id,
            "right_latest_ns": int(right[0][-1]), "pelvis_latest_ns": int(pelvis[0][-1]),
            "right_age_ms": (now_ns - int(right[0][-1])) / 1e6,
            "pelvis_age_ms": (now_ns - int(pelvis[0][-1])) / 1e6,
        }
        if now_ns - int(right[0][-1]) > max_age_ns or now_ns - int(pelvis[0][-1]) > max_age_ns:
            return self._empty("selected IMU stream is stale", recoverable=True)
        dt_ns = int(round(_NS_PER_S / spec.control_rate_hz))
        # Independent sources advance one common watermark. Wall-clock ticks
        # cannot fabricate a new source grid while either device is unchanged.
        watermark = min(int(right[0][-1]), int(pelvis[0][-1]), now_ns)
        grid_end = watermark - (watermark % dt_ns)
        if self.last_grid_end_ns is not None and grid_end <= self.last_grid_end_ns:
            return self._empty("no new source grid")
        grid = grid_end - np.arange(count - 1, -1, -1, dtype=np.int64) * dt_ns
        right_values, right_times = self._causal_resample(right[0], right[1], grid, max_age_ns)
        if right_values is None:
            return self._empty("warmup: missing frame, non-finite value, or source age",
                               recoverable=self._resample_issue in {"history", "age"})
        pelvis_values, pelvis_times = self._causal_resample(pelvis[0], pelvis[1], grid, max_age_ns)
        if pelvis_values is None:
            return self._empty("warmup: missing frame, non-finite value, or source age",
                               recoverable=self._resample_issue in {"history", "age"})
        # The MTw descriptor can expose additional magnetometer/orientation
        # fields after the six raw acc/gyr fields.  The trained contract uses
        # only those first six fields from each physical unit.
        features = np.concatenate((right_values[:, :6], pelvis_values[:, :6]), axis=1).T.astype(np.float32, copy=False)
        if features.shape != (12, count) or not np.isfinite(features).all():
            return self._empty("invalid IMU feature tensor")
        self.last_grid_end_ns = grid_end
        # Keep canonical causal SI samples for the independent gate cursor.
        # The TCN tensor remains its own model-normalized 94-point window.
        raw_values = np.concatenate((right_values[:, :6], pelvis_values[:, :6]), axis=1)
        self.last_result = ImuWindowResult(features, grid_end, right_times, pelvis_times,
                                           grid.copy(), raw_values.copy())
        return features

    def _causal_resample(
        self,
        times: np.ndarray,
        values: np.ndarray,
        grid: np.ndarray,
        max_age_ns: int,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        times = np.asarray(times, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        self._resample_issue = None
        if times.size == 0 or values.ndim != 2:
            self._resample_issue = "shape"
            return None, None
        if np.any(np.diff(times) < 0):
            self._resample_issue = "order"
            return None, None
        indices = np.searchsorted(times, grid, side="right") - 1
        if np.any(indices < 0):
            self._resample_issue = "history"
            return None, None
        source_times = times[indices]
        ages = grid - source_times
        if np.any(ages < 0) or np.any(ages > max_age_ns):
            self._resample_issue = "age"
            return None, None
        selected = values[indices]
        if selected.shape[1] < 6 or not np.isfinite(selected[:, :6]).all():
            self._resample_issue = "non_finite"
            return None, None
        return selected, source_times

    def _declared_source_rate_hz(
        self,
        hub: SubscriptionHub,
        spec: FeatureSpec,
        descriptor: dict[str, object],
        metadata: dict[str, object],
    ) -> float | None:
        """Resolve the one declared raw rate used for the selected IMU input.

        A positive descriptor or registered stream rate is the physical source
        fact.  ``FeatureInput.rate_hz`` is used only when no physical rate is
        available, as an expected-rate hint.  A missing or non-positive rate
        remains unknown and is left to the normal causal age checks.
        """
        for source in (descriptor, metadata):
            for key in (
                "nominal_rate_hz",
                "nominal_sample_rate_hz",
                "sample_rate_hz",
                "rate_hz",
            ):
                value = source.get(key)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0:
                    return value
        try:
            value = float(hub.rate_hz(self.config.modality))
        except (KeyError, TypeError, ValueError):
            value = 0.0
        if np.isfinite(value) and value > 0:
            return value
        input_rate = next(
            (
                item.rate_hz
                for item in spec.inputs
                if item.modality == self.config.modality and item.rate_hz is not None
            ),
            None,
        )
        if input_rate is None:
            return None
        value = float(input_rate)
        return value if np.isfinite(value) and value > 0 else None

    @staticmethod
    def _window_samples(spec: FeatureSpec) -> int | None:
        samples = float(spec.window_s) * float(spec.control_rate_hz)
        rounded = round(samples) if np.isfinite(samples) else 0
        if (
            not np.isfinite(samples)
            or rounded < 1
            or not np.isclose(samples, rounded, rtol=0.0, atol=1e-6)
        ):
            return None
        return int(rounded)

    def _empty(self, reason: str, *, warmup: bool = False, recoverable: bool = False) -> np.ndarray:
        self.last_reason = "warmup" if warmup else reason
        self.waiting_for_valid_window = warmup or recoverable
        return np.empty((0, 0), dtype=np.float32)


__all__ = ["ImuTcnFeatureExtractor", "ImuWindowResult"]
