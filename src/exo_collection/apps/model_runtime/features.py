"""Feature assembly: turn live subscriptions into a ``[C, T]`` model input.

The default extractor is intentionally simple and well-defined: it holds the
most recent sample per control-grid instant (zero-order hold), flattens each
declared modality to its scalar channels, concatenates them in ``inputs`` order,
and applies optional per-channel mean/std normalization.  Users whose training
pipeline filters, differentiates, or renames channels subclass
``FeatureExtractor`` instead of changing this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from exo_collection.apps.model_runtime.spec import FeatureInput, FeatureSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub

_NS_PER_S = 1_000_000_000


class FeatureExtractionError(RuntimeError):
    """Raised when live features cannot be assembled (missing stream/channel)."""


def hold_resample(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Zero-order-hold ``values`` (rows over ascending ``times``) onto ``grid``.

    Returns ``values``-shaped ``[len(grid), k]``.  Grid instants before the first
    sample hold the first sample; instants after the last sample hold the last.
    """
    times = np.asarray(times, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    grid = np.asarray(grid, dtype=np.int64)
    if times.size == 0:
        return np.zeros((grid.size, values.shape[1]), dtype=np.float32)
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, times.size - 1)
    return values[indices]


class FeatureExtractor(ABC):
    """Build a ``[C, T]`` float32 feature window from the subscription hub."""

    @abstractmethod
    def extract(self, hub: SubscriptionHub, spec: FeatureSpec) -> np.ndarray:
        """Return ``[C, T]``, or a zero-length array when data is insufficient."""


class DefaultWindowFeatureExtractor(FeatureExtractor):
    """Flat-channel, zero-order-hold feature assembly (see module docstring)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_grid_end_ns: int | None = None
        self.last_reason: str | None = None

    def preflight(self, hub: SubscriptionHub, spec: FeatureSpec) -> dict[str, object]:
        saved = (self.last_grid_end_ns, self.last_reason)
        try:
            self.last_grid_end_ns = None
            tensor = self.extract(hub, spec)
            errors = [] if tensor.size else [self.last_reason or "等待完整输入窗口"]
        except (FeatureExtractionError, KeyError, ValueError) as exc:
            errors = [str(exc)]
        finally:
            self.last_grid_end_ns, self.last_reason = saved
        return {"ready": not errors, "errors": errors}

    def _empty(self, reason: str) -> np.ndarray:
        self.last_reason = reason
        return np.empty((0, 0), dtype=np.float32)

    def extract(self, hub: SubscriptionHub, spec: FeatureSpec) -> np.ndarray:
        self.last_reason = None
        window_samples = max(1, int(round(spec.window_s * spec.control_rate_hz)))
        dt_ns = int(round(_NS_PER_S / spec.control_rate_hz))

        latest_ts: list[int] = []
        for entry in spec.inputs:
            # Resolve channels before warmup, so a misspelled channel cannot
            # silently look like a merely incomplete window.
            self._channel_columns(hub, entry)
            latest = hub.latest(entry.modality)
            if latest is None:
                return self._empty(f"等待模态 {entry.modality} 输入")
            latest_ts.append(latest[0])
        # The slowest stream is the watermark. A fast modality cannot create
        # repeated predictions by extending an unchanged slow stream forever.
        now_ns = min(latest_ts)
        now_ns -= now_ns % dt_ns
        if self.last_grid_end_ns is not None and now_ns <= self.last_grid_end_ns:
            return self._empty("no new source grid")
        grid = np.asarray(
            [now_ns - int(round((window_samples - 1 - i) * dt_ns)) for i in range(window_samples)],
            dtype=np.int64,
        )

        blocks: list[np.ndarray] = []
        for entry in spec.inputs:
            times, values = hub.window_ending(entry.modality, int(grid[0]), int(grid[-1]))
            if values.shape[0] == 0:
                return self._empty(f"等待模态 {entry.modality} 输入")
            if np.any(np.diff(times) < 0):
                raise FeatureExtractionError(f"modality {entry.modality!r} timestamps are not ordered")
            # Holding a real preceding sample is causal; fabricating the
            # beginning of a window by repeating its first future sample is not.
            if int(times[0]) > int(grid[0]):
                return self._empty(f"等待模态 {entry.modality} 完整窗口 ({window_samples} 帧)")
            columns = self._channel_columns(hub, entry)
            if values.ndim != 2 or values.shape[1] != len(hub.flat_channels(entry.modality)):
                raise FeatureExtractionError(f"modality {entry.modality!r} sample shape does not match channels")
            selected = values[:, columns]
            resampled = hold_resample(times, selected, grid)  # [T, k]
            if not np.isfinite(resampled).all():
                raise FeatureExtractionError(f"modality {entry.modality!r} contains non-finite features")
            blocks.append(resampled)

        tensor = np.concatenate(blocks, axis=1).T.astype(np.float32, copy=False)  # [C, T]
        tensor = self._normalize(tensor, spec)
        if not np.isfinite(tensor).all():
            raise FeatureExtractionError("normalization produced non-finite features")
        self.last_grid_end_ns = now_ns
        return tensor

    def expected_channel_count(self, hub: SubscriptionHub, spec: FeatureSpec) -> int:
        total = 0
        for entry in spec.inputs:
            total += self._channel_columns(hub, entry).size
        return total

    def _channel_columns(self, hub: SubscriptionHub, entry: FeatureInput) -> np.ndarray:
        flat = hub.flat_channels(entry.modality)
        if not flat:
            raise FeatureExtractionError(
                f"modality {entry.modality!r} exposes no channels"
            )
        if entry.channels is None:
            return np.arange(len(flat), dtype=np.int64)
        indices: list[int] = []
        for name in entry.channels:
            try:
                indices.append(flat.index(name))
            except ValueError:
                raise FeatureExtractionError(
                    f"channel {name!r} not found in modality {entry.modality!r}; "
                    f"available flat channels: {list(flat)}"
                ) from None
        return np.asarray(indices, dtype=np.int64)

    def _normalize(self, tensor: np.ndarray, spec: FeatureSpec) -> np.ndarray:
        mean = spec.normalization.mean
        std = spec.normalization.std
        if not mean and not std:
            return tensor
        channels = tensor.shape[0]
        if len(mean) != channels or (std and len(std) != channels):
            raise FeatureExtractionError(
                f"normalization length {len(mean)}/{len(std)} does not match "
                f"{channels} feature channels"
            )
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(channels, 1)
        if not np.isfinite(mean_arr).all():
            raise FeatureExtractionError("normalization mean must be finite")
        centered = tensor - mean_arr
        if std:
            std_arr = np.asarray(std, dtype=np.float32).reshape(channels, 1)
            if not np.isfinite(std_arr).all() or np.any(std_arr <= 0):
                raise FeatureExtractionError("normalization std must be finite and positive")
            centered = centered / std_arr
        return centered.astype(np.float32, copy=False)


__all__ = [
    "DefaultWindowFeatureExtractor",
    "FeatureExtractionError",
    "FeatureExtractor",
    "hold_resample",
]
