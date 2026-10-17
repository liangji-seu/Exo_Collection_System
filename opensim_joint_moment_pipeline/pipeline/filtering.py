"""Zero-phase low-pass filtering helpers for marker and force data."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass_zero_phase(
    values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    *,
    order: int = 4,
    preserve_missing: bool = True,
) -> np.ndarray:
    """Filter along axis 0 without adding phase delay.

    Missing samples are linearly interpolated only for the filtering operation.
    They are restored afterwards when ``preserve_missing`` is true.
    """
    data = np.asarray(values, dtype=np.float64)
    if cutoff_hz <= 0 or sample_rate_hz <= 0 or cutoff_hz >= sample_rate_hz / 2:
        return data.copy()
    original_shape = data.shape
    flat = data.reshape(data.shape[0], -1)
    out = flat.copy()
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    x = np.arange(flat.shape[0], dtype=np.float64)
    for column in range(flat.shape[1]):
        series = flat[:, column]
        valid = np.isfinite(series)
        if valid.sum() < 12:
            continue
        filled = np.interp(x, x[valid], series[valid])
        # Short, genuinely finite segments still need a bounded padding length.
        default_padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        filtered = sosfiltfilt(sos, filled, padlen=min(int(default_padlen), len(filled) - 1))
        if preserve_missing:
            filtered[~valid] = np.nan
        out[:, column] = filtered
    return out.reshape(original_shape)


def lowpass_segmented(
    values: np.ndarray,
    valid: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    *,
    order: int = 4,
    min_segment: int = 12,
    time_s: np.ndarray | None = None,
) -> np.ndarray:
    """Zero-phase lowpass each contiguous ``valid`` segment independently.

    ``valid`` describes available measurements, not a force/contact threshold.
    Continuous low-load/zero-load measurements must not be split at an event
    threshold. Missing data and optional true timestamp gaps split the filter.

    ``values`` 可为 (n,) 或 (n, c)；``valid`` 为 (n,) bool。返回同 shape 数组，
    非接触帧为 NaN。
    """
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    squeeze = False
    if data.ndim == 1:
        data = data[:, None]
        squeeze = True
    out = np.full_like(data, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out[:, 0] if squeeze else out

    breaks = np.diff(idx) > 1
    if time_s is not None:
        time = np.asarray(time_s, dtype=np.float64)
        breaks |= np.diff(time[idx]) > 1.5 / float(sample_rate_hz)
    boundaries = np.where(breaks)[0]
    starts = np.concatenate([[0], boundaries + 1])
    ends = np.concatenate([boundaries, [idx.size - 1]])
    for s, e in zip(starts, ends):
        seg = idx[s:e + 1]
        sub = data[seg]
        if seg.size < min_segment or not np.isfinite(sub).any():
            out[seg] = sub
            continue
        out[seg] = lowpass_zero_phase(
            sub, sample_rate_hz, cutoff_hz, order=order, preserve_missing=True
        )
    if squeeze:
        out = out[:, 0]
    return out


__all__ = ["lowpass_zero_phase", "lowpass_segmented"]
