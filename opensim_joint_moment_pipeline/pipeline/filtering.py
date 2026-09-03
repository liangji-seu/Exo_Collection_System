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
        filtered = sosfiltfilt(sos, filled)
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
) -> np.ndarray:
    """Zero-phase lowpass each contiguous ``valid`` segment independently.

    用于接触开关信号（GRF）：只在接触段内滤波，非接触帧置 NaN，避免滤波器的
    边界振铃跨过「接触→摆动」边界制造假力（prompt6 §3.8 第 5 条）。

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
        return values.copy()

    boundaries = np.where(np.diff(idx) > 1)[0]
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
