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


__all__ = ["lowpass_zero_phase"]
