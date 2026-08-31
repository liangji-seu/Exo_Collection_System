"""低通滤波：Butterworth + 零相位 filtfilt。

原则：
- 保留 raw / filtered 两份数据，不原地覆盖。
- cutoff 不写死，由 config 控制（marker cutoff 与 GRF cutoff 分开）。
- NaN 安全的滤波：有 NaN 的段单独处理（返回 NaN 保持位置）。
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, filtfilt
except ImportError as exc:  # pragma: no cover
    raise ImportError("滤波需要 scipy：pip install scipy") from exc


def butterworth_lowpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """零相位 4 阶 Butterworth 低通。输入 (n_frames,) 或 (n_frames, n_ch)。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or cutoff_hz is None or fs is None or fs <= 0 or cutoff_hz <= 0:
        return x
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        return x
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    axis = 0
    return filtfilt(b, a, x, axis=axis)


def filter_nan_safe(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """对含 NaN 的序列逐列滤波，NaN 位置保持 NaN（对每个连续有效段滤波）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    out = np.full_like(x, np.nan)
    for col in range(x.shape[1]):
        col_data = x[:, col]
        valid = ~np.isnan(col_data)
        if not valid.any():
            out[:, col] = col_data
            continue
        # 对连续有效段滤波（简化：找连续段，段长度 > 2*order 才滤波）
        idx = np.where(valid)[0]
        # 分连续段
        start = 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or idx[i] != idx[i - 1] + 1:
                seg = idx[start:i]
                if seg.size > 3 * order:
                    out[seg, col] = butterworth_lowpass(col_data[seg], cutoff_hz, fs, order)
                else:
                    out[seg, col] = col_data[seg]
                start = i
    if out.shape[1] == 1:
        out = out[:, 0]
    return out


__all__ = ["butterworth_lowpass", "filter_nan_safe"]
