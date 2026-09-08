"""Winter 样条力矩 baseline：右腿 IMU pitch → 步态相位 → 髋力矩基准曲线。

把 Winter《Gait Biomech》"Hip Mean" 髋关节力矩随步态相位（0–100%）的三次样条
查找表（``hip_torque_lut.csv``）当作「参考力矩」；用右腿 IMU 的 ``pitch``（矢状面
倾角）估计步态相位，再把相位映射成一条随时间变化的 baseline 力矩，用于和
OpenSim 反解出的实际髋力矩（``hip_flexion_r``）叠加对比。

约定（已与使用者确认）：

- 相位来源 = 右腿 IMU ``pitch``（矢状面倾角）。
- 0% 相位 = 右腿脚跟触地（heel strike）。对大腿佩戴的 IMU，脚跟触地时髋屈曲最大，
  ``pitch`` 局部极大，因此以 ``pitch`` 极大值作为每个步态周期的起点。
- 相位在相邻两个脚跟触地之间线性插值 0→100%；跨度过大的间隔（明显停顿）判为
  无步态、相位置 NaN，不产生 baseline。

纯 NumPy + 标准库实现，不依赖 scipy / Qt，便于离线单测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型提示，避免运行时导入重量级模块
    from .local_tools import SignalPlayback, TrialPlayback

PITCH_CHANNEL = "pitch"
LUT_FILENAME = "hip_torque_lut.csv"
_DEFAULT_LUT_PATH = Path(__file__).resolve().parent / "data" / LUT_FILENAME


def load_hip_torque_lut(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 ``hip_torque_lut.csv``，返回 ``(phase_pct, torque_nm)``（按相位升序）。

    CSV 无表头，两列分别为步态相位百分比（0–100）与参考髋力矩（N·m），由
    ``cal.py`` 对 Winter 数据做周期性三次样条后导出。
    """
    source = Path(path) if path is not None else _DEFAULT_LUT_PATH
    array = np.loadtxt(source, delimiter=",")
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"力矩样条 LUT 形状应为 (n, 2)，实际 {array.shape}")
    phase = np.asarray(array[:, 0], dtype=np.float64)
    torque = np.asarray(array[:, 1], dtype=np.float64)
    order = np.argsort(phase)
    return phase[order], torque[order]


def estimate_gait_phase(
    pitch: np.ndarray,
    sample_rate_hz: float,
    *,
    heel_strike_at_max: bool = True,
    min_sep_s: float = 0.6,
    max_stride_factor: float = 1.8,
    prominence_deg: float | None = None,
) -> np.ndarray:
    """由右腿 IMU ``pitch``（度）估计步态相位，返回 0–100 的相位数组（NaN = 无步态）。

    流程：Hann 窗平滑 → 找 ``pitch`` 极值（默认极大值 = 脚跟触地）→ 按高度做
    非极大抑制（``min_sep_s`` 为相邻触地之间的最小间隔，用于抑制摆动相内的次峰，
    需小于真实步态周期）→ 相邻触地之间线性插值相位。间隔超过中位步距
    ``max_stride_factor`` 倍（明显停顿）的区间置 NaN。
    """
    x = np.asarray(pitch, dtype=np.float64)
    n = x.size
    phase = np.full(n, np.nan, dtype=np.float64)
    fs = float(sample_rate_hz)
    if n < 3 or not np.isfinite(fs) or fs <= 0:
        return phase

    finite = np.isfinite(x)
    if not finite.all() and finite.any():
        index = np.arange(n)
        x = np.interp(index, index[finite], x[finite])
    if not np.isfinite(x).any():
        return phase

    x = _smooth(x, _odd_window(fs, 0.2))
    if prominence_deg is None:
        low, high = np.percentile(x, (10.0, 90.0))
        prominence_deg = max(1.0, 0.15 * float(high - low))

    min_sep = max(2, int(round(min_sep_s * fs)))
    half_width = max(2, int(round(0.1 * fs)))
    peaks = _find_peaks(
        x if heel_strike_at_max else -x,
        half_width=half_width,
        min_sep=min_sep,
        prominence=float(prominence_deg),
    )
    if peaks.size < 2:
        return phase

    intervals = np.diff(peaks).astype(np.float64)
    median_interval = float(np.median(intervals)) if intervals.size else 0.0
    for k in range(peaks.size - 1):
        start, stop = int(peaks[k]), int(peaks[k + 1])
        if intervals[k] > max_stride_factor * median_interval:
            # 明显长于正常步距 → 判为停顿，整段不产生相位。
            continue
        phase[start : stop + 1] = np.linspace(0.0, 100.0, stop - start + 1)
    return phase


def baseline_torque_from_phase(
    phase: np.ndarray,
    lut_phase: np.ndarray,
    lut_torque: np.ndarray,
) -> np.ndarray:
    """把 0–100 的步态相位映射为参考髋力矩（NaN 相位 → NaN 力矩）。"""
    phase = np.asarray(phase, dtype=np.float64)
    torque = np.full(phase.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(phase)
    if valid.any():
        torque[valid] = np.interp(phase[valid], lut_phase, lut_torque)
    return torque


def find_right_leg_imu(playback: "TrialPlayback") -> "SignalPlayback | None":
    """从 playback 里定位右腿 IMU 序列（解耦或单设备），找不到返回 ``None``。"""
    for series in getattr(playback, "imu_sensors", ()) or ():
        if _label_is_right_leg(series.sensor_labels):
            return series
    imu = getattr(playback, "imu", None)
    if imu is not None and _label_is_right_leg(imu.sensor_labels):
        return imu
    return None


def right_leg_pitch(
    playback: "TrialPlayback",
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """返回右腿 IMU 的 ``(time_s, pitch_deg, sample_rate_hz)``，不可用返回 ``None``。"""
    series = find_right_leg_imu(playback)
    if series is None or series.time_s.size < 2:
        return None
    pitch_index = _channel_index(series.channels, PITCH_CHANNEL)
    if pitch_index is None or pitch_index >= series.values.shape[1]:
        return None
    time_s = np.asarray(series.time_s, dtype=np.float64)
    pitch = np.asarray(series.values[:, pitch_index], dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(pitch)
    if int(finite.sum()) < 2:
        return None
    time_s = time_s[finite]
    pitch = pitch[finite]
    sample_rate_hz = 1.0 / max(float(np.median(np.diff(time_s))), 1e-9)
    return time_s, pitch, sample_rate_hz


def moment_imu_offset_s(playback: "TrialPlayback") -> float:
    """力矩 CSV 与 IMU 之间的时钟偏移（秒）。

    ``ground_truth.csv`` 的时间轴相对 C3D 首帧（``c3d_t0_host_monotonic_ns``），而
    回放里的 IMU 时间轴相对 ``formal_t0``（采集触发时刻），两者相差约 0.15 s（约
    10% 步态周期）。该偏移记录在 ``derived/opensim/*/manifest.json`` 的
    ``sync.c3d_t0_host_monotonic_ns`` 中；找不到时回退 0（保持与现有力矩面板一致）。
    """
    formal_t0 = int(getattr(playback, "formal_t0_host_monotonic_ns", 0) or 0)
    manifest_path = getattr(playback, "manifest_path", None)
    if manifest_path is None:
        return 0.0
    trial_root = Path(manifest_path).parent.parent
    c3d_t0 = _find_c3d_t0_host_ns(trial_root)
    if c3d_t0 is None:
        return 0.0
    return (c3d_t0 - formal_t0) / 1e9


def build_hip_baseline(
    playback: "TrialPlayback",
    *,
    lut: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """端到端 baseline：返回 ``(time_s_c3d, torque_nm)``，无右腿 IMU 时返回 ``None``。

    ``time_s_c3d`` 已换算到力矩 CSV 的 C3D 时间轴，可直接 ``np.interp`` 到力矩采样
    点上叠加；``torque_nm`` 为 Winter 参考髋力矩，NaN 表示无步态/停顿区间。
    """
    right = right_leg_pitch(playback)
    if right is None:
        return None
    time_s_formal, pitch, sample_rate_hz = right
    if lut is None:
        lut = load_hip_torque_lut()
    lut_phase, lut_torque = lut
    phase = estimate_gait_phase(pitch, sample_rate_hz)
    torque = baseline_torque_from_phase(phase, lut_phase, lut_torque)
    if not np.isfinite(torque).any():
        return None
    return time_s_formal - moment_imu_offset_s(playback), torque


# -- 内部工具 ---------------------------------------------------------------


def _label_is_right_leg(labels) -> bool:
    for label in labels or ():
        text = str(label).casefold()
        if "right" in text or "右" in text:
            return True
    return False


def _channel_index(channels, target: str) -> int | None:
    for index, name in enumerate(channels):
        if str(name).casefold() == target.casefold():
            return index
    return None


def _odd_window(fs: float, seconds: float) -> int:
    window = max(3, int(round(fs * seconds)))
    return window if window % 2 else window + 1


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return np.asarray(x, dtype=np.float64)
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def _find_peaks(
    x: np.ndarray,
    *,
    half_width: int,
    min_sep: int,
    prominence: float,
) -> np.ndarray:
    """无 scipy 的峰值检测：窗口内局部极值 + 非极大抑制 + 局部突出度过滤。"""
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    half = max(1, int(half_width))
    is_peak = np.zeros(n, dtype=bool)
    for index in range(n):
        low = max(0, index - half)
        high = min(n, index + half + 1)
        if x[index] == x[low:high].max():
            is_peak[index] = True
    peak_index = _collapse_plateaus(np.flatnonzero(is_peak))
    if peak_index.size == 0:
        return peak_index

    order = peak_index[np.argsort(-x[peak_index], kind="stable")]
    kept: list[int] = []
    for peak in order:
        if all(abs(int(peak) - existing) >= min_sep for existing in kept):
            kept.append(int(peak))
    kept = np.sort(np.asarray(kept, dtype=np.int64))

    if prominence > 0 and kept.size:
        proms = np.asarray(
            [
                x[peak] - x[max(0, peak - min_sep) : min(n, peak + min_sep + 1)].min()
                for peak in kept
            ]
        )
        kept = kept[proms >= prominence]
    return kept


def _collapse_plateaus(index: np.ndarray) -> np.ndarray:
    """把连续相等的平台点折叠为其中心点。"""
    if index.size <= 1:
        return index
    groups: list[int] = []
    start = int(index[0])
    previous = start
    for value in index[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
        else:
            groups.append((start + previous) // 2)
            start = previous = current
    groups.append((start + previous) // 2)
    return np.asarray(groups, dtype=np.int64)


def _find_c3d_t0_host_ns(trial_root: Path) -> int | None:
    opensim_dir = Path(trial_root) / "derived" / "opensim"
    if not opensim_dir.is_dir():
        return None
    for run_manifest in sorted(opensim_dir.glob("*/manifest.json")):
        try:
            data = json.loads(run_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sync = data.get("sync") if isinstance(data, dict) else None
        c3d_t0 = sync.get("c3d_t0_host_monotonic_ns") if isinstance(sync, dict) else None
        if isinstance(c3d_t0, (int, float)):
            return int(c3d_t0)
    return None


__all__ = [
    "LUT_FILENAME",
    "PITCH_CHANNEL",
    "baseline_torque_from_phase",
    "build_hip_baseline",
    "estimate_gait_phase",
    "find_right_leg_imu",
    "load_hip_torque_lut",
    "moment_imu_offset_s",
    "right_leg_pitch",
]
