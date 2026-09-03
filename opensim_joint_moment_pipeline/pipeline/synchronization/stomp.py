"""IMU 跺脚冲击 ↔ Gaitway 总垂直力冲击的稳健配对。

受试者在正式采集前会规律跺脚 3～5 次，作为 IMU 与测力台之间的外部同步
动作。本模块：

- 对加速度模长 / 总垂直力做高通包络，突出冲击、抑制缓慢的站姿漂移；
- 支持 3/4/5 次跺脚，不固定数量；
- 不再用「全记录全局幅值阈值」找峰（正式走路的大量足跟冲击会抬高峰值
  中位数，反而把采集开头真正的跺脚排除在候选之外）。改为：枚举所有候选
  爆发段，取**最早**一段「间隔规律 + 与后续活动隔离」的冲击爆发段——即
  采集开头、正式走路之前的同步动作；
- 用近恒定 offset 的单调配对（容忍漏检 / 多检一个峰）；
- 输出每对峰时间、各自差值、中位数、MAD、置信等级，并估计仿射漂移
  ``t_gaitway = a * t_host + b``；短记录无法可靠估计时明确标记 UNASSESSED。

时间方向约定（与 pipeline 一致）：``t_gaitway = t_c3d + offset``，
即 ``offset = t_gaitway - t_host``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

_MIN_PEAK_DISTANCE_S = 0.45      # 峰间最小真实时间间隔（s）；按时间轴判定，兼容 IMU 掉帧
_MAX_INTERVAL_CV = 0.50          # 爆发段相邻间隔的变异系数上界（容忍漏检一个峰）
_PAIR_BAND_S = 0.30              # 配对时允许的 offset 离散半径（s）；真实跺脚峰定位噪声可达 ~0.2s
_HIGH_MAD_S = 0.05               # 高可信：MAD ≤ 50 ms
_MED_MAD_S = 0.20                # 中可信：MAD ≤ 200 ms
_MIN_PAIRS = 3                   # 高可信至少 3 对峰
_MAX_STOMPS = 5                  # 协议最多 5 次跺脚
_SYNC_SEARCH_WINDOW_S = 25.0     # 跺脚在采集开头（站立→跺脚→走），只在前 25s 内找
_BURST_SPAN_S = 3.5              # 一次跺脚爆发段最大时间跨度（5 次 0.8s 跺脚 ≈ 3.2s）
_ACTIVATE_RATIO = 5.0            # 「冲击级」= 幅值 ≥ 5× 包络 P25（去站立噪声，保留弱跺脚）
_IMPACT_RATIO = 15.0             # 爆发段须从 ≥15× 包络 P25 的「硬跺脚」开始（去称重/挪步）
_MIN_BURST_AMP_RATIO = 0.05      # 爆发段内峰须 ≥ 5% 最大峰：剔除比真跺脚小一个量级的噪声伪峰
_NEXT_ACTIVITY_RATIO = 0.25      # 隔离判定忽略弱过渡伪峰：后续活动须 ≥ 0.25× 爆发段中位幅值
_ISOLATION_GAP_S = 2.0           # 爆发段后需 ≥2s 无后续活动，才算「前置同步动作」而非走路
_PRE_QUIET_S = 2.0               # 爆发段前需 ≥2s 无「硬冲击」，排除走路中段的伪爆发段


@dataclass(frozen=True)
class Peak:
    index: int
    time_s: float
    amplitude: float      # 原始包络值（保留单位，用于区分跺脚/走路）
    prominence: float     # 归一化包络上的峰显著度（0~1，用于阈值判定）


@dataclass(frozen=True)
class StompPair:
    imu_index: int
    gaitway_index: int
    imu_time_s: float
    gaitway_time_s: float

    @property
    def offset_s(self) -> float:
        return self.gaitway_time_s - self.imu_time_s


@dataclass(frozen=True)
class StompAlignment:
    pairs: tuple[StompPair, ...]
    offsets_s: np.ndarray
    median_offset_s: float
    mad_s: float
    drift_ppm: float | None      # None = UNASSESSED（短记录 / 样本不足）
    scale_a: float | None
    confidence: str              # HIGH / MEDIUM / LOW
    imu_peak_count: int = 0      # 审计：IMU 检测到的峰数 / 选中爆发段峰数
    gaitway_peak_count: int = 0
    imu_burst_count: int = 0
    gaitway_burst_count: int = 0


@dataclass(frozen=True)
class StompRejection:
    """跺脚配对失败的可审计诊断：两侧峰数、爆发段数与各自拒绝原因。

    无跺脚的 Session 不能伪造成功，必须用这些数值证明「确实没有足够同步
    动作」并转入人工标定（prompt6 §3.1 第 7 条）。
    """

    imu_peak_count: int
    imu_burst_count: int
    imu_reason: str
    gaitway_peak_count: int
    gaitway_burst_count: int
    gaitway_reason: str
    align_reason: str | None = None   # 两侧均选出爆发段、但单调配对失败时非 None


def highpass_envelope(values: np.ndarray, rate_hz: float, cutoff_hz: float = 2.0) -> np.ndarray:
    """对一维信号做高通后取绝对值包络（突出冲击，去除缓慢基线）。"""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 16 or not np.isfinite(v).any():
        return np.zeros_like(v)
    filled = np.where(np.isfinite(v), v, np.nanmedian(v))
    sos = butter(2, float(cutoff_hz), btype="low", fs=float(rate_hz), output="sos")
    baseline = sosfiltfilt(sos, filled)
    return np.abs(filled - baseline)


def _normalize(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(v, [2, 98])
    # 只把下限抬到 0，不裁剪上限：稀疏冲击的包络若用 2-98 分位归一化，冲击峰会
    # 远超 98 分位而被 clip 成 1.0 的平顶，find_peaks 因此丢失真正的峰位
    # （例如把跺脚主峰 69.9 漏掉、误报其 0.3s 前的小幅前震）。保留相对高度才能
    # 让 find_peaks 准确定位主峰；幅值过滤交给 _stomp_burst 的激活阈值。
    return (v - low) / max(high - low, 1e-12)


def _detect_peaks(
    times: np.ndarray, envelope: np.ndarray, *, prominence: float
) -> list[Peak]:
    """在归一化包络上找峰，返回 ``Peak``（携带原始幅值 + 归一化 prominence）。"""
    t = np.asarray(times, dtype=np.float64)
    raw = np.asarray(envelope, dtype=np.float64)
    if t.size < 3:
        return []
    norm = _normalize(raw)
    indices, props = find_peaks(norm, prominence=prominence)
    if indices.size == 0:
        return []

    # 峰间距判定用「真实时间轴」而非样本数：IMU 采样有掉帧（样本间隔 8.3~41.7ms），
    # 若按 find_peaks(distance=int(rate*S)) 把时间窗折算成样本数，会因掉帧而漏掉
    # 时间上 0.49s、但样本上只隔 ~37 个的弱跺脚。这里先取出全部局部峰，再按归一化
    # 幅值降序贪心挑选，保证保留峰之间在真实时间上至少间隔 ``_MIN_PEAK_DISTANCE_S``。
    order = np.argsort(norm[indices])[::-1]
    kept: list[int] = []
    for pos in order:
        idx = int(indices[pos])
        ti = float(t[idx])
        if all(abs(ti - float(t[j])) >= _MIN_PEAK_DISTANCE_S for j in kept):
            kept.append(idx)
    kept.sort()
    prom_map = {int(i): float(p) for i, p in zip(indices, props["prominences"])}
    return [Peak(i, float(t[i]), float(raw[i]), prom_map[i]) for i in kept]


def _stomp_burst(
    peaks: list[Peak],
    envelope: np.ndarray,
    *,
    search_window_s: float | None = None,
) -> tuple[list[Peak] | None, str]:
    """选出「采集开头、正式走路之前」的跺脚爆发段，返回 ``(爆发段, 拒绝原因)``。

    区分跺脚与走路不靠单一幅值阈值（正式走路的足跟冲击幅值与跺脚大量重叠，
    弱跺脚甚至低于强足跟冲击），而是靠**时间结构 + 真实冲击**：

    1. **去站立噪声**：以包络低分位（P25，代表站立/摆腿期的安静水平）为基线，
       只保留幅值 ≥ ``_ACTIVATE_RATIO`` × 基线的「冲击级」峰。走路会抬高包络
       高/中分位，但低分位仍由安静期主导，故该基线对走路不敏感（此前用「峰
       幅值分位」做基线会被走路抬高，误删开头真正的跺脚）。
    2. **选前置爆发段**：在冲击峰里逐段枚举（以首个峰起 ``_BURST_SPAN_S`` 内、
       最多 ``_MAX_STOMPS`` 个），取**最早**一段同时满足——
       - 从「硬跺脚」开始（首峰幅值 ≥ ``_IMPACT_RATIO`` × 基线，排除称重/挪步）；
       - 前置静默（首峰前 ``_PRE_QUIET_S`` 内无「硬冲击」，排除走路中段伪爆发段）；
       - 段内峰幅值同量级（剔除不足 ``_MIN_BURST_AMP_RATIO`` × 最大峰的噪声伪峰）；
       - 相邻间隔规律（变异系数 ≤ ``_MAX_INTERVAL_CV``）；
       - 与后续活动隔离（段后下一个同量级活动峰至少 ``_ISOLATION_GAP_S`` 之外）。

    「硬跺脚 + 前置静默」排除称重/挪步与走路中段，「隔离」排除连续走路，两者
    合起来确保无跺脚的 Session 不会伪造出高可信结果，而是返回 ``None`` 交由
    人工标定。
    """
    if len(peaks) < _MIN_PAIRS:
        return None, f"峰数不足（{len(peaks)} < {_MIN_PAIRS}）"
    subset = [p for p in peaks if search_window_s is None or p.time_s <= search_window_s]
    if len(subset) < _MIN_PAIRS:
        return None, "前置窗口内峰数不足"

    # 取 |包络| 的低分位作为安静基线：真实包络（|高通| 输出）非负，此处 abs 仅
    # 是为了容忍合成测试里可能出现的负噪声；低分位代表站立/摆腿期的安静水平，
    # 不受正式走路足跟冲击（抬高高分位）影响。
    noise_ref = float(np.percentile(np.abs(envelope), 25))
    if noise_ref <= 0:
        return None, "包络基线为 0，无有效冲击"

    impacts = [p for p in subset if p.amplitude >= _ACTIVATE_RATIO * noise_ref]
    if len(impacts) < _MIN_PAIRS:
        return None, "冲击级峰不足"

    n = len(impacts)
    for i in range(n - _MIN_PAIRS + 1):
        first = impacts[i]

        # 爆发段必须从「硬跺脚」开始：称重/挪步的弱冲击不该被当作第一次跺脚
        # （否则会把前置弱峰与真跺脚混成一段，抬高 offset 离散度）。
        if first.amplitude < _IMPACT_RATIO * noise_ref:
            continue

        # 前置静默：正式跺脚前 _PRE_QUIET_S 内不得有「硬冲击」。走路中段的步态
        # 规律且偶有停顿，会伪装成「隔离爆发段」；真正采集开头的跺脚前是安静
        # 站立（或轻微称重），故要求前一段硬冲击至少 _PRE_QUIET_S 之外。
        prev_hard = None
        for k in range(i - 1, -1, -1):
            if impacts[k].amplitude >= _IMPACT_RATIO * noise_ref:
                prev_hard = impacts[k]
                break
        if prev_hard is not None and first.time_s - prev_hard.time_s < _PRE_QUIET_S:
            continue

        burst = [p for p in impacts[i:] if p.time_s - first.time_s <= _BURST_SPAN_S]
        # 相对幅值过滤：噪声伪峰比真跺脚小一个量级以上，若不剔除会挤占
        # ``_MAX_STOMPS`` 名额，把真正的跺脚挤出爆发段（进而破坏隔离判定）。以
        # 爆发段最大峰为参考（硬跺脚必在其中），剔除不足 ``_MIN_BURST_AMP_RATIO``
        # 倍的伪峰；弱跺脚（约 10~15% 最大峰）不受影响。
        burst_ref = max((p.amplitude for p in burst), default=0.0)
        burst = [p for p in burst if p.amplitude >= _MIN_BURST_AMP_RATIO * burst_ref]
        burst = burst[:_MAX_STOMPS]
        if len(burst) < _MIN_PAIRS:
            continue

        intervals = np.diff([p.time_s for p in burst])
        if intervals.min() <= 0:
            continue
        cv = float(np.std(intervals) / np.mean(intervals))
        if cv > _MAX_INTERVAL_CV:
            continue

        # 硬跺脚：爆发段内最大幅值须远超站立基线（排除纯称重/挪步的假爆发段）。
        if max(p.amplitude for p in burst) < _IMPACT_RATIO * noise_ref:
            continue

        # 隔离判定：爆发段后下一个「同量级」活动峰（幅值 ≥ _NEXT_ACTIVITY_RATIO×
        # 爆发段中位幅值）需离最后一个跺脚足够远。跺脚后迈出第一步前往往有一个
        # 幅值远低于跺脚的过渡伪峰（摆腿/轻触板），它不该破坏隔离——真正的走路
        # 足跟冲击才是「后续活动」。
        burst_amp_ref = float(np.median([p.amplitude for p in burst]))
        nxt = None
        # 隔离判定要从爆发段「真实末峰」之后找，不能按 `i + len(burst)` 起跳：相对
        # 幅值过滤会在爆发段内部挖掉噪声伪峰，使 `len(burst)` 小于末峰在 ``impacts``
        # 里的实际位置，导致把末峰自身误判成「后续活动」。
        last_pos = impacts.index(burst[-1])
        for q in impacts[last_pos + 1:]:
            if q.amplitude >= _NEXT_ACTIVITY_RATIO * burst_amp_ref:
                nxt = q
                break
        if nxt is not None and nxt.time_s - burst[-1].time_s < _ISOLATION_GAP_S:
            continue
        return burst, "ok"

    return None, "未找到间隔规律、含硬跺脚且与后续活动隔离的冲击爆发段"


def _align_sequences(
    imu_peaks: list[Peak], gaitway_peaks: list[Peak]
) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    """在近恒定 offset 下做单调 1:1 配对，返回 ``(索引对, offsets)``。

    枚举每一对可能的起始配对作为参考 offset，沿时间轴贪心单调匹配（容忍
    漏检 / 多检一个峰），取对数最多、离散度最小的解。
    """
    imu_t = np.array([p.time_s for p in imu_peaks])
    gait_t = np.array([p.time_s for p in gaitway_peaks])
    best: tuple[int, float, list[tuple[int, int]], np.ndarray] | None = None
    for a in range(len(imu_t)):
        for b in range(len(gait_t)):
            ref = gait_t[b] - imu_t[a]
            pairs: list[tuple[int, int]] = []
            j = b
            for i in range(a, len(imu_t)):
                while j < len(gait_t) and gait_t[j] - imu_t[i] < ref - _PAIR_BAND_S:
                    j += 1
                if j < len(gait_t) and abs(gait_t[j] - imu_t[i] - ref) <= _PAIR_BAND_S:
                    pairs.append((i, j))
                    j += 1
            if len(pairs) < _MIN_PAIRS:
                continue
            offsets = np.array([gait_t[j] - imu_t[i] for i, j in pairs])
            median = float(np.median(offsets))
            mad = float(np.median(np.abs(offsets - median)))
            score = (len(pairs), -mad)
            if best is None or score > (best[0], best[1]):
                best = (len(pairs), -mad, pairs, offsets)
    if best is None:
        return None
    return best[2], best[3]


def _fit_affine(
    imu_t: np.ndarray, gait_t: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """拟合一阶 ``gait = a * imu + b``，返回 ``(a, b, ppm)``；不可靠时返回 None。

    跺脚爆发段本身只有约 3～4 s，而峰定位噪声约 8 ms（一个 IMU 样本），
    这么短的跨度上估计出的 ppm 低于噪声底，没有意义。因此跨度不足 5 s 时
    明确返回 UNASSESSED（``None``），不伪造一个接近 0 的漂移。
    """
    if len(imu_t) < 4:
        return None, None, None
    span = float(np.ptp(imu_t))
    if span < 5.0:
        return None, None, None
    a, b = np.polyfit(imu_t, gait_t, 1)
    ppm = float((a - 1.0) * 1e6)
    return float(a), float(b), ppm


def pair_stomps_diagnosed(
    imu_times: np.ndarray,
    imu_envelope: np.ndarray,
    gaitway_times: np.ndarray,
    gaitway_envelope: np.ndarray,
    *,
    prominence: float = 0.05,
) -> tuple[StompAlignment | None, StompRejection]:
    """检测并配对跺脚，返回 ``(对齐结果, 失败诊断)``。

    失败时对齐为 ``None``，但 ``StompRejection`` 仍保留两侧峰数、爆发段数与
    拒绝原因，供 UI 展示可审计证据并转入人工标定（不伪造成功）。
    """
    imu_peaks = _detect_peaks(imu_times, imu_envelope, prominence=prominence)
    gait_peaks = _detect_peaks(gaitway_times, gaitway_envelope, prominence=prominence)

    imu_sel, imu_reason = _stomp_burst(imu_peaks, imu_envelope, search_window_s=_SYNC_SEARCH_WINDOW_S)
    gait_sel, gait_reason = _stomp_burst(gait_peaks, gaitway_envelope, search_window_s=_SYNC_SEARCH_WINDOW_S)

    def _rejection(align_reason: str | None = None) -> StompRejection:
        return StompRejection(
            imu_peak_count=len(imu_peaks),
            imu_burst_count=len(imu_sel) if imu_sel else 0,
            imu_reason=imu_reason,
            gaitway_peak_count=len(gait_peaks),
            gaitway_burst_count=len(gait_sel) if gait_sel else 0,
            gaitway_reason=gait_reason,
            align_reason=align_reason,
        )

    if imu_sel is None or gait_sel is None:
        return None, _rejection()

    aligned = _align_sequences(imu_sel, gait_sel)
    if aligned is None:
        return None, _rejection(
            "两侧均选出爆发段，但无法找到 ≥3 对近恒定 offset 的单调配对"
        )
    pairs_idx, offsets = aligned
    imu_t = np.array([p.time_s for p in imu_sel])
    gait_t = np.array([p.time_s for p in gait_sel])

    median = float(np.median(offsets))
    mad = float(np.median(np.abs(offsets - median)))
    a, b, ppm = _fit_affine(imu_t[[i for i, _ in pairs_idx]],
                            gait_t[[j for _, j in pairs_idx]])

    n = len(pairs_idx)
    if n >= _MIN_PAIRS and mad <= _HIGH_MAD_S:
        confidence = "HIGH"
    elif n >= _MIN_PAIRS and mad <= _MED_MAD_S:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    pairs = tuple(
        StompPair(
            imu_index=int(i),
            gaitway_index=int(j),
            imu_time_s=float(imu_t[i]),
            gaitway_time_s=float(gait_t[j]),
        )
        for i, j in pairs_idx
    )
    return StompAlignment(
        pairs=pairs,
        offsets_s=offsets,
        median_offset_s=median,
        mad_s=mad,
        drift_ppm=ppm,
        scale_a=a,
        confidence=confidence,
        imu_peak_count=len(imu_peaks),
        gaitway_peak_count=len(gait_peaks),
        imu_burst_count=len(imu_sel),
        gaitway_burst_count=len(gait_sel),
    ), _rejection()


def pair_stomps(
    imu_times: np.ndarray,
    imu_envelope: np.ndarray,
    gaitway_times: np.ndarray,
    gaitway_envelope: np.ndarray,
    *,
    prominence: float = 0.05,
) -> StompAlignment | None:
    """检测并配对跺脚，返回 ``StompAlignment`` 或 ``None``（不足 3 对峰）。

    ``prominence`` 为归一化包络上的峰显著度阈值（与信号单位无关）；幅值打分
    在原始包络上进行，以区分跺脚与走路足跟冲击。需要失败诊断时用
    :func:`pair_stomps_diagnosed`。
    """
    alignment, _ = pair_stomps_diagnosed(
        imu_times, imu_envelope, gaitway_times, gaitway_envelope, prominence=prominence
    )
    return alignment


def detect_impact(
    times: np.ndarray, values: np.ndarray, rate_hz: float, *, prominence: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """便捷入口：返回 ``(原始包络, 峰下标)``，供 UI 叠加显示与选峰。

    峰检测在归一化包络上进行（``prominence`` 为归一化阈值），返回的包络
    保持原始单位以便 UI 直接显示。
    """
    envelope = highpass_envelope(values, rate_hz)
    peaks = _detect_peaks(times, envelope, prominence=prominence)
    return envelope, np.array([p.index for p in peaks], dtype=np.int64)


def diagnose_impacts(
    times: np.ndarray,
    envelope: np.ndarray,
    *,
    prominence: float = 0.05,
    search_window_s: float | None = None,
) -> tuple[list[Peak], list[Peak] | None, str]:
    """审计入口：返回 ``(检测到的全部峰, 选中的爆发段, 拒绝原因)``。

    供 UI 展示候选峰与拒绝原因，以及真实数据验收脚本输出可解释结果；
    ``pair_stomps`` 内部也复用同一套 ``_detect_peaks`` / ``_stomp_burst``。
    """
    peaks = _detect_peaks(times, envelope, prominence=prominence)
    burst, reason = _stomp_burst(peaks, envelope, search_window_s=search_window_s)
    return peaks, burst, reason


__all__ = [
    "StompAlignment",
    "StompPair",
    "StompRejection",
    "detect_impact",
    "diagnose_impacts",
    "highpass_envelope",
    "pair_stomps",
    "pair_stomps_diagnosed",
]
