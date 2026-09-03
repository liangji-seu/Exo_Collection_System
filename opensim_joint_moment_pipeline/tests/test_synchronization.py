"""同步引擎纯算法单元测试（C3D↔H5、时钟、跺脚配对、漂移）。"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from pipeline.synchronization import (
    clock_health,
    highpass_envelope,
    match_c3d_to_h5,
    normalize_marker_name,
    pair_stomps,
    read_host_monotonic_ns,
)


# --------------------------------------------------------------------------
# marker 名称规范化
# --------------------------------------------------------------------------
def test_normalize_marker_name():
    assert normalize_marker_name("003_no_exo_dynamic:R.ASIS") == "R.ASIS"
    assert normalize_marker_name("003_no_exo_dynamic/R.ASIS") == "R.ASIS"
    assert normalize_marker_name("R.ASIS") == "R.ASIS"
    # 虚拟 marker 带点/下划线，规范化不改写内容
    assert normalize_marker_name("003_no_exo_static:V_R.Hip_JC") == "V_R.Hip_JC"


# --------------------------------------------------------------------------
# C3D ↔ mocap.h5 精确匹配
# --------------------------------------------------------------------------
def _make_points(n_frames, names, offset=0, missing=()):
    """生成 (n_frames, n_names, 3) 的 marker 轨迹，含缺失哨兵 9999999。"""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n_frames, len(names), 3)) * 500.0
    for idx in missing:
        pts[:, idx, :] = 9999999.0
    return pts


def test_match_c3d_to_h5_exact_offset():
    names = ["R.ASIS", "L.ASIS", "R.Knee", "L.Knee", "R.Ankle"]
    c3d = _make_points(100, names)
    h5 = np.concatenate([np.zeros((30, len(names), 3)), c3d, np.zeros((20, len(names), 3))])
    labels = [f"subj:{n}" for n in names]
    h5_names = [f"subj/{n}" for n in names]

    result = match_c3d_to_h5(c3d, labels, h5, h5_names)
    assert result.start_frame == 30
    assert result.overlap_frames == 100
    assert result.rms_mm < 1e-3
    assert result.max_error_mm < 1e-3
    assert result.unique


def test_match_c3d_to_h5_excludes_sentinel_medial_markers():
    # 动态导出内嵌静态副本：medial marker 在 C3D 全零、H5 全哨兵，必须排除。
    names = ["R.ASIS", "R.Knee", "R.Knee.Medial"]
    c3d = _make_points(100, names, missing=(2,))
    c3d[:, 2, :] = 0.0  # C3D 静态副本全零
    h5 = np.concatenate([np.zeros((20, len(names), 3)), c3d])
    h5[:, 2, :] = 9999999.0  # H5 静态副本全哨兵
    labels = [f"subj:{n}" for n in names]
    h5_names = [f"subj/{n}" for n in names]

    result = match_c3d_to_h5(c3d, labels, h5, h5_names)
    assert result.start_frame == 20
    assert result.rms_mm < 1e-3
    assert "R.Knee.Medial" not in result.matched_markers
    assert "R.ASIS" in result.matched_markers


def test_match_rms_uses_valid_coord_denominator():
    # RMS 分母必须是「有效坐标数」：有效坐标整体带 3mm 偏移，一半帧的 R.Knee
    # 被遮挡（哨兵）。正确 RMS 应≈3mm，而不是被哨兵帧稀释成更小值（§3.7 第 1 条）。
    names = ["R.ASIS", "L.ASIS", "R.Knee", "L.Knee"]
    n = 100
    rng = np.random.default_rng(11)
    c3d = rng.normal(size=(n, 4, 3)) * 100.0
    h5 = c3d + 3.0
    for f in range(0, n, 2):
        c3d[f, 2, :] = 9999999.0
        h5[f, 2, :] = 9999999.0
    labels = [f"s:{name}" for name in names]
    h5_names = [f"s/{name}" for name in names]
    result = match_c3d_to_h5(c3d, labels, h5, h5_names)
    assert result.start_frame == 0
    assert result.rms_mm == pytest.approx(3.0, abs=0.01)


def test_match_first_frame_nan_does_not_break_locating():
    # 首帧某 marker 遮挡（NaN）时，距离不能用 NaN 污染、退化成任意取候选
    # （§3.7 第 2 条）；应仍用有效坐标定位到正确起点。
    names = ["R.ASIS", "L.ASIS", "R.Knee", "L.Knee", "R.Ankle"]
    c3d = _make_points(100, names)
    c3d[0, 0, :] = np.nan  # 首帧 R.ASIS 遮挡
    h5 = np.concatenate([np.zeros((30, len(names), 3)), c3d, np.zeros((20, len(names), 3))])
    labels = [f"subj:{n}" for n in names]
    h5_names = [f"subj/{n}" for n in names]
    result = match_c3d_to_h5(c3d, labels, h5, h5_names)
    assert result.start_frame == 30
    assert result.rms_mm < 1e-3


def test_match_first_frame_fully_nan_raises():
    # 首帧全部遮挡（有效坐标不足），必须显式报错，不能静默取任意候选。
    names = ["R.ASIS", "L.ASIS", "R.Knee"]
    c3d = _make_points(20, names)
    c3d[0, :, :] = np.nan
    labels = [f"s:{n}" for n in names]
    h5_names = [f"s/{n}" for n in names]
    with pytest.raises(ValueError):
        match_c3d_to_h5(c3d, labels, c3d.copy(), h5_names)


def test_match_periodic_motion_not_unique():
    # 周期动作让两个不同起点都近似零 RMS，唯一性必须为 False（§3.7 第 4 条）。
    names = ["R.ASIS", "L.ASIS", "R.Knee", "L.Knee", "R.Ankle"]
    period = _make_points(50, names)
    h5 = np.concatenate([period, period])  # 同一段信号重复两次
    labels = [f"subj:{n}" for n in names]
    h5_names = [f"subj/{n}" for n in names]
    result = match_c3d_to_h5(period, labels, h5, h5_names)
    assert result.rms_mm < 1e-3  # 精确匹配
    assert not result.unique     # 但起点不唯一（0 与 50 都精确）


def test_match_middle_sentinel_excluded_from_rms():
    # 中间某帧某 marker 遮挡：匹配起点不受影响，RMS 只在有效坐标上算（§3.7 第 4 条）。
    names = ["R.ASIS", "L.ASIS", "R.Knee"]
    c3d = _make_points(100, names)
    c3d[50, 0, :] = 9999999.0  # 第 50 帧 R.ASIS 遮挡
    h5 = np.concatenate([np.zeros((10, len(names), 3)), c3d])
    labels = [f"subj:{n}" for n in names]
    h5_names = [f"subj/{n}" for n in names]
    result = match_c3d_to_h5(c3d, labels, h5, h5_names)
    assert result.start_frame == 10
    assert result.rms_mm < 1e-3
    assert result.max_error_mm < 1e-3


def test_match_h5_shorter_than_c3d():
    # H5 比 C3D 短：重叠帧数应被正确截断，起点仍为 0（§3.7 第 6 条）。
    names = ["R.ASIS", "L.ASIS", "R.Knee", "L.Knee", "R.Ankle"]
    c3d = _make_points(100, names)
    h5 = c3d[:60]
    labels = [f"subj:{n}" for n in names]
    h5_names = [f"subj/{n}" for n in names]
    result = match_c3d_to_h5(c3d, labels, h5, h5_names)
    assert result.start_frame == 0
    assert result.overlap_frames == 60
    assert result.rms_mm < 1e-3


def test_match_no_common_valid_marker_raises():
    # 公共 marker 均无有效数据，必须显式报错（§3.7 第 6 条）。
    names = ["R.ASIS", "L.ASIS"]
    c3d = _make_points(20, names, missing=(0, 1))
    h5 = _make_points(20, names, missing=(0, 1))
    labels = [f"s:{n}" for n in names]
    h5_names = [f"s/{n}" for n in names]
    with pytest.raises(ValueError):
        match_c3d_to_h5(c3d, labels, h5, h5_names)


# --------------------------------------------------------------------------
# 主机时钟 uint64 相减防溢出
# --------------------------------------------------------------------------
def test_clock_uint64_converted_to_int64(tmp_path):
    # host_monotonic_ns 是「开机以来的纳秒」，远小于 int64 上界；转 int64 无损。
    path = tmp_path / "clock.h5"
    base = np.uint64(5_000_000_000_000)
    with h5py.File(path, "w") as f:
        f["samples/host_monotonic_ns"] = base + np.arange(100, dtype=np.uint64) * 10_000_000
    with h5py.File(path, "r") as f:
        times = read_host_monotonic_ns(f)
    assert times.dtype == np.int64
    assert int(times[0]) == int(base)
    assert clock_health(times).monotonic


def test_clock_health_detects_decrease_without_wraparound():
    # 递减序列：转 int64 后 diff 为负，而 uint64 相减会环绕成巨大正数。
    times = np.array([5_000_000_000_000, 5_010_000_000_000, 5_000_000_000_000],
                     dtype=np.uint64)
    times_i64 = times.astype(np.int64)
    assert np.diff(times_i64)[1] < 0
    health = clock_health(times_i64)
    assert not health.monotonic
    assert health.n_decreasing == 1


# --------------------------------------------------------------------------
# 跺脚配对
# --------------------------------------------------------------------------
def _impulse_envelope(times, peak_times, amplitudes, *, noise, floor=0.05):
    """构造「包络」信号：底噪 + 高斯冲击峰，峰间距/幅值可自定义。"""
    env = np.full_like(np.asarray(times, dtype=np.float64), floor)
    rng = np.random.default_rng(7)
    env = env + rng.normal(0.0, noise, size=env.shape)
    for t0, amp in zip(peak_times, amplitudes):
        env += amp * np.exp(-0.5 * ((times - t0) / 0.03) ** 2)
    return env


def _build_pair_inputs(offset_s, n_stomps=5, *, imu_drop=(), force_drop=()):
    """生成 IMU/力包络 + 时间轴，跺脚在 imu 侧按 0.8s 等距，力侧 = imu + offset。"""
    imu_t = np.arange(0.0, 30.0, 0.01)
    force_t = np.arange(0.0, 35.0, 0.001)
    imu_peaks = [10.0 + 0.8 * k for k in range(n_stomps)]
    force_peaks = [p + offset_s for p in imu_peaks]
    imu_amp = [10.0] * n_stomps
    force_amp = [200.0] * n_stomps
    # 后面加一组更弱、更晚的「走路」周期峰，不应被选中
    imu_walk = [18.0 + 1.1 * k for k in range(6)]
    force_walk = [p + offset_s for p in imu_walk]
    imu_peaks_all = imu_peaks + imu_walk
    force_peaks_all = force_peaks + force_walk
    imu_amp_all = imu_amp + [3.0] * len(imu_walk)
    force_amp_all = force_amp + [60.0] * len(force_walk)

    imu_peaks_all = [p for k, p in enumerate(imu_peaks_all) if k not in imu_drop]
    imu_amp_all = [a for k, a in enumerate(imu_amp_all) if k not in imu_drop]
    force_peaks_all = [p for k, p in enumerate(force_peaks_all) if k not in force_drop]
    force_amp_all = [a for k, a in enumerate(force_amp_all) if k not in force_drop]

    imu_env = _impulse_envelope(imu_t, imu_peaks_all, imu_amp_all, noise=0.05)
    force_env = _impulse_envelope(force_t, force_peaks_all, force_amp_all, noise=1.0)
    return imu_t, imu_env, force_t, force_env


@pytest.mark.parametrize("n_stomps", [3, 4, 5])
def test_pair_stomps_recovers_offset(n_stomps):
    offset = 5.835
    imu_t, imu_env, force_t, force_env = _build_pair_inputs(offset, n_stomps)
    result = pair_stomps(imu_t, imu_env, force_t, force_env)
    assert result is not None
    assert len(result.pairs) == n_stomps
    assert result.median_offset_s == pytest.approx(offset, abs=0.05)
    assert result.confidence == "HIGH"


def test_pair_stomps_missed_force_peak_degrades_but_matches():
    offset = 5.835
    imu_t, imu_env, force_t, force_env = _build_pair_inputs(offset, 5, force_drop=(2,))
    result = pair_stomps(imu_t, imu_env, force_t, force_env)
    assert result is not None
    assert 3 <= len(result.pairs) <= 5
    assert result.median_offset_s == pytest.approx(offset, abs=0.1)


def test_pair_stomps_offset_sign_convention():
    # offset 正 → 力时间 = imu 时间 + offset；负 offset 也应正确恢复。
    for offset in (5.835, -2.5):
        imu_t, imu_env, force_t, force_env = _build_pair_inputs(offset, 5)
        result = pair_stomps(imu_t, imu_env, force_t, force_env)
        assert result.median_offset_s == pytest.approx(offset, abs=0.05)


def test_pair_stomps_low_confidence_on_jitter():
    offset = 5.835
    imu_t = np.arange(0.0, 30.0, 0.01)
    force_t = np.arange(0.0, 35.0, 0.001)
    imu_peaks = [10.0 + 0.8 * k for k in range(5)]
    imu_env = _impulse_envelope(imu_t, imu_peaks, [10.0] * 5, noise=0.05)
    # 给力峰加抖动，让各对 offset 离散度变大。抖动须落在配对带
    # ``_PAIR_BAND_S``（0.20s）以内，否则离群对会被正确剔除；±0.15s 会让
    # MAD≈0.15 > 0.05（高可信上界），但 < 0.20（中可信上界），故为 MEDIUM。
    jitter = np.array([0.0, -0.15, +0.15, -0.15, +0.15])
    force_peaks = [p + offset for p in imu_peaks]
    force_env = _impulse_envelope(
        force_t, [p + d for p, d in zip(force_peaks, jitter)], [200.0] * 5, noise=0.5
    )
    result = pair_stomps(imu_t, imu_env, force_t, force_env)
    assert result is not None
    assert len(result.pairs) == 5
    assert result.mad_s > 0.05
    assert result.confidence in ("MEDIUM", "LOW")


def test_pair_stomps_returns_none_without_stomps():
    imu_t = np.arange(0.0, 10.0, 0.01)
    force_t = np.arange(0.0, 10.0, 0.001)
    rng = np.random.default_rng(3)
    imu_env = rng.normal(0.1, 0.05, size=imu_t.shape)
    force_env = rng.normal(1.0, 0.5, size=force_t.shape)
    assert pair_stomps(imu_t, imu_env, force_t, force_env) is None


# --------------------------------------------------------------------------
# 漂移估计：短记录 → UNASSESSED（None）
# --------------------------------------------------------------------------
def test_drift_unassessed_on_short_record():
    # 跨度 < 5s（跺脚本身只有 ~3s），漂移应为 None（UNASSESSED），不伪造 0。
    imu_t, imu_env, force_t, force_env = _build_pair_inputs(5.835, 5)
    result = pair_stomps(imu_t, imu_env, force_t, force_env)
    assert result.drift_ppm is None


# --------------------------------------------------------------------------
# 高通包络
# --------------------------------------------------------------------------
def test_highpass_envelope_removes_baseline():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 5.0, 0.001)
    signal = 800.0 + 0.5 * t + rng.normal(0.0, 1.0, size=t.shape)  # 大基线
    env = highpass_envelope(signal, 1000.0, cutoff_hz=2.0)
    # 无冲击时包络应接近 0（基线被移除）
    assert float(np.median(env)) < 5.0
