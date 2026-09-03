"""抗混叠滤波 / 分段滤波（prompt6 §3.8）单元测试。

验证：接触段独立零相位低通（非接触帧 NaN、不跨边界振铃），以及 GRF 在降采样前
先抗混叠——默认 20 Hz 截止可去除 1000 Hz → 100 Hz 重采样带来的高频混叠。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.filtering import lowpass_segmented
from pipeline.gaitway import GaitwayAsciiData, build_bilateral_grf


def test_lowpass_segmented_isolates_segments_and_nans_gaps() -> None:
    t = np.arange(0.0, 2.0, 0.001)  # 1000 Hz
    # DC + 400 Hz 高频，中间有一段「非接触」gap。
    signal = 500.0 + 60.0 * np.sin(2 * np.pi * 400.0 * t)
    valid = np.ones_like(t, dtype=bool)
    valid[900:1100] = False  # gap

    out = lowpass_segmented(signal, valid, sample_rate_hz=1000.0, cutoff_hz=20.0)

    # 非接触帧置 NaN，绝不让滤波振铃泄漏进 gap。
    assert np.all(np.isnan(out[900:1100]))
    # 接触段仍是有限值，且 400 Hz 噪声被 20 Hz 低通衰减到接近 DC。
    seg = out[500:900]
    raw = signal[500:900]
    assert np.all(np.isfinite(seg))
    assert float(np.std(seg)) < 10.0
    assert float(np.std(seg)) < float(np.std(raw))  # 明显衰减
    # 两段接触段分别滤波，互不跨越 gap 泄漏。
    assert np.all(np.isfinite(out[1100:1400]))


def test_lowpass_segmented_returns_input_when_all_invalid() -> None:
    x = np.array([1.0, 2.0, 3.0])
    out = lowpass_segmented(x, np.zeros(3, dtype=bool), 100.0, 20.0)
    np.testing.assert_allclose(out, x)


def _gaitway(rate: float, t: np.ndarray, fz_r: np.ndarray) -> GaitwayAsciiData:
    zeros = np.zeros_like(t)
    return GaitwayAsciiData(
        path=Path("x.txt"),
        metadata={"Sample rate (Hz)": str(int(rate))},
        time_s=t,
        columns={
            "FzR(N)": fz_r, "FyR(N)": zeros, "FxR(N)": zeros,
            "CoPxR(m)": zeros, "CoPyR(m)": zeros,
            "FzL(N)": zeros, "FyL(N)": zeros, "FxL(N)": zeros,
            "CoPxL(m)": zeros, "CoPyL(m)": zeros,
            "GRFz vertical (N)": fz_r,
        },
    )


def test_build_bilateral_grf_antialias_removes_high_frequency() -> None:
    """240 Hz 噪声以 100 Hz 重采样会混叠成 40 Hz；20 Hz 截止应先把它滤掉。"""
    native_rate = 1000.0
    t = np.arange(0.0, 3.0, 1.0 / native_rate)
    fz = 400.0 + 150.0 * np.sin(2 * np.pi * 240.0 * t)  # 全部接触
    gaitway = _gaitway(native_rate, t, fz)
    query = np.arange(0.0, 3.0, 0.01)  # mocap 100 Hz

    def z_std(cutoff):
        feet, _, qc = build_bilateral_grf(
            gaitway, query, 0.0, np.eye(3),
            force_threshold_N=50.0, cutoff_hz=cutoff,
            opensim_x_sign=1.0, opensim_z_sign=1.0,
        )
        # 单位 fp→mocap 旋转 + 符号 +1 下，垂直力（FzR）落在 OpenSim 帧第 1 列。
        return float(np.std(feet[0]["force"][:, 1])), qc

    filtered_std, filtered_qc = z_std(20.0)
    aliased_std, aliased_qc = z_std(None)

    # 20 Hz 抗混叠后高频噪声被去除，输出平稳（接近 DC 400）。
    assert filtered_std < 10.0
    # 不滤波直接重采样 → 240 Hz 混叠成 40 Hz 的低频摆动，std 明显偏大。
    assert aliased_std > 50.0
    # 截止频率进入 QC 审计（默认 20，None 时如实记为 None）。
    assert filtered_qc["grf_cutoff_hz"] == 20.0
    assert aliased_qc["grf_cutoff_hz"] is None
