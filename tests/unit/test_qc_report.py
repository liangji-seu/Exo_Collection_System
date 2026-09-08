"""data_studio.qc_report 的纯逻辑测试：heel strike、归一化步态周期、run 目录加载。

只验证分析函数与 ``load_gait_qc`` 的数据契约，不启动报告窗口。合成一个最小 run 目录
（result.json + viewer/*.npy + IK/ID .mot + ground_truth.csv）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from exo_collection.apps.data_studio.qc_report import (  # noqa: E402
    detect_heel_strikes,
    load_gait_qc,
    normalize_gait_cycles,
    representative_cycle,
)


def _write_mot(path: Path, columns: list[str], rows: list[list[float]]) -> None:
    lines = ["test", "endheader", "\t".join(columns)]
    lines.extend("\t".join(f"{v:.6g}" for v in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run_dir(session_dir: Path, *, n: int = 400) -> Path:
    run_dir = session_dir / "derived" / "opensim" / "run_test"
    viewer = run_dir / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)

    time_s = np.arange(n, dtype=np.float64) * 0.01
    grf = np.zeros((n, 2, 3), dtype=np.float32)
    for i in range(n):
        stance = 800.0 if (i % 100) >= 40 else 0.0
        grf[i, 0, 1] = stance          # 右 Fz（y 为竖直）
        grf[i, 1, 1] = stance * 0.9    # 左 Fz
    moments = np.zeros((n, 6), dtype=np.float32)
    moments[:, 0] = np.sin(time_s) * 10.0
    moments[:, 1] = np.cos(time_s) * 8.0
    np.save(viewer / "time_s.npy", time_s)
    np.save(viewer / "grf.npy", grf)
    np.save(viewer / "moments.npy", moments)

    ik_path = run_dir / "hh19_static_calibrated_ik.mot"
    _write_mot(
        ik_path,
        ["time", "hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
         "ankle_angle_r", "ankle_angle_l"],
        [[t, 30 * np.sin(t), 20 * np.sin(t), 40 * np.sin(t), 40 * np.sin(t),
          10 * np.sin(t), 10 * np.sin(t)] for t in time_s],
    )

    id_path = run_dir / "hh19_static_calibrated_id.mot"
    _write_mot(
        id_path,
        ["time", "pelvis_tilt_moment", "pelvis_list_moment", "pelvis_rotation_moment",
         "pelvis_tx_force", "pelvis_ty_force", "pelvis_tz_force"],
        [[t, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0] for t in time_s],
    )

    result = {
        "processing": {"marker_cutoff_hz": 6.0, "grf_cutoff_hz": 20.0},
        "marker_qc": {
            "overall": {"rms_mean_cm": 1.0, "rms_p95_cm": 1.2,
                        "max_marker_p95_cm": 2.0, "max_marker_max_cm": 2.5},
            "markers": {
                "A": {"mean_cm": 0.5, "p95_cm": 0.6, "max_cm": 0.7},
                "B": {"mean_cm": 1.0, "p95_cm": 1.5, "max_cm": 2.0},
            },
        },
        "id_qc": {"residual_force": {"rms_N": 50.0, "p95_N": 80.0}},
        "qc": {"status": "WARN", "summary": "test"},
        "viewer": {"viewer_dir": str(viewer), "frame_rate_hz": 100.0, "n_frames": n},
        "files": {"ik": str(ik_path), "id": str(id_path), "viewer_dir": str(viewer)},
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return run_dir


def _write_gt(session_dir: Path, time_s: np.ndarray) -> None:
    header = ("time_s,imu_pitch,imu_roll,imu_yaw,hip_flexion_r,hip_flexion_l,"
              "knee_angle_r,knee_angle_l,ankle_angle_r,ankle_angle_l")
    rows = [
        f"{t:.6f},{np.sin(t):.6f},{np.cos(t):.6f},{t:.6f},0,0,0,0,0,0"
        for t in time_s
    ]
    (session_dir / "ground_truth.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )


def _square_fz(n: int = 400, period: int = 100, stance_frac: float = 0.6) -> np.ndarray:
    fz = np.zeros(n, dtype=np.float64)
    onset = int(period * (1.0 - stance_frac))
    for i in range(n):
        if (i % period) >= onset:
            fz[i] = 800.0
    return fz


# ── detect_heel_strikes ──────────────────────────────────────────


def test_detect_heel_strikes_rising_edges() -> None:
    fz = _square_fz()
    heel = detect_heel_strikes(fz, frame_rate=100.0)
    # stance 从 phase>=40 开始 → 上升沿在 40, 140, 240, 340。
    np.testing.assert_array_equal(heel, [40, 140, 240, 340])


def test_detect_heel_strikes_min_gap() -> None:
    fz = np.zeros(120)
    fz[10:20] = 800.0   # 上升沿 10
    fz[60:70] = 800.0   # 上升沿 60（间隔 50 帧）
    # min_gap_s=0.5 → 50 帧：间隔 50 不满足严格 >，仍保留第二个。
    assert list(detect_heel_strikes(fz, min_gap_s=0.5, frame_rate=100.0)) == [10, 60]
    # min_gap_s=0.6 → 60 帧：间隔 50 < 60，第二个被丢弃。
    assert list(detect_heel_strikes(fz, min_gap_s=0.6, frame_rate=100.0)) == [10]


def test_detect_heel_strikes_empty() -> None:
    assert detect_heel_strikes(np.zeros(0)).size == 0
    assert detect_heel_strikes(np.zeros(100)).size == 0


# ── normalize_gait_cycles / representative_cycle ─────────────────


def test_normalize_gait_cycles_ramp() -> None:
    n = 400
    time_s = np.arange(n, dtype=np.float64) * 0.01
    signal = time_s.copy()  # 线性斜坡：各周期形状一致
    heel = np.array([40, 140, 240, 340])
    x, mean, std = normalize_gait_cycles(time_s, signal, heel)
    np.testing.assert_allclose(x, np.linspace(0.0, 100.0, 101))
    assert mean.shape == (101,)
    # 三个周期的斜坡起点分别为 0.40/1.40/2.40 s，均值 = 1.40 + x/100。
    assert abs(mean[50] - 1.90) < 1e-9
    # 三个周期斜率相同、截距不同 → std = 截距的标准差（与 x 无关）。
    assert abs(std[50] - np.std([0.90, 1.90, 2.90])) < 1e-9


def test_normalize_gait_cycles_insufficient() -> None:
    x, mean, std = normalize_gait_cycles(np.arange(10) * 0.01, np.zeros(10), np.array([1]))
    assert mean.size == 0 and std.size == 0


def test_representative_cycle_picks_median_duration() -> None:
    n = 400
    time_s = np.arange(n, dtype=np.float64) * 0.01
    signal = time_s.copy()
    heel = np.array([40, 140, 240, 340])
    rep = representative_cycle(time_s, signal, heel)
    assert rep is not None
    x, values = rep
    assert x.shape == (101,)
    # 时长中位数为 1.0 s（各周期相等）→ 取第一个周期，起点 0.40 s。
    assert abs(values[50] - 0.90) < 1e-9


# ── load_gait_qc ────────────────────────────────────────────────


def test_load_gait_qc_full(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    n = 400
    time_s = np.arange(n, dtype=np.float64) * 0.01
    _make_run_dir(session_dir, n=n)
    _write_gt(session_dir, time_s)

    data = load_gait_qc(session_dir)
    assert data is not None
    assert data.time_s.shape == (n,)
    assert data.grf_fz_r.shape == (n,)
    assert data.grf_fz_l.shape == (n,)
    assert data.moments.shape == (n, 6)
    assert data.angles is not None and data.angles.shape == (n, 6)
    assert data.angle_names == ("hip_flexion_r", "hip_flexion_l", "knee_angle_r",
                                "knee_angle_l", "ankle_angle_r", "ankle_angle_l")
    assert data.marker_qc_overall is not None
    assert data.id_residual_force == {"rms_N": 50.0, "p95_N": 80.0}
    assert data.id_residual_moment is not None
    np.testing.assert_allclose(data.id_residual_moment["rms_Nm"], np.sqrt(14.0), rtol=1e-9)
    assert data.marker_cutoff_hz == 6.0
    assert data.grf_cutoff_hz == 20.0
    assert data.qc_status == "WARN"
    assert data.imu_pitch is not None and data.imu_pitch.shape == (n,)


def test_load_gait_qc_no_run_dir(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    assert load_gait_qc(session_dir) is None


def test_load_gait_qc_half_run(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "derived" / "opensim" / "run_x").mkdir(parents=True)
    assert load_gait_qc(session_dir) is None
