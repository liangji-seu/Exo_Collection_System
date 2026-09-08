"""gait_baseline 的纯 NumPy 单测：LUT、步态相位、baseline 映射与端到端组装。

这里刻意用 ``types.SimpleNamespace`` 伪造 playback / IMU 序列，避免引入 Qt / h5py，
让 gait_baseline 保持「无 Qt、可离线单测」的约束。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exo_collection.apps.data_studio.gait_baseline import (
    baseline_torque_from_phase,
    build_hip_baseline,
    estimate_gait_phase,
    find_right_leg_imu,
    load_hip_torque_lut,
    moment_imu_offset_s,
    right_leg_pitch,
)


# -- helpers ---------------------------------------------------------------


def _cos_strides(n_strides: int, samples_per_stride: int, amplitude: float = 10.0):
    """周期余弦：极大值（脚跟触地）落在每个步距边界 (frac = 0, 1, …)。"""
    frac = (np.arange(n_strides * samples_per_stride) % samples_per_stride) / samples_per_stride
    return amplitude * np.cos(2.0 * np.pi * frac)


def _series(time_s, pitch, channels=("acc_x", "pitch", "yaw"), labels=("imu_right_leg",)):
    pitch = np.asarray(pitch, dtype=np.float64)
    values = np.zeros((pitch.size, len(channels)), dtype=np.float64)
    values[:, 1] = pitch  # pitch 固定放 index 1
    return SimpleNamespace(
        time_s=np.asarray(time_s, dtype=np.float64),
        values=values,
        channels=channels,
        sensor_labels=labels,
    )


# -- LUT -------------------------------------------------------------------


def test_load_hip_torque_lut_loads_bundled_lut() -> None:
    phase, torque = load_hip_torque_lut()

    assert phase.shape == (1001,)
    assert torque.shape == (1001,)
    assert float(phase[0]) == 0.0 and float(phase[-1]) == 100.0
    assert np.all(np.diff(phase) > 0.0)  # 升序
    assert -11.5 <= float(torque.min()) <= -11.0
    assert 18.5 <= float(torque.max()) <= 18.7


def test_load_hip_torque_lut_sorts_unsorted_input(tmp_path) -> None:
    path = tmp_path / "lut.csv"
    path.write_text("50,1.0\n0,2.0\n100,3.0\n", encoding="utf-8")

    phase, torque = load_hip_torque_lut(path)

    np.testing.assert_allclose(phase, [0.0, 50.0, 100.0])
    np.testing.assert_allclose(torque, [2.0, 1.0, 3.0])


def test_load_hip_torque_lut_rejects_malformed(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("1.0\n2.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hip_torque_lut(path)


# -- estimate_gait_phase ---------------------------------------------------


def test_estimate_gait_phase_recovers_sawtooth_phase() -> None:
    fs = 100.0
    pitch = _cos_strides(6, 100)  # 6 个步距，脚跟触地在 0/100/…/500

    phase = estimate_gait_phase(pitch, fs)

    # 内侧步距边界相位回到 0、中间回到 50（首尾步距受卷积零填充影响，不在此断言）。
    assert phase[150] == pytest.approx(50.0, abs=2.0)
    assert phase[200] == pytest.approx(0.0, abs=1.0)
    assert phase[250] == pytest.approx(50.0, abs=2.0)
    assert phase[300] == pytest.approx(0.0, abs=1.0)
    assert phase[400] == pytest.approx(0.0, abs=1.0)
    assert phase[450] == pytest.approx(50.0, abs=2.0)
    # 绝大多数采样点都落在有效步距内。
    assert float(np.isfinite(phase).mean()) > 0.8


def test_estimate_gait_phase_constant_pitch_is_all_nan() -> None:
    phase = estimate_gait_phase(np.full(200, 3.0), 100.0)
    assert np.isnan(phase).all()


def test_estimate_gait_phase_marks_long_pause_as_nan() -> None:
    fs = 100.0
    before = _cos_strides(3, 100)
    pause = np.zeros(500)  # 5 s 停顿
    after = _cos_strides(3, 100)
    pitch = np.concatenate([before, pause, after])

    phase = estimate_gait_phase(pitch, fs)

    # 停顿中央没有步态。
    pause_start = before.size
    assert np.isnan(phase[pause_start + 200 : pause_start + 300]).all()
    # 停顿两侧的步距中央仍有有效相位。
    assert np.isfinite(phase[50])
    after_mid = before.size + pause.size + 150
    assert np.isfinite(phase[after_mid])


def test_estimate_gait_phase_requires_minimum_samples() -> None:
    # 采样过少（< 3）时无法估计相位 → 全 NaN。
    assert np.isnan(estimate_gait_phase(np.array([1.0, 2.0]), 100.0)).all()
    # 采样率为 0 / 非有限 → 全 NaN。
    assert np.isnan(estimate_gait_phase(np.arange(10.0), 0.0)).all()


# -- baseline_torque_from_phase -------------------------------------------


def test_baseline_torque_from_phase_interpolates_and_preserves_nan() -> None:
    phase = np.array([0.0, 50.0, 100.0, np.nan])
    lut_phase = np.array([0.0, 100.0])
    lut_torque = np.array([0.0, 20.0])

    torque = baseline_torque_from_phase(phase, lut_phase, lut_torque)

    np.testing.assert_allclose(torque[:3], [0.0, 10.0, 20.0])
    assert np.isnan(torque[3])


# -- find_right_leg_imu / right_leg_pitch ----------------------------------


def test_find_right_leg_imu_matches_right_label() -> None:
    left = _series(np.arange(10), np.zeros(10), labels=("imu_left_leg",))
    pelvis = _series(np.arange(10), np.zeros(10), labels=("imu_pelvis",))
    right = _series(np.arange(10), np.zeros(10), labels=("imu_right_leg",))
    playback = SimpleNamespace(imu_sensors=(left, pelvis, right), imu=None)

    assert find_right_leg_imu(playback) is right


def test_find_right_leg_imu_falls_back_to_imu_series() -> None:
    right = _series(np.arange(10), np.zeros(10), labels=("imu_right_leg",))
    playback = SimpleNamespace(imu_sensors=(), imu=right)

    assert find_right_leg_imu(playback) is right


def test_find_right_leg_imu_returns_none_without_right() -> None:
    left = _series(np.arange(10), np.zeros(10), labels=("imu_left_leg",))
    playback = SimpleNamespace(imu_sensors=(left,), imu=None)
    assert find_right_leg_imu(playback) is None


def test_right_leg_pitch_extracts_pitch_and_sample_rate() -> None:
    time_s = np.arange(200) / 100.0
    pitch = _cos_strides(2, 100)
    playback = SimpleNamespace(imu_sensors=(_series(time_s, pitch),), imu=None)

    result = right_leg_pitch(playback)

    assert result is not None
    t_out, pitch_out, fs = result
    np.testing.assert_allclose(t_out, time_s)
    np.testing.assert_allclose(pitch_out, pitch)
    assert fs == pytest.approx(100.0, rel=0.01)


def test_right_leg_pitch_returns_none_without_pitch_channel() -> None:
    series = SimpleNamespace(
        time_s=np.arange(10, dtype=np.float64),
        values=np.zeros((10, 2)),
        channels=("acc_x", "acc_y"),
        sensor_labels=("imu_right_leg",),
    )
    playback = SimpleNamespace(imu_sensors=(series,), imu=None)
    assert right_leg_pitch(playback) is None


# -- moment_imu_offset_s ---------------------------------------------------


def test_moment_imu_offset_s_reads_c3d_t0_from_derived_manifest(tmp_path) -> None:
    session = tmp_path / "session"
    (session / "derived" / "opensim" / "run1").mkdir(parents=True)
    (session / "derived" / "opensim" / "run1" / "manifest.json").write_text(
        json.dumps({"sync": {"c3d_t0_host_monotonic_ns": 1_000_000_000}}),
        encoding="utf-8",
    )
    manifest_path = session / ".exo" / "manifest.json"
    playback = SimpleNamespace(
        manifest_path=manifest_path,
        formal_t0_host_monotonic_ns=900_000_000,
    )

    assert moment_imu_offset_s(playback) == pytest.approx(0.1)


def test_moment_imu_offset_s_falls_back_to_zero(tmp_path) -> None:
    session = tmp_path / "session"
    manifest_path = session / ".exo" / "manifest.json"
    playback = SimpleNamespace(
        manifest_path=manifest_path,
        formal_t0_host_monotonic_ns=900_000_000,
    )
    assert moment_imu_offset_s(playback) == 0.0


# -- build_hip_baseline ----------------------------------------------------


def test_build_hip_baseline_returns_none_without_right_imu() -> None:
    playback = SimpleNamespace(
        imu_sensors=(),
        imu=None,
        formal_t0_host_monotonic_ns=0,
        manifest_path=Path("manifest.json"),
    )
    assert build_hip_baseline(playback) is None


def test_build_hip_baseline_produces_torque_on_c3d_time(tmp_path) -> None:
    fs = 100.0
    time_s = np.arange(600) / fs
    pitch = _cos_strides(6, 100)
    playback = SimpleNamespace(
        imu_sensors=(_series(time_s, pitch),),
        imu=None,
        formal_t0_host_monotonic_ns=900_000_000,
        manifest_path=tmp_path / "session" / ".exo" / "manifest.json",
    )

    result = build_hip_baseline(playback)

    assert result is not None
    t_c3d, torque = result
    assert t_c3d.shape == (600,)
    assert torque.shape == (600,)
    # 至少覆盖一个完整步距的 baseline，且落在 LUT 力矩范围内。
    finite = torque[np.isfinite(torque)]
    assert finite.size > 100
    assert float(finite.min()) >= -12.0 and float(finite.max()) <= 19.0
    # 内侧步距（index 250）的相位 ≈ 50% → 力矩应接近 LUT 相位 50% 处的值。
    phase, lut_torque = load_hip_torque_lut()
    expected_mid = float(np.interp(50.0, phase, lut_torque))
    assert torque[250] == pytest.approx(expected_mid, abs=1.0)
