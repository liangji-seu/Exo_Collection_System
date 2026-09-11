"""data_studio.global_preview.read_truth_preview 的纯逻辑测试。

只验证 ground_truth.csv 列提取契约：按列名精确匹配髋力矩（hip_flexion_r/l）与
IMU 姿态（imu_roll/pitch/yaw），缺失某列时该组为空、不抛异常。不启动 Qt 窗口。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from exo_collection.apps.data_studio.global_preview import (  # noqa: E402
    read_gaitway_speed,
    read_truth_preview,
)


def _write_gt(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_read_truth_preview_extracts_hip_and_pitch(tmp_path: Path) -> None:
    header = "time_s,imu_acc_x,imu_pitch,imu_yaw,hip_flexion_r,hip_flexion_l"
    path = _write_gt(
        tmp_path / "ground_truth.csv",
        header,
        [
            "0.00,0,10.0,0,1.0,2.0",
            "0.01,0,11.0,0,1.5,2.5",
        ],
    )
    preview = read_truth_preview(path)
    assert preview is not None
    np.testing.assert_allclose(preview.time_s, [0.00, 0.01])
    assert preview.moment_channels == ("hip_flexion_r", "hip_flexion_l")
    np.testing.assert_allclose(preview.moment_values[:, 0], [1.0, 1.5])
    np.testing.assert_allclose(preview.moment_values[:, 1], [2.0, 2.5])
    # imu_roll 不在表头，按声明顺序只提取 pitch/yaw。
    assert preview.imu_channels == ("imu_pitch", "imu_yaw")
    np.testing.assert_allclose(preview.imu_values[:, 0], [10.0, 11.0])
    np.testing.assert_allclose(preview.imu_values[:, 1], [0.0, 0.0])


def test_read_truth_preview_missing_imu_is_empty(tmp_path: Path) -> None:
    header = "time_s,hip_flexion_r,hip_flexion_l"
    path = _write_gt(
        tmp_path / "ground_truth.csv",
        header,
        ["0.00,1.0,2.0", "0.01,1.5,2.5"],
    )
    preview = read_truth_preview(path)
    assert preview is not None
    assert preview.moment_channels == ("hip_flexion_r", "hip_flexion_l")
    assert preview.imu_channels == ()
    assert preview.imu_values.shape == (2, 0)


def test_read_truth_preview_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_truth_preview(tmp_path / "nope.csv") is None


def test_read_truth_preview_returns_none_for_no_matching_columns(tmp_path: Path) -> None:
    path = _write_gt(
        tmp_path / "ground_truth.csv",
        "time_s,foo,bar",
        ["0.00,1.0,2.0"],
    )
    assert read_truth_preview(path) is None


_SPEED_HEADER = "Time (s)\tSpeed (m/s)\tSpeed target (m/s)"


def _write_speed_txt(
    path: Path, header: str, rows: list[str], preamble: str = ""
) -> Path:
    text = (preamble + "\n" if preamble else "") + header + "\n" + "\n".join(rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_gaitway_speed_parses_and_shifts(tmp_path: Path) -> None:
    _write_speed_txt(
        tmp_path / "trial.txt",
        _SPEED_HEADER,
        ["0.000\t0.0\t0.0", "0.001\t0.5\t0.5", "0.002\t1.0\t1.0"],
    )
    trace = read_gaitway_speed(tmp_path, offset_s=2.5)
    assert trace is not None
    np.testing.assert_allclose(trace.time_s_c3d, [-2.5, -2.499, -2.498])
    np.testing.assert_allclose(trace.speed, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(trace.speed_target, [0.0, 0.5, 1.0])


def test_read_gaitway_speed_reads_offset_from_manifest(tmp_path: Path) -> None:
    _write_speed_txt(
        tmp_path / "trial.txt", _SPEED_HEADER, ["0.000\t0.0\t0.0", "0.001\t0.5\t0.5"]
    )
    run_dir = tmp_path / "derived" / "opensim" / "run_x"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"sync": {"gaitway_offset_s": 2.5}}), encoding="utf-8"
    )
    trace = read_gaitway_speed(tmp_path)
    assert trace is not None
    np.testing.assert_allclose(trace.time_s_c3d, [-2.5, -2.499])


def test_read_gaitway_speed_offset_falls_back_to_zero(tmp_path: Path) -> None:
    _write_speed_txt(
        tmp_path / "trial.txt", _SPEED_HEADER, ["1.000\t0.0\t0.0", "1.001\t0.5\t0.5"]
    )
    trace = read_gaitway_speed(tmp_path)
    assert trace is not None
    np.testing.assert_allclose(trace.time_s_c3d, [1.0, 1.001])


def test_read_gaitway_speed_target_is_optional(tmp_path: Path) -> None:
    _write_speed_txt(
        tmp_path / "trial.txt",
        "Time (s)\tSpeed (m/s)",
        ["0.000\t0.0", "0.001\t0.5"],
    )
    trace = read_gaitway_speed(tmp_path, offset_s=0.0)
    assert trace is not None
    np.testing.assert_allclose(trace.speed, [0.0, 0.5])
    assert trace.speed_target.size == 0


def test_read_gaitway_speed_skips_non_gaitway_txt(tmp_path: Path) -> None:
    _write_speed_txt(tmp_path / "a_force.txt", "Time (s)\tFoo", ["0.000\t1.0"])
    _write_speed_txt(
        tmp_path / "b_speed.txt",
        _SPEED_HEADER,
        ["0.000\t0.0\t0.0", "0.001\t1.0\t1.0"],
    )
    trace = read_gaitway_speed(tmp_path, offset_s=0.0)
    assert trace is not None
    np.testing.assert_allclose(trace.speed, [0.0, 1.0])


def test_read_gaitway_speed_returns_none_without_txt(tmp_path: Path) -> None:
    assert read_gaitway_speed(tmp_path) is None


def test_read_gaitway_speed_returns_none_without_speed_column(tmp_path: Path) -> None:
    _write_speed_txt(
        tmp_path / "force.txt",
        "Time (s)\tFzL(N)\tGRFz vertical (N)",
        ["0.000\t1.0\t2.0"],
    )
    assert read_gaitway_speed(tmp_path) is None
