"""opensim_overlay 的纯 NumPy 单测：从构造的 run 目录加载验证三信号。

不依赖 opensim / Qt，用 tmp_path 伪造 ``derived/opensim/run_*/`` 目录树，
覆盖「取竖直 Fz（y 分量）」「跳过半成品 run」「缺数据回退 None」等关键行为。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from exo_collection.apps.data_studio.opensim_overlay import (
    find_latest_run_dir,
    load_hip_validation,
)


def _write_mot(path: Path, time_s: np.ndarray, angle: np.ndarray) -> None:
    lines = [
        "Coordinates",
        "version=1",
        f"nRows={time_s.size}",
        "nColumns=2",
        "inDegrees=yes",
        "endheader",
        "time\thip_flexion_r",
    ]
    for t, a in zip(time_s, angle):
        lines.append(f"{t}\t{a}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_run(session: Path, name: str, *, complete: bool = True) -> Path:
    run = session / "derived" / "opensim" / name
    run.mkdir(parents=True)
    viewer = run / "viewer"
    viewer.mkdir()

    n = 5
    time_s = np.arange(n, dtype=np.float64) / 100.0  # 0 .. 0.04 s
    grf = np.zeros((n, 2, 3), dtype=np.float64)
    grf[:, 0, 1] = np.array([0.0, 100.0, 300.0, 500.0, 800.0])  # 右脚竖直 Fz
    grf[:, 1, 1] = np.array([900.0, 800.0, 400.0, 100.0, 0.0])  # 左脚竖直 Fz
    moments = np.zeros((n, 6), dtype=np.float64)
    moments[:, 0] = np.array([0.0, -10.0, -45.0, -30.0, 5.0])  # hip_flexion_r

    np.save(viewer / "time_s.npy", time_s)
    np.save(viewer / "grf.npy", grf)
    np.save(viewer / "moments.npy", moments)

    ik_path = run / "hh19_static_calibrated_ik.mot"
    _write_mot(ik_path, time_s, np.array([10.0, 15.0, 19.0, 17.0, 12.0]))

    if complete:
        result = {
            "files": {"ik": str(ik_path)},
            "viewer": {"viewer_dir": str(viewer)},
        }
        (run / "result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    return run


def test_load_hip_validation_returns_none_without_derived(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    assert load_hip_validation(session) is None


def test_load_hip_validation_reads_fz_angle_moment(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _build_run(session, "run_20260101_000000_0001")

    validation = load_hip_validation(session)

    assert validation is not None
    assert validation.run_dir.name == "run_20260101_000000_0001"
    assert validation.time_s.shape == (5,)
    # 右脚竖直 Fz = grf[:, 0, 1]（y 分量）。
    np.testing.assert_allclose(
        validation.fz_r, [0.0, 100.0, 300.0, 500.0, 800.0]
    )
    np.testing.assert_allclose(
        validation.hip_angle_r, [10.0, 15.0, 19.0, 17.0, 12.0]
    )
    np.testing.assert_allclose(
        validation.hip_moment_r, [0.0, -10.0, -45.0, -30.0, 5.0]
    )


def test_load_hip_validation_skips_incomplete_newer_run(tmp_path: Path) -> None:
    session = tmp_path / "session"
    # 较新的 run 缺 result.json（OpenSim 跑完但 viewer 导出中断），应回退到较旧完整 run。
    _build_run(session, "run_20260101_000000_0002", complete=False)
    _build_run(session, "run_20260101_000000_0001", complete=True)

    validation = load_hip_validation(session)

    assert validation is not None
    assert validation.run_dir.name == "run_20260101_000000_0001"


def test_find_latest_run_dir_returns_newest(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _build_run(session, "run_20260101_000000_0001")
    _build_run(session, "run_20260101_000000_0002")

    latest = find_latest_run_dir(session)

    assert latest is not None
    assert latest.name == "run_20260101_000000_0002"


def test_load_hip_validation_returns_none_when_missing_hip_angle(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run = _build_run(session, "run_20260101_000000_0001")
    # 把 IK .mot 换成不含 hip_flexion_r 列的文件。
    ik_path = run / "hh19_static_calibrated_ik.mot"
    ik_path.write_text(
        "Coordinates\nversion=1\nnRows=1\nnColumns=2\ninDegrees=yes\n"
        "endheader\ntime\tknee_angle_r\n0.0\t1.0\n",
        encoding="utf-8",
    )

    assert load_hip_validation(session) is None
