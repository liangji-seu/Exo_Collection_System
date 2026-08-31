"""C3D reader / inspection 测试（用真实 XINGYING c3d，找不到则 skip）。"""

from pathlib import Path

import pytest

from postprocess.c3d.reader import read_c3d
from postprocess.c3d.inspect_c3d import inspect
from postprocess.c3d.extract_forces import classify_channel, detect_grf_mode, classify_channels


def _find_test_c3d():
    root = Path(__file__).resolve().parents[3]  # -> 1_exo_数据采集方案
    hits = list(root.rglob("*.c3d"))
    return hits[0] if hits else None


C3D = _find_test_c3d()


def test_classify_channel():
    assert classify_channel("Fx1").kind == "Fx"
    assert classify_channel("Fx1").plate_index == 1
    assert classify_channel("COPx1").kind == "COPx"
    assert classify_channel("Tz1").kind == "Tz"


def test_detect_grf_mode_total():
    classes = classify_channels(["Fx1", "Fy1", "Fz1", "COPx1", "COPy1", "Tz1"])
    assert detect_grf_mode(classes) == "TOTAL_ONLY"


@pytest.mark.skipif(C3D is None, reason="未找到测试 c3d")
def test_read_real_c3d():
    data = read_c3d(C3D)
    assert data.points_mm.ndim == 3
    assert data.points_mm.shape[2] == 3  # xyz
    assert data.n_frames > 0
    assert len(data.point_labels) == data.points_mm.shape[1]
    assert data.analogs.shape[0] == data.n_frames


@pytest.mark.skipif(C3D is None, reason="未找到测试 c3d")
def test_inspect_real_c3d():
    rep = inspect(read_c3d(C3D))
    assert rep["point"]["n_points"] == 64
    assert rep["force"]["grf_mode"] == "TOTAL_ONLY"
    # 整文件（static 18 真实 + dynamic 14 真实 = 32）按短名分类
    assert rep["marker_class_counts"]["real"] == 32
    assert rep["marker_class_counts"]["suspected_virtual"] == 32
    assert rep["sync"]["integer_ratio"] is True
