"""Exo Calculate 回放页（viewer）的契约测试。

不 import opensim、不联网；用合成 ``viewer/*.npy`` + ``viewer_meta.json`` 验证
``load_viewer_data`` 与 ``ViewerWidget`` 的加载/建帧，锁定导出格式契约。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from exo_collection.apps.calculate.viewer import (  # noqa: E402
    ViewerWidget,
    load_viewer_data,
)

_MODEL_NAMES = [
    "V.Sacral", "R.ASIS", "L.ASIS", "R.Thigh", "L.Thigh", "R.Knee", "L.Knee",
    "R.Knee.Medial", "L.Knee.Medial", "R.Shank", "L.Shank", "R.Ankle", "L.Ankle",
    "R.Ankle.Medial", "L.Ankle.Medial", "R.Heel", "L.Heel", "R.Toe", "L.Toe",
]
_MEDIAL = {"R.Knee.Medial", "L.Knee.Medial", "R.Ankle.Medial", "L.Ankle.Medial"}
_EXP_NAMES = [m for m in _MODEL_NAMES if m not in _MEDIAL]


def _write_synthetic_viewer(d: Path, n: int = 60) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t = np.linspace(16.0, 16.6, n)
    np.save(d / "time_s.npy", t)
    np.save(d / "model_markers.npy",
            rng.normal(0, 100, (n, len(_MODEL_NAMES), 3)).astype(np.float32))
    np.save(d / "experimental_markers.npy",
            rng.normal(0, 100, (n, len(_EXP_NAMES), 3)).astype(np.float32))
    np.save(d / "body_origins.npy", rng.normal(0, 100, (n, 12, 3)).astype(np.float32))
    np.save(d / "cop.npy", rng.normal(0, 50, (n, 2, 3)).astype(np.float32))
    np.save(d / "grf.npy", np.abs(rng.normal(0, 300, (n, 2, 3))).astype(np.float32))
    np.save(d / "moments.npy", rng.normal(50, 20, (n, 6)).astype(np.float32))
    meta = {
        "schema_version": "1.0.0",
        "n_frames": n,
        "frame_rate_hz": 100.0,
        "mass_kg": 80.0,
        "model_marker_names": _MODEL_NAMES,
        "experimental_marker_names": _EXP_NAMES,
        "medial_marker_names": sorted(_MEDIAL),
        "body_names": ["pelvis", "torso", "femur_r", "tibia_r", "talus_r", "calcn_r",
                       "toes_r", "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l"],
        "skeleton_segments": [["pelvis", "torso"], ["pelvis", "femur_r"]],
        "moment_names": ["hip_flexion_r", "hip_flexion_l", "knee_angle_r",
                         "knee_angle_l", "ankle_angle_r", "ankle_angle_l"],
        "moment_curve_labels": ["hip_flexion_r", "hip_flexion_l", "knee_angle_r",
                                "knee_angle_l", "ankle_angle_r", "ankle_angle_l"],
        "cop_order": ["right", "left"],
    }
    (d / "viewer_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_load_viewer_data_schema(tmp_path: Path) -> None:
    _write_synthetic_viewer(tmp_path)
    data = load_viewer_data(tmp_path)
    assert data.n_frames == 60
    assert data.model_markers.shape == (60, 19, 3)
    assert data.experimental_markers.shape == (60, 15, 3)
    assert data.body_origins.shape == (60, 12, 3)
    assert data.cop.shape == (60, 2, 3)
    assert data.grf.shape == (60, 2, 3)
    assert data.moments.shape == (60, 6)
    assert len(data.model_marker_names) == 19
    assert len(data.experimental_marker_names) == 15
    # 内侧 4 点只出现在模型 marker，绝不进实验 marker。
    assert data.medial_marker_names == _MEDIAL
    assert not (_MEDIAL & set(data.experimental_marker_names))
    assert data.mass_kg == 80.0


def test_load_viewer_data_missing_meta_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        load_viewer_data(tmp_path)


def test_viewer_widget_load_and_frame(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _write_synthetic_viewer(tmp_path)
    widget = ViewerWidget()
    widget.load(tmp_path)
    assert widget.has_data
    # 建帧不越界、游标同步。
    widget.set_frame(59)
    widget.set_frame(-5)
    assert widget.has_data
    widget.load(None)
    assert not widget.has_data
