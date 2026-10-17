from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from us_region_annotator import MainWindow, configure_font  # noqa: E402
from us_region_annotation_core import add_segment, set_label  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def synthetic_session(channels=(3,), frames=9):
    times = np.linspace(0.0, 4.0, frames)
    channel_values = tuple(channels)
    raw = np.zeros((frames, len(channel_values), 997), dtype=np.float32)
    for index in range(len(channel_values)):
        raw[:, index] = 80 + index + np.arange(frames, dtype=np.float32)[:, None]
    return SimpleNamespace(
        times=times,
        channels=channel_values,
        raw_frames=raw,
        profiles=np.abs(raw[:, :, 20:977] - 128),
        valid=np.ones((frames, len(channel_values)), dtype=bool),
        depth=np.arange(20, 977),
        raw_depth=np.arange(997),
        metadata={},
    )


def load_window(window, binary_path: Path, channels=(3,)):
    binary_path.write_bytes(b"synthetic ultrasound")
    window._loading_path = str(binary_path)
    window._loaded(synthetic_session(channels))
    window._loading_path = None
    window.show()
    QtWidgets.QApplication.processEvents()


def close_window(window):
    window._dirty = False
    window.close()
    window.deleteLater()
    QtWidgets.QApplication.processEvents()


def detail_click(window, x: float, depth: float):
    scene_point = window.detail.plotItem.vb.mapViewToScene(QtCore.QPointF(x, depth))
    detail_point = window.detail.mapFromScene(scene_point)
    viewport_point = window.detail.viewport().mapFrom(window.detail, detail_point)
    QtTest.QTest.mouseClick(
        window.detail.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=viewport_point,
    )


def create_segment(window, start=1.0, end=3.0):
    window.region.setRegion((start, end))
    QtTest.QTest.mouseClick(window.add_segment_btn, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert window.current_segment is not None
    return window.current_segment


def test_configure_font_is_safe_with_offscreen_qapplication(app):
    configure_font(app)
    assert isinstance(app.font(), QtGui.QFont)


def test_qtest_draws_sloped_cut_and_assigns_selected_block(app, tmp_path):
    window = MainWindow()
    load_window(window, tmp_path / "first.bin")
    segment = create_segment(window)
    window.line_mode_btn.click()
    detail_click(window, 2.0, 120.0)
    detail_click(window, 2.0, 300.0)
    QtTest.QTest.qWait(160)
    assert len(segment["cuts"]) == 1
    assert np.allclose(segment["cuts"][0], [120.0, 300.0], atol=18.0)

    window.line_mode_btn.click()
    detail_click(window, 1.5, 50.0)
    assert window.selected_block == 0
    window.label_combo.setCurrentIndex(window.label_combo.findData(1))
    QtTest.QTest.mouseClick(window.apply_label_btn, QtCore.Qt.MouseButton.LeftButton)
    assert segment["labels"][0] == 1
    close_window(window)


def test_save_open_roundtrip_and_npz_export_include_unknown_mask(app, tmp_path, monkeypatch):
    window = MainWindow()
    load_window(window, tmp_path / "source.bin")
    segment = create_segment(window)
    segment["cuts"] = [[100.0, 150.0]]
    segment["labels"] = [1, -1]
    window._dirty = True

    annotation_path = tmp_path / "annotation.json"
    window.annotation_path = annotation_path
    window.save_json()
    assert annotation_path.exists()
    assert json.loads(annotation_path.read_text(encoding="utf-8"))["channels"]["3"]

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(annotation_path), "JSON (*.json)"),
    )
    window.open_json()
    assert window.annotation_path == annotation_path
    assert window.document["channels"]["3"][0]["labels"] == [1, -1]

    npz_path = tmp_path / "annotation.npz"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(npz_path), "NPZ (*.npz)"),
    )
    window.export_npz()
    with np.load(npz_path, allow_pickle=False) as exported:
        assert exported["mask"].dtype == np.int8
        assert exported["mask"].shape == (9, 997)
        assert int(exported["channel"]) == 3
        assert np.any(exported["mask"] == 1)
        assert np.any(exported["mask"] == -1)
        assert json.loads(str(exported["labels"]))["3"] == "不确定"
    close_window(window)


def test_undo_reverts_qtest_drawn_cut(app, tmp_path):
    window = MainWindow()
    load_window(window, tmp_path / "undo.bin")
    segment = create_segment(window)
    window.line_mode_btn.click()
    detail_click(window, 2.0, 160.0)
    detail_click(window, 2.0, 260.0)
    QtTest.QTest.qWait(160)
    assert len(segment["cuts"]) == 1
    QtTest.QTest.mouseClick(window.undo_btn, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert len(window.document["channels"]["3"][0]["cuts"]) == 0
    close_window(window)


def test_loading_new_source_clears_old_annotation_path_and_selection(app, tmp_path):
    window = MainWindow()
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    load_window(window, first_path)
    create_segment(window)
    window.selected_block = 0
    window.annotation_path = tmp_path / "old.json"
    window._dirty = False

    second_path.write_bytes(b"a different source")
    window._loading_path = str(second_path)
    window._loaded(synthetic_session((3,)))
    window._loading_path = None
    assert window.source["binary_path"] == str(second_path.resolve())
    assert window.annotation_path is None
    assert window.current_segment is None
    assert window.selected_block is None
    assert window.segment_list.count() == 0
    assert window._overview_marks == []
    close_window(window)


def test_channel_and_segment_switch_clear_selected_block_before_labeling(app, tmp_path):
    window = MainWindow()
    load_window(window, tmp_path / "channels.bin", channels=(3, 4))
    first = add_segment(window.document, "3", 0.0, 1.0)
    second = add_segment(window.document, "3", 2.0, 3.0)
    other = add_segment(window.document, "4", 0.0, 1.0)
    window._refresh_segments()
    window.segment_list.setCurrentRow(0)
    window.selected_block = 0
    window.segment_list.setCurrentRow(1)
    assert window.current_segment is second
    assert window.selected_block is None

    window.label_combo.setCurrentIndex(window.label_combo.findData(1))
    QtTest.QTest.mouseClick(window.apply_label_btn, QtCore.Qt.MouseButton.LeftButton)
    assert second["labels"] == [-1]
    window.channel_combo.setCurrentIndex(1)
    QtWidgets.QApplication.processEvents()
    assert window.current_segment is other
    assert window.selected_block is None
    QtTest.QTest.mouseClick(window.apply_label_btn, QtCore.Qt.MouseButton.LeftButton)
    assert other["labels"] == [-1]
    assert first["labels"] == [-1]
    close_window(window)


def test_speed_selection_changes_playback_timer_interval(app, tmp_path):
    window = MainWindow()
    load_window(window, tmp_path / "speed.bin")
    initial = window.timer.interval()
    window.speed_combo.setCurrentText("2×")
    QtWidgets.QApplication.processEvents()
    twice = window.timer.interval()
    window.speed_combo.setCurrentText("0.5×")
    QtWidgets.QApplication.processEvents()
    half = window.timer.interval()
    assert twice < initial < half
    close_window(window)


def test_display_controls_preserve_annotation_state_and_cursor(app, tmp_path):
    window = MainWindow()
    load_window(window, tmp_path / "display.bin")
    segment = create_segment(window, 1.0, 3.0)
    segment["cuts"] = [[120.0, 260.0]]
    segment["labels"] = [1, -1]
    window.current_frame = 4
    window.set_frame(4)
    overview_range = ([.8, 3.2], [30.0, 700.0])
    detail_range = ([1.2, 2.4], [40.0, 500.0])
    window.overview.setRange(xRange=overview_range[0], yRange=overview_range[1], padding=0)
    window.detail.setRange(xRange=detail_range[0], yRange=detail_range[1], padding=0)
    original = window._matrix().copy()
    document = json.dumps(window.document, sort_keys=True)
    region = tuple(window.region.getRegion())
    selected_id = window.current_segment["id"]
    window.contrast_spin.setValue(2.0)
    window.brightness_spin.setValue(1.5)
    window.palette_combo.setCurrentIndex(0)
    window.mask_checkbox.setChecked(False)
    QtWidgets.QApplication.processEvents()
    assert np.array_equal(original, window._matrix(), equal_nan=True)
    assert document == json.dumps(window.document, sort_keys=True)
    assert tuple(window.region.getRegion()) == region
    assert window.current_segment["id"] == selected_id and window.current_frame == 4
    assert np.allclose(window.overview.viewRange()[0], overview_range[0], atol=.05)
    assert np.allclose(window.detail.viewRange()[0], detail_range[0], atol=.05)
    assert window.frame_line in window.overview.plotItem.items
    assert window.detail_frame_line in window.detail.plotItem.items
    assert window._last_levels is not None and window._last_lut is not None
    close_window(window)
