from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from us_motion_viewer import MotionViewerWindow, create_smoke_core


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout: float = 3.0) -> None:
    app = _app()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("等待界面状态超时")


def test_player_loads_and_interactions_reset_playback() -> None:
    app = _app()
    window = MotionViewerWindow(
        create_smoke_core(), initial_path="smoke", initial_channel=2, initial_time=2.0
    )
    window.show()
    _wait_until(lambda: window.current_result is not None)

    assert window.selected_channel() == 2
    assert abs(float(window.current_result["current_time"]) - 2.0) < 0.051
    assert window.lag_combo.currentData() == 0.20
    assert len(window.current_result["blocks"]) == 20

    window.show_frame(0, pause=True)
    _wait_until(lambda: window.current_result is not None)
    assert "分析区未知比例 100.0%" in window.metrics.text()

    window.toggle_playing()
    assert window._playing
    window.slider.setValue(60)
    assert not window._playing
    _wait_until(lambda: window.current_index == 60 and window.current_result is not None)

    window.toggle_playing()
    window.channel_combo.setCurrentIndex(2)
    assert not window._playing
    _wait_until(lambda: window.selected_channel() == 3 and window.current_result is not None)

    window.show_frame(window.current_index + 1, pause=True)
    _wait_until(lambda: window._active_serial is None)
    window.close()
    app.processEvents()
    assert window._closing


def test_band_depth_mapping_tracks_plot_viewbox() -> None:
    app = _app()
    window = MotionViewerWindow(create_smoke_core(), initial_path="smoke")
    window.resize(1180, 760)
    window.show()
    _wait_until(lambda: window.current_result is not None)
    window.plot.setXRange(100.0, 900.0, padding=0.0)
    app.processEvents()
    window._sync_band_axis()

    view_box = window.plot.getViewBox()
    for depth in (200.0, 800.0):
        scene_point = view_box.mapViewToScene(QPointF(depth, 0.0))
        plot_pixel = window.plot.mapFromScene(scene_point).x()
        assert abs(window.band.pixel_for_depth(depth) - plot_pixel) <= 1.5
    window.close()


def test_plot_uses_independent_raw_adc_frame_and_band_is_below_plot() -> None:
    app = _app()
    window = MotionViewerWindow(create_smoke_core(), initial_path="smoke", initial_channel=2)
    window.resize(1180, 760)
    window.show()
    _wait_until(lambda: window.current_result is not None)

    raw_x, raw_y = window.current_curve.getData()
    result = window.current_result
    assert np.array_equal(np.asarray(raw_x), np.asarray(result["raw_depth"], dtype=float))
    assert np.array_equal(np.asarray(raw_y), np.asarray(result["raw_current"], dtype=float))
    assert np.asarray(raw_x).size == 997
    assert np.asarray(raw_y).min() >= 0 and np.asarray(raw_y).max() <= 255
    x_range, y_range = window.plot.getViewBox().viewRange()
    assert np.allclose(x_range, (0.0, 996.0), atol=1e-6)
    assert np.allclose(y_range, (0.0, 255.0), atol=1e-6)
    assert window.band.y() > window.plot.y()
    window.close()


def test_band_keeps_analysis_depth_coordinate_with_raw_plot() -> None:
    app = _app()
    window = MotionViewerWindow(create_smoke_core(), initial_path="smoke")
    window.show()
    _wait_until(lambda: window.current_result is not None)

    result = dict(window.current_result)
    result["depth"] = np.arange(957, dtype=float) + 20.0
    result["blocks"] = [{"start_index": 86, "stop_index": 135, "label": "left"}]
    window._render_result(result)

    assert window.band._depth[86] == 106.0
    assert window.band._depth[134] == 154.0
    window.close()


def test_reload_discards_stale_worker_result() -> None:
    class OrderedCore:
        def __init__(self) -> None:
            self.base = create_smoke_core()

        def load_session(self, path, progress=None, cancel=None):
            if str(path).endswith("slow"):
                time.sleep(0.12)
            result = self.base.load_session(path, progress=progress, cancel=cancel)
            result.metadata["source"] = str(path)
            return result

        def estimate_frame(self, *args, **kwargs):
            return self.base.estimate_frame(*args, **kwargs)

    app = _app()
    window = MotionViewerWindow(OrderedCore())
    window.load_path("slow")
    window.load_path("new")
    _wait_until(lambda: window.session is not None and window.current_result is not None)
    time.sleep(0.15)
    app.processEvents()

    assert window.session.metadata["source"].endswith("new")
    window.close()


def test_export_screenshot_and_csv(tmp_path: Path) -> None:
    app = _app()
    window = MotionViewerWindow(create_smoke_core(), initial_path="smoke", initial_time=1.0)
    window.show()
    _wait_until(lambda: window.current_result is not None)

    image = window.export_screenshot(tmp_path / "frame")
    table = window.export_csv(tmp_path / "frame")

    assert image == tmp_path / "frame.png"
    assert image.is_file() and image.stat().st_size > 1000
    assert table == tmp_path / "frame.csv"
    lines = table.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == (
        "session_path,channel,current_time,past_time,lag_seconds,depth,current,past,"
        "direction,screen_pass,support,region"
    )
    assert len(lines) == 958
    assert any(",unknown" in line for line in lines[1:])
    window.close()
    app.processEvents()


def test_slow_estimate_cannot_overwrite_seek_or_drop_last_frame() -> None:
    class SlowCore:
        def __init__(self) -> None:
            self.base = create_smoke_core()

        def load_session(self, *args, **kwargs):
            return self.base.load_session(*args, **kwargs)

        def estimate_frame(self, *args, **kwargs):
            time.sleep(0.08)
            return self.base.estimate_frame(*args, **kwargs)

    app = _app()
    window = MotionViewerWindow(SlowCore(), initial_path="smoke")
    window.show()
    _wait_until(lambda: window.current_result is not None)

    window.show_frame(10)
    window.show_frame(60, pause=True)
    _wait_until(
        lambda: window.current_result is not None
        and abs(float(window.current_result["current_time"]) - 3.0) < 0.001,
        timeout=4.0,
    )

    last = window.slider.maximum()
    window.show_frame(last - 1, pause=True)
    _wait_until(lambda: window._active_serial is None)
    window.speed_combo.setCurrentIndex(window.speed_combo.findData(4.0))
    window.toggle_playing()
    _wait_until(lambda: not window._playing and window.current_index == last)
    _wait_until(
        lambda: window.current_result is not None
        and abs(float(window.current_result["current_time"]) - 8.95) < 0.001,
        timeout=4.0,
    )
    window.close()
    app.processEvents()


def test_slow_estimate_playback_publishes_recent_completed_frames() -> None:
    class SlowCore:
        def __init__(self) -> None:
            self.base = create_smoke_core()

        def load_session(self, *args, **kwargs):
            return self.base.load_session(*args, **kwargs)

        def estimate_frame(self, *args, **kwargs):
            time.sleep(0.08)
            return self.base.estimate_frame(*args, **kwargs)

    app = _app()
    window = MotionViewerWindow(SlowCore(), initial_path="smoke", initial_time=1.0)
    window.show()
    _wait_until(lambda: window.current_result is not None)
    initial_time = float(window.current_result["current_time"])
    window.toggle_playing()

    deadline = time.perf_counter() + 0.6
    advanced_during_play = False
    while time.perf_counter() < deadline:
        app.processEvents()
        if (
            window._playing
            and window.current_result is not None
            and float(window.current_result["current_time"]) > initial_time
        ):
            advanced_during_play = True
            break
        time.sleep(0.005)

    assert advanced_during_play
    window.toggle_playing()
    window.close()
    app.processEvents()
