from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Exo_Collection_System.us_motion_compare_viewer import (
    ComparisonViewerWindow,
    _direction_style,
    _region_direction_label,
    create_smoke_core,
)


def _wait(window, predicate, timeout=4.0):
    app = QApplication.instance() or QApplication([])
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("等待对比界面状态超时")


def test_smoke_load_compute_seek_and_method_switch():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", initial_channel=2,
                                    initial_start=0, initial_end=2, auto_analyze=True)
    window.show()
    _wait(window, lambda: window.analysis is not None)
    assert window.channel_combo.currentData() == 2
    assert window.slider.maximum() > 0
    assert window.raw_curve.getData()[0].size == 997
    old = window.method_a_combo.currentData()
    window.method_a_combo.setCurrentIndex(1)
    assert old != window.method_a_combo.currentData()
    window.slider.setValue(min(5, window.slider.maximum()))
    assert window.current_index == min(5, window.slider.maximum())
    window.close(); app.processEvents()


def test_channel_change_clears_old_result():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=True)
    window.show(); _wait(window, lambda: window.analysis is not None)
    assert window.analysis is not None
    window.channel_combo.setCurrentIndex(1)
    assert window.analysis is None
    assert window.status_label.text().startswith("通道已改变")
    window.close(); app.processEvents()


def test_candidate_is_gray_context_and_screen_is_colored_only():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=True)
    window.show(); _wait(window, lambda: window.analysis is not None)
    window.show_frame(0)
    _x, pass_y = window.plot_a.direction_pass_curve.getData()
    assert np.isnan(pass_y).all()  # warmup is outside the screen mask
    window.show_frame(8)
    _x, pass_y = window.plot_a.direction_pass_curve.getData()
    assert np.isfinite(pass_y).any()
    region = {"label": "right", "candidate_label": "right", "confirmed_label": "left"}
    assert _region_direction_label(region) == "left"
    assert _direction_style(_region_direction_label(region)) == ("#2F80ED", "←")
    window.close(); app.processEvents()


def test_x_range_is_shared_and_frame_render_does_not_reset_it():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=True)
    window.show(); _wait(window, lambda: window.analysis is not None)
    window.raw_plot.setXRange(120, 420, padding=0); app.processEvents()
    expected = window.raw_plot.getViewBox().viewRange()[0]
    for view in window._x_views:
        assert np.allclose(view.viewRange()[0], expected)
    window.show_frame(min(4, window.slider.maximum())); app.processEvents()
    for view in window._x_views:
        assert np.allclose(view.viewRange()[0], expected)
    window.close(); app.processEvents()


def test_playback_speed_scales_20hz_frame_steps():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=True)
    window.show(); _wait(window, lambda: window.analysis is not None)
    for speed, ticks in ((0.25, 4), (0.5, 2), (1.0, 1)):
        window.current_index = 0; window.slider.setValue(0); window._speed = speed; window._playing = True
        for _ in range(ticks): window._play_tick()
        assert window.current_index == 1
    window._playing = False
    window.close(); app.processEvents()


def test_reset_restores_raw_amplitude_and_end_play_restarts_at_zero():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=True)
    window.show(); _wait(window, lambda: window.analysis is not None)
    original = window.raw_plot.getViewBox().viewRange()[1]
    window.raw_plot.setYRange(-10, 10, padding=0)
    window.reset_views()
    assert np.allclose(window.raw_plot.getViewBox().viewRange()[1], original)
    window.slider.setValue(window.slider.maximum())
    window.current_index = window.slider.maximum()
    window.toggle_play()
    assert window.current_index == 0
    window._playing = False
    window.close(); app.processEvents()


def test_cancel_invalidates_late_result_and_allows_recompute():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=False)
    window.show(); _wait(window, lambda: window.session is not None)
    window._invalidate("正在计算…")
    old_generation = window._generation
    window.compute_button.setEnabled(False); window.cancel_button.setEnabled(True)
    window.cancel_work()
    assert window._generation == old_generation + 1
    late = {"times": np.array([0.0]), "raw": np.zeros((1, 997)), "methods": {}}
    window._on_analyzed(old_generation, late)
    assert window.analysis is None
    window.start_analysis(); _wait(window, lambda: window.analysis is not None)
    window.close(); app.processEvents()


def test_time_range_change_clears_result_while_running_or_after_result():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=True)
    window.show(); _wait(window, lambda: window.analysis is not None)
    old_generation = window._generation
    window.start_spin.setValue(0.5)
    assert window.analysis is None
    assert window._generation > old_generation
    assert window.compute_button.isEnabled()
    window.close(); app.processEvents()


def test_playback_bar_is_visible_above_raw_plot_at_1280x720():
    app = QApplication.instance() or QApplication([])
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", auto_analyze=False)
    window.resize(1280, 720); window.show(); app.processEvents()
    assert window.play_button.isVisible()
    assert window.prev_button.isVisible() and window.next_button.isVisible()
    assert window.play_button.geometry().bottom() < window.raw_plot.geometry().top()
    assert window.play_button.geometry().bottom() <= window.height()
    window.close(); app.processEvents()
