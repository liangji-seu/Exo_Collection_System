"""The uncertainty preview is a passive view of gate decisions only."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime.uncertainty_panel import UncertaintyPanel
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = UncertaintyPanel(window_s=10)
    widget.set_mode("ao")
    widget.resize(720, 240)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def status(mode="ao", state="allow", score=0.2, threshold=0.4, *,
           sequence=1, source_ns=1_000_000_000, generation=1):
    return SimpleNamespace(mode=mode, state=state, score=score, threshold=threshold,
                           sequence=sequence, source_grid_ns=source_ns,
                           generation=generation, age_ms=4.0)


def sample(panel, elapsed, gate_status, *, origin_ns=1_000_000_000):
    panel.add_pipeline({"time_ns": origin_ns + int(elapsed * 1e9),
                        "gate_status": gate_status})


def test_duplicate_decisions_do_not_add_points_or_break_the_line(panel):
    first = status(score=0.25, source_ns=1_000_000_000)
    sample(panel, 0, first)
    for elapsed in (0.05, 0.10, 0.15, 0.20, 0.25):
        sample(panel, elapsed, first)
    sample(panel, 0.30, status(score=0.35, sequence=2, source_ns=1_300_000_000))

    xs, ys = panel._score_curve.getData()
    np.testing.assert_allclose(xs, [0, 0.3])
    np.testing.assert_allclose(ys, [25, 35])
    assert not np.isnan(ys).any()
    assert panel._cursor_line.value() == pytest.approx(0.3)
    assert "分类器不适用分数（%，未校准）" in panel._status_label.text()


def test_late_pipeline_and_decision_timestamps_do_not_rewind_or_reorder(panel):
    sample(panel, 0, status(source_ns=1_000_000_000))
    sample(panel, 0.4, status(score=0.3, sequence=2, source_ns=1_400_000_000))
    sample(panel, 0.6, status(score=0.9, sequence=3, source_ns=1_200_000_000))
    sample(panel, 0.7, status(score=0.35, sequence=4, source_ns=1_700_000_000))
    sample(panel, 0.2, status(score=0.95, sequence=5, source_ns=1_200_000_000))

    xs, ys = panel._score_curve.getData()
    np.testing.assert_allclose(xs, [0, 0.4, 0.7])
    np.testing.assert_allclose(ys, [20, 30, 35])
    assert panel._cursor_line.value() == pytest.approx(0.7)


def test_wrap_breaks_curve_and_stale_interval_is_not_plotted(panel):
    sample(panel, 0, status(source_ns=1_000_000_000))
    sample(panel, 9.8, status(score=0.3, sequence=2, source_ns=10_800_000_000))
    sample(panel, 10.0, status(state="stale", score=0.8, sequence=3,
                               source_ns=11_000_000_000))
    assert "过期" in panel._status_label.text()
    assert "80.00%" in panel._status_label.text()
    sample(panel, 10.2, status(score=0.35, sequence=4, source_ns=11_200_000_000))

    xs, ys = panel._score_curve.getData()
    assert np.isnan(xs).sum() == 1  # wrap and stale interval share this break
    assert np.isnan(ys).sum() == 1
    np.testing.assert_allclose(xs[np.isfinite(xs)], [9.8, 0.2])
    np.testing.assert_allclose(ys[np.isfinite(ys)], [30, 35])
    assert panel._cursor_line.value() == pytest.approx(0.2)


def test_mode_reset_uses_raw_phase_variance_and_none_is_not_zero(panel):
    sample(panel, 0, status())
    np.testing.assert_allclose(panel._plot.viewRange()[1], [0, 100])
    panel.set_mode("phase_ensemble")
    assert panel._origin_ns is None and not panel._points
    assert panel._plot.getPlotItem().getAxis("left").labelText == "相位方差"
    phase = status(mode="phase_ensemble", score=0.01234, threshold=0.02,
                   source_ns=2_000_000_000)
    sample(panel, 0, phase, origin_ns=2_000_000_000)
    assert panel._score_curve.getData()[1].tolist() == [pytest.approx(0.01234)]
    assert "因果中位数相位预测方差（原始单位）" in panel._status_label.text()
    assert "0.01234" in panel._status_label.text()
    phase_range = panel._plot.viewRange()[1]
    assert phase_range[0] <= 0 <= phase_range[1]
    assert phase_range[1] >= 0.02
    assert phase_range[1] < 1  # the old AO 0–100 range must not squash variance

    panel.set_mode("none")
    sample(panel, 0.1, None, origin_ns=3_000_000_000)
    assert panel._status_label.text() == "门控：未启用"
    assert panel._points == () or len(panel._points) == 0
    assert panel._score_curve.getData()[1] is None


def test_unknown_and_missing_scores_never_become_zero(panel):
    sample(panel, 0, status(state="warming", score=None, threshold=None))
    sample(panel, 0.1, status(state="mystery", score=float("nan"), threshold=0.4,
                              sequence=2, source_ns=1_100_000_000))
    assert len(panel._points) == 0
    assert panel._score_curve.getData()[1] is None
    assert "—" in panel._status_label.text()


def test_stop_and_fault_keep_history_but_remove_live_status(panel):
    sample(panel, 0, status())
    panel.mark_inactive()
    assert len(panel._points) == 1
    assert "已停止" in panel._status_label.text()
    assert "分数 — · 阈值 —" in panel._status_label.text()
    panel.mark_inactive(fault=True)
    assert len(panel._points) == 1
    assert "故障停止" in panel._status_label.text()


def test_plot_is_fixed_and_ignores_wheel_zoom(panel):
    sample(panel, 0, status())
    before = panel._plot.viewRange()
    point = panel._plot.rect().center()
    event = QWheelEvent(QPointF(point), QPointF(panel._plot.mapToGlobal(point)), QPoint(),
                        QPoint(0, 120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.instance().sendEvent(panel._plot.viewport(), event)
    assert not event.isAccepted()
    np.testing.assert_allclose(panel._plot.viewRange(), before)
    np.testing.assert_allclose(panel._plot.viewRange()[0], [0, 10])


def test_window_registers_uncertainty_dock_and_add_menu(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    try:
        dock = window._workspace.dock_for("uncertainty")
        assert dock is not None
        assert dock.windowTitle().startswith("不确定性判定")
        assert "uncertainty" in window._workspace._default_visible
        assert any(action.objectName() == "add_preview_uncertainty"
                   for action in window._workspace._add_menu.actions())
        assert window._uncertainty_panel.parentWidget() is dock
        assert window._uncertainty_panel._plot.minimumHeight() == 150
    finally:
        window.close()
        app.processEvents()
