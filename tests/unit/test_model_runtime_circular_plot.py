"""One shared clock and a fixed circular plot, without ports or hardware."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector.signal_previews import HoverDetailsPlotWidget, NoWheelViewBox
from exo_collection.apps.model_runtime.preview_panels import ModelOutputPanel
from exo_collection.apps.model_runtime.spec import OutputSpec


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = ModelOutputPanel(window_s=10)
    widget.resize(1100, 550)
    widget.set_output_spec(OutputSpec(unit="Nm/kg"), active_sides=("right",))
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def sample(panel, elapsed, *, raw=0.1, desired=1.2):
    panel.add_pipeline({"time_ns": int(1_000_000_000 + elapsed * 1e9),
                        "unit": "Nm/kg", "raw_estimate_right": raw,
                        "estimate_nm": raw * 80, "estimate_left_nm": 0,
                        "raw_estimate_left": 0, "desired_right_nm": desired,
                        "desired_left_nm": 0, "assistance_errors": []})


def wheel_event(widget):
    point = widget.rect().center()
    return QWheelEvent(QPointF(point), QPointF(widget.mapToGlobal(point)), QPoint(),
                       QPoint(0, 120), Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)


def test_one_plot_two_axes_only_real_active_side(panel):
    sample(panel, 0)
    panel.add_prediction(1_000_000_000, 0, 6)
    assert len(panel.findChildren(HoverDetailsPlotWidget)) == 1
    assert panel._normalized_plot is panel._plot
    assert panel._raw_left_curve.isVisible() is False
    assert not panel._raw_left_y
    assert set(panel._traces) == {"期望辅助右 (Nm)"}
    assert "左" not in panel._values_label.text()
    item = panel._plot.getPlotItem()
    assert "Nm" in item.getAxis("left").labelText
    assert "Nm/kg" not in item.getAxis("left").labelText
    assert "Nm/kg" not in item.getAxis("right").labelText
    assert "Nm" in item.getAxis("right").labelText
    assert panel._traces["期望辅助右 (Nm)"]["curve"].getViewBox() is panel._assistance_view
    panel._plot._set_hover_details_visible(False)
    assert item.legend.isVisible()
    assert item.getAxis("left").label.isVisible()
    assert item.getAxis("right").label.isVisible()


def test_circular_time_axis_breaks_at_wrap_and_drops_old_cycles(panel):
    sample(panel, 0)
    sample(panel, 9.8)
    sample(panel, 10.2)
    xs, ys = panel._normalized_curve.getData()
    np.testing.assert_allclose(xs[np.isfinite(xs)], [9.8, 0.2], atol=1e-7)
    assert np.isnan(ys).sum() == 1
    assert panel._cursor_line.value() == pytest.approx(0.2)
    np.testing.assert_allclose(panel._plot.viewRange()[0], [0, 10])
    # Late hardware polling must use the same origin and never rewind cursor.
    panel.add_command(SimpleNamespace(host_monotonic_ns=11_100_000_000, left_nm=0,
                                      right_nm=0.8, enabled=True, torque_limit_nm=2))
    assert panel._traces["指令右 (Nm)"]["xs"][-1] == pytest.approx(0.1)
    assert panel._cursor_line.value() == pytest.approx(0.2)
    sample(panel, 35)
    np.testing.assert_allclose(panel._normalized_curve.getData()[0], [5])
    assert not panel._traces["指令右 (Nm)"]["ys"]
    assert panel._cursor_line.value() == pytest.approx(5)
    assert panel._plot.getPlotItem().vb.state["autoRange"][0] is False


def test_missing_assistance_is_not_zero_and_model_reset_clears_all(panel):
    sample(panel, 0, desired=None)
    assert not panel._traces
    assert list(panel._normalized_y) == [8]
    sample(panel, 0.5)
    panel.set_output_spec(OutputSpec(unit="Nm"))
    assert panel._t0_ns is None and not panel._traces
    assert panel._cursor_line.value() == 0
    assert not panel._normalized_y
    assert "Nm/kg" not in panel._plot.getPlotItem().getAxis("left").labelText
    panel.add_pipeline({"time_ns": 100_000_000_000, "unit": "Nm",
                        "raw_estimate_right": 4, "raw_estimate_left": 2,
                        "estimate_nm": 4, "estimate_left_nm": 2,
                        "desired_right_nm": 2, "desired_left_nm": 1})
    assert panel._raw_left_curve.isVisible()
    assert set(panel._traces) == {"期望辅助右 (Nm)", "期望辅助左 (Nm)"}
    assert list(panel._normalized_x) == [0]
    panel.clear()
    assert not panel._normalized_y and not panel._raw_left_y and not panel._traces


def test_wheel_changes_neither_model_view_or_auxiliary_axis(panel):
    sample(panel, 0)
    sample(panel, 1)
    app = QApplication.instance()
    app.processEvents()
    before = [list(map(list, box.viewRange())) for box in
              (panel._plot.getViewBox(), panel._assistance_view)]
    event = wheel_event(panel._plot.viewport())
    app.sendEvent(panel._plot.viewport(), event)
    assert not event.isAccepted()
    for box, expected in zip((panel._plot.getViewBox(), panel._assistance_view), before):
        assert isinstance(box, NoWheelViewBox)
        box.wheelEvent(event, axis=1)
        np.testing.assert_allclose(box.viewRange(), expected)


def test_shared_sensor_plots_ignore_wheel_and_keep_hover():
    app = QApplication.instance() or QApplication([])
    # Same class constructs ultrasound, IMU, encoder and EMG plots in both apps.
    plot = HoverDetailsPlotWidget()
    try:
        plot.setTitle("IMU")
        plot.setLabel("left", "加速度")
        plot.resize(500, 300)
        plot.show()
        plot.setRange(xRange=(0, 999), yRange=(-10, 10), padding=0)
        app.processEvents()
        before = plot.viewRange()
        event = wheel_event(plot.viewport())
        app.sendEvent(plot.viewport(), event)
        assert not event.isAccepted()
        np.testing.assert_allclose(plot.viewRange(), before)
        plot._set_hover_details_visible(True)
        assert plot.getPlotItem().titleLabel.isVisible()
        plot._set_hover_details_visible(False)
        assert not plot.getPlotItem().titleLabel.isVisible()
    finally:
        plot.close()


def test_bilateral_six_entry_legend_stays_outside_waveform(panel):
    panel.set_output_spec(OutputSpec(unit="Nm"))
    panel.add_pipeline({"time_ns": 1_000_000_000, "unit": "Nm",
                        "raw_estimate_left": 2, "raw_estimate_right": 4,
                        "estimate_left_nm": 2, "estimate_nm": 4,
                        "desired_left_nm": 1, "desired_right_nm": 2})
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_100_000_000, left_nm=.8,
                                      right_nm=1.6, enabled=True, torque_limit_nm=3))
    for _ in range(12):
        QApplication.instance().processEvents()
    item = panel._plot.getPlotItem()
    assert len(item.legend.items) == 6
    assert item.legend.sceneBoundingRect().bottom() <= item.vb.sceneBoundingRect().top() + 1
    assert item.vb.sceneBoundingRect().height() >= 180


def test_nm_prediction_uses_mass_conversion_once_and_shared_y_scale(panel):
    # Core conversion: .3 Nm/kg * 80 kg = 24 Nm. Assistance additionally
    # applies .4 gain = 9.6 Nm. The display must use each supplied value once.
    panel.add_pipeline({"time_ns": 1_000_000_000, "unit": "Nm/kg",
                        "raw_estimate_right": .3, "raw_estimate_left": 0,
                        "estimate_nm": 24, "estimate_left_nm": 0,
                        "desired_right_nm": 9.6, "desired_left_nm": 0})
    for _ in range(10):
        QApplication.instance().processEvents()
    assert list(panel._normalized_y) == [24]
    assert panel._traces["期望辅助右 (Nm)"]["ys"][-1] == 9.6
    assert "+24.000 Nm" in panel._values_label.text()
    assert "+9.600 Nm" in panel._values_label.text()
    item = panel._plot.getPlotItem()
    assert panel._assistance_view is item.vb
    np.testing.assert_allclose(item.getAxis("left").range, item.getAxis("right").range)
    assert item.vb.viewRange()[1][0] <= 9.6 < 24 <= item.vb.viewRange()[1][1]
    assert "Nm/kg" not in panel._values_label.text()
    assert any(label.text == "模型预测右 (Nm)" for _, label in item.legend.items)


def test_no_mass_does_not_relabel_raw_as_nm_and_native_nm_is_not_rescaled(panel):
    panel.add_pipeline({"time_ns": 1_000_000_000, "unit": "Nm/kg",
                        "raw_estimate_right": .3, "estimate_nm": None,
                        "estimate_left_nm": 0, "desired_right_nm": None,
                        "assistance_errors": ["待填写体重"]})
    assert not panel._normalized_y and not panel._traces
    assert "待填写体重 / 不可换算" in panel._values_label.text()
    assert "0.300" not in panel._values_label.text()
    panel.set_output_spec(OutputSpec(unit="Nm"), active_sides=("right",))
    panel.add_pipeline({"time_ns": 2_000_000_000, "unit": "Nm",
                        "raw_estimate_right": 3, "estimate_nm": 3,
                        "desired_right_nm": 1.2, "mass_kg": 80})
    assert list(panel._normalized_y) == [3]
    assert "+3.000 Nm" in panel._values_label.text()
    panel.clear()
    assert not panel._normalized_y and not panel._traces


def test_real_send_survives_pipeline_status_and_feedback_refreshes(panel):
    payload = {"time_ns": 1_000_000_000, "unit": "Nm/kg", "raw_estimate_right": -.8,
               "estimate_nm": -64, "estimate_left_nm": 0,
               "desired_right_nm": -25.6, "desired_left_nm": 0}
    panel.add_pipeline(payload)
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_010_000_000, left_nm=0,
                                      right_nm=-9, enabled=True, torque_limit_nm=9))
    for index in range(3):
        panel.add_pipeline({**payload, "time_ns": 1_050_000_000 + index * 50_000_000})
        panel.set_output_status("底层已确认使能")
        panel.set_feedback(SimpleNamespace(state=2, error=0, left_position_rad=0,
            left_velocity_rad_s=0, left_torque_nm=0, right_position_rad=0,
            right_velocity_rad_s=0, right_torque_nm=-8.7), .01)
        text = panel._values_label.text()
        assert "期望辅助（限幅前） 右 -25.600 Nm" in text
        assert "最近实际下发（限幅后） 右 -9.000 Nm" in text
        assert "该帧阈值 ±9 Nm ｜使能" in text
    assert panel._normalized_y[-1] == -64  # Neither prediction nor desired clipped.
    assert panel._traces["期望辅助右 (Nm)"]["ys"][-1] == -25.6
    labels = [label.text for _, label in panel._plot.getPlotItem().legend.items]
    assert "期望右 (Nm，限幅前)" in labels and "下发右 (Nm，限幅后)" in labels
    # A newer desired value must not reinterpret an older command as evidence
    # of whether that command was clipped. Only its actual value is retained.
    panel.add_pipeline({**payload, "time_ns": 1_250_000_000, "desired_right_nm": -3})
    assert "最近实际下发（限幅后） 右 -9.000 Nm" in panel._values_label.text()
    assert "限幅中" not in panel._values_label.text()
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_260_000_000, left_nm=0,
                                      right_nm=0, enabled=False, torque_limit_nm=9))
    panel.set_output_status("预测已停止 · 零力矩并失能")
    assert "最近实际下发（限幅后） 右 +0.000 Nm" in panel._values_label.text()
    assert "该帧阈值 ±9 Nm ｜失能" in panel._values_label.text()
    assert "右 -9.000 Nm" not in panel._values_label.text()


@pytest.mark.parametrize("reset", ["clear", "model_switch", "disconnect"])
def test_command_reset_forgets_old_send_and_bilateral_values_are_explicit(panel, reset):
    panel.set_output_spec(OutputSpec(unit="Nm"))
    panel.add_pipeline({"time_ns": 1_000_000_000, "unit": "Nm",
                        "estimate_left_nm": 4, "estimate_nm": -6,
                        "desired_left_nm": 2, "desired_right_nm": -3})
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_010_000_000, left_nm=2,
                                      right_nm=-3, enabled=True, torque_limit_nm=9))
    assert "左 +2.000 / 右 -3.000 Nm" in panel._values_label.text()
    if reset == "clear":
        panel.clear()
    elif reset == "model_switch":
        panel.set_output_spec(OutputSpec(unit="Nm"), active_sides=("right",))
    else:
        panel.clear_command()
        assert panel._traces["期望辅助右 (Nm)"]["ys"]
        assert panel._normalized_y
    panel.set_output_status("电机未连接 · 未下发")
    panel.set_feedback(None)
    assert "最近实际下发" not in panel._values_label.text()
    assert "该帧阈值" not in panel._values_label.text()
    assert not any(key.startswith("指令") for key in panel._traces)


def test_shared_right_estimate_displays_two_motor_values_without_left_prediction(panel):
    panel.set_output_spec(OutputSpec(unit="Nm/kg"), active_sides=("right",),
                          motor_distribution="right_to_both_same_sign")
    payload = {"time_ns": 1_000_000_000, "unit": "Nm/kg", "raw_estimate_right": .3,
               "estimate_nm": 24, "estimate_left_nm": 0,
               "desired_right_nm": 9.6, "desired_left_nm": 9.6,
               "motor_distribution": "right_to_both_same_sign"}
    panel.add_pipeline(payload)
    assert list(panel._normalized_y) == [24] and not panel._raw_left_y
    assert not panel._raw_left_curve.isVisible()
    assert "模型预测 右 +24.000 Nm" in panel._values_label.text()
    assert "左 +9.600 / 右 +9.600 Nm" in panel._values_label.text()
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_010_000_000, left_nm=9,
                                      right_nm=9, enabled=True, torque_limit_nm=9))
    panel.add_pipeline({**payload, "time_ns": 1_050_000_000})
    panel.set_output_status("底层已确认使能")
    assert "最近实际下发（限幅后） 左 +9.000 / 右 +9.000 Nm" in panel._values_label.text()
    assert "该帧阈值 ±9 Nm ｜使能" in panel._values_label.text()
    assert set(panel._traces) == {"期望辅助右 (Nm)", "指令右 (Nm)"}
    labels = [label.text for _, label in panel._plot.getPlotItem().legend.items]
    assert len(labels) == 3 and not any("模型预测左" in label for label in labels)
    assert any("期望左右同步(同号)" in label for label in labels)
    assert any("下发左右同步(同号)" in label for label in labels)
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_100_000_000, left_nm=0,
                                      right_nm=0, enabled=False, torque_limit_nm=9))
    panel.set_output_status("故障归零失能")
    assert "最近实际下发（限幅后） 左 +0.000 / 右 +0.000 Nm" in panel._values_label.text()
    assert "±9 Nm ｜失能" in panel._values_label.text()
    panel.clear()
    assert panel._motor_distribution == "right_to_both_same_sign"
    assert not panel._normalized_y and not panel._traces
    panel.set_output_spec(OutputSpec(unit="Nm"), active_sides=("right",))
    panel.add_pipeline({"time_ns": 2_000_000_000, "unit": "Nm", "estimate_nm": 4,
                        "desired_right_nm": 2, "desired_left_nm": 0,
                        "motor_distribution": "per_side"})
    assert panel._motor_distribution == "per_side"
    assert "期望辅助（限幅前） 右 +2.000 Nm" in panel._values_label.text()
    assert "最近实际下发" not in panel._values_label.text()
    assert not any("同步" in label.text for _, label in panel._plot.getPlotItem().legend.items)


def test_shared_mode_does_not_mislabel_unequal_actual_motor_frames_as_synchronized(panel):
    panel.set_output_spec(OutputSpec(unit="Nm"), active_sides=("right",),
                          motor_distribution="right_to_both_same_sign")
    panel.add_command(SimpleNamespace(host_monotonic_ns=1_000_000_000, left_nm=8,
                                      right_nm=9, enabled=True, torque_limit_nm=9))
    assert "左 +8.000 / 右 +9.000 Nm" in panel._values_label.text()
    assert set(panel._traces) == {"指令左 (Nm)", "指令右 (Nm)"}
    assert not any("下发左右同步" in label.text for _, label in panel._plot.getPlotItem().legend.items)
