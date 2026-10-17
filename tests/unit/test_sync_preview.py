"""Regression tests for the IMU/mocap synchronization preview."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402

from exo_collection.apps.calculate.sync_view import SyncView  # noqa: E402
from exo_collection.apps.calculate.sync_view import _ManualPair  # noqa: E402
from exo_collection.apps.calculate.workers import (  # noqa: E402
    SyncDataBundle,
    _marker_acceleration,
)


def test_marker_acceleration_is_gap_safe_and_keeps_units() -> None:
    samples = 32
    points = np.column_stack(
        (
            (np.arange(samples, dtype=np.float64) + 1.0) ** 2,
            np.zeros(samples),
            np.zeros(samples),
        )
    )
    residuals = np.ones(samples)
    points[10] = 0.0
    points[20, 1] = np.nan
    residuals[26] = -1.0

    acceleration, valid = _marker_acceleration(points, residuals, 100.0)

    assert np.flatnonzero(~valid).tolist() == [10, 20, 26]
    # x=i² has 2*100² mm/s² acceleration in every interior valid sample.
    finite = np.isfinite(acceleration).all(axis=1)
    assert np.nanmax(acceleration[:, 0]) == pytest.approx(20_000.0)
    assert not finite[10] and not finite[20] and not finite[26]
    # Two samples at each valid-run edge are intentionally undefined.
    assert not finite[8] and not finite[9] and not finite[11] and not finite[12]
    assert not finite[0] and not finite[1] and finite[2]


def _bundle(*, marker: bool = True) -> SyncDataBundle:
    n = 32
    time = np.arange(n, dtype=np.float64) / 100.0
    marker_acc = np.full(n, np.nan)
    marker_acc[2:-2] = 1.0
    return SyncDataBundle(
        c3d_t0_host_ns=0,
        c3d_h5_match_rms_mm=0.0,
        c3d_h5_start_frame=0,
        imu_sensor_label="imu_right_leg",
        imu_time_s=time,
        imu_acc=np.ones((n, 3)),
        imu_acc_norm=np.ones(n),
        imu_envelope=np.ones(n),
        gaitway_time_s=time,
        gaitway_total_fz=np.ones(n),
        gaitway_fz_left=np.ones(n),
        gaitway_fz_right=np.ones(n),
        marker_time_s=time if marker else None,
        marker_acc_norm=marker_acc if marker else None,
        marker_name="R.Ankle" if marker else None,
        marker_valid=np.ones(n, dtype=bool) if marker else None,
        marker_invalid_count=0,
    )


def test_sync_view_labels_imu_and_hides_marker_by_default() -> None:
    QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        view.set_manual_data(_bundle())
        names = {item.name() for item in view._imu_plot.listDataItems() if item.name()}
        assert {"IMU X", "IMU Y", "IMU Z", "IMU 冲击包络"} <= names
        assert "动捕 R.Ankle 辅助" not in names
        assert view._marker_aux_checkbox.isEnabled()
        assert "C3D 无效帧 0" in view._marker_status.text()

        view._marker_aux_checkbox.setChecked(True)
        names = {item.name() for item in view._imu_plot.listDataItems() if item.name()}
        assert "动捕 R.Ankle 辅助" in names
    finally:
        view.close()


def test_sync_view_links_only_mapped_c3d_time_axis() -> None:
    QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        view.set_manual_data(_bundle())
        imu_vb = view._imu_plot.plotItem.vb
        gait_vb = view._gaitway_plot.plotItem.vb
        assert gait_vb.linkedView(gait_vb.XAxis) is imu_vb

        view._axis_combo.setCurrentIndex(1)
        assert gait_vb.linkedView(gait_vb.XAxis) is None
        view._axis_combo.setCurrentIndex(0)
        assert gait_vb.linkedView(gait_vb.XAxis) is imu_vb
    finally:
        view.close()


def _force_bundle() -> SyncDataBundle:
    bundle = _bundle()
    phase = np.linspace(0, 4 * np.pi, len(bundle.gaitway_time_s))
    return replace(bundle, gaitway_total_fz=800 + 100 * np.sin(phase),
                   gaitway_fz_left=400 + 300 * np.sin(phase),
                   gaitway_fz_right=400 - 300 * np.sin(phase))


@pytest.mark.parametrize("load_before_show", [False, True])
def test_first_load_fits_all_finite_signals_without_any_user_interaction(load_before_show: bool) -> None:
    app = QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        view.resize(1200, 900)
        bundle = _force_bundle()
        if load_before_show:
            # Async data may arrive while the manual page is still hidden.
            view.set_manual_data(bundle)
        view._stack.setCurrentIndex(1)
        view.show()
        app.processEvents()
        if not load_before_show:
            view.set_manual_data(bundle)
        app.processEvents()
        low, high = view._gaitway_plot.viewRange()[1]
        values = np.concatenate([bundle.gaitway_total_fz, bundle.gaitway_fz_left, bundle.gaitway_fz_right])
        assert low < values.min()
        assert high > values.max()
        assert view._gaitway_plot.getViewBox().autoRangeEnabled()[1]
        assert len(view._gaitway_plot.listDataItems()) >= 3
        assert view._gaitway_plot.viewRange()[0][0] <= bundle.gaitway_time_s[0]
        assert view._gaitway_plot.viewRange()[0][1] >= bundle.gaitway_time_s[-1]
    finally:
        view.close()


def test_new_trial_and_fit_button_recover_stale_narrow_force_range() -> None:
    app = QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        bundle = _force_bundle()
        view.set_manual_data(bundle)
        view._gaitway_plot.setYRange(337.0, 338.5, padding=0)
        view._imu_plot.setYRange(.9, 1.1, padding=0)
        view._offset_spin.setValue(7.1431)
        view._pairs.append(_ManualPair(100_000, 100_007.1431))
        view._replot()
        app.processEvents()
        np.testing.assert_allclose(view._gaitway_plot.viewRange()[1], [337, 338.5])

        view._fit_plots_button.click()
        app.processEvents()
        lower, upper = view._gaitway_plot.viewRange()[1]
        assert lower < bundle.gaitway_fz_left.min()
        assert upper > bundle.gaitway_total_fz.max()
        # Annotation at y=0/t=100000 must not enlarge fitted data bounds.
        assert lower > 0
        assert view._imu_plot.viewRange()[0][1] < 10
        assert len(view._pairs) == 1
        assert view._current_offset_s == pytest.approx(7.1431)
        assert view._gaitway_plot.getViewBox().autoRangeEnabled()[1]

        view._gaitway_plot.setYRange(337, 338.5, padding=0)
        view._selected_imu_time = 123
        view.set_manual_data(bundle)
        app.processEvents()
        assert view._gaitway_plot.viewRange()[1][1] > bundle.gaitway_total_fz.max()
        assert not view._pairs
        assert view._selected_imu_time is None
        assert "0 对" in view._pairs_label.text()
    finally:
        view.close()


def test_offset_and_auxiliary_redraw_preserve_intentional_zoom() -> None:
    app = QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        view.set_manual_data(_force_bundle())
        view._imu_plot.setXRange(.1, .2, padding=0)
        view._imu_plot.setYRange(-2, 2, padding=0)
        view._gaitway_plot.setYRange(200, 700, padding=0)
        app.processEvents()
        before_imu = np.array(view._imu_plot.viewRange())
        before_force = np.array(view._gaitway_plot.viewRange())
        view._offset_spin.setValue(.01)
        view._marker_aux_checkbox.setChecked(True)
        app.processEvents()
        np.testing.assert_allclose(view._imu_plot.viewRange(), before_imu)
        np.testing.assert_allclose(view._gaitway_plot.viewRange(), before_force)
        gait = view._gaitway_plot.getViewBox()
        assert gait.linkedView(gait.XAxis) is view._imu_plot.getViewBox()
    finally:
        view.close()


@pytest.mark.parametrize("value", [np.nan, 0.0, 338.0])
def test_fit_handles_nan_or_constant_force(value: float) -> None:
    app = QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        bundle = _bundle()
        values = np.full(len(bundle.gaitway_time_s), value)
        view.set_manual_data(replace(bundle, gaitway_total_fz=values,
                                     gaitway_fz_left=values, gaitway_fz_right=values))
        app.processEvents()
        low, high = view._gaitway_plot.viewRange()[1]
        assert np.isfinite([low, high]).all()
        assert high > low
        if np.isfinite(value):
            assert low <= value <= high
    finally:
        view.close()


def test_peak_selection_ignores_other_buttons_axes_and_accepted_controls() -> None:
    app = QApplication.instance() or QApplication([])
    view = SyncView()
    try:
        view.resize(1200, 900)
        view._stack.setCurrentIndex(1)
        view.show()
        view.set_manual_data(_force_bundle())
        app.processEvents()

        class Click:
            def __init__(self, pos, button=Qt.MouseButton.LeftButton, accepted=False):
                self.pos, self.mouse_button, self.accepted = pos, button, accepted
            def scenePos(self):
                return self.pos
            def button(self):
                return self.mouse_button
            def isAccepted(self):
                return self.accepted

        ivb, gvb = view._imu_plot.getViewBox(), view._gaitway_plot.getViewBox()
        ipos = ivb.mapViewToScene(QPointF(.1, 1))
        for event in (Click(ipos, Qt.MouseButton.RightButton), Click(QPointF(-50, -50)), Click(ipos, accepted=True)):
            view._on_imu_click(event)
            assert view._selected_imu_time is None
        view._on_imu_click(Click(ipos))
        assert view._selected_imu_time == pytest.approx(.1)
        view._offset_spin.setValue(.05)
        gpos = gvb.mapViewToScene(QPointF(.12, 400))
        view._on_gaitway_click(Click(gpos, Qt.MouseButton.RightButton))
        assert not view._pairs
        view._on_gaitway_click(Click(gpos))
        assert view._pairs[0].gaitway_time_s == pytest.approx(.17)
        assert view._current_offset_s == pytest.approx(.07)
    finally:
        view.close()
