"""Regression tests for the IMU/mocap synchronization preview."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from exo_collection.apps.calculate.sync_view import SyncView  # noqa: E402
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
