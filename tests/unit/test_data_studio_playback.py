from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTabWidget, QWidget
from PySide6.QtTest import QTest

from exo_collection.apps.data_studio.local_dialogs import (
    PlaybackDialog,
    _SweepSignalPlot,
    _SweepWaterfallPlot,
)
from exo_collection.apps.data_studio.local_tools import (
    SignalPlayback,
    TrialPlayback,
    UltrasoundPlayback,
)


def _complete_playback() -> TrialPlayback:
    time_s = np.linspace(0.0, 12.0, 121)
    base_channels = (
        "acc_x", "acc_y", "acc_z",
        "gyr_x", "gyr_y", "gyr_z",
        "mag_x", "mag_y", "mag_z",
        "roll", "pitch", "yaw",
    )
    imu_channels = tuple(
        f"{sensor}:{channel}"
        for sensor in range(1, 4)
        for channel in base_channels
    )
    return TrialPlayback(
        manifest_path=Path("manifest.json"),
        trial_uuid="00000000-0000-0000-0000-000000000001",
        condition_code="WALK_LEVEL",
        formal_t0_host_monotonic_ns=1,
        ultrasound=UltrasoundPlayback(
            time_s=time_s,
            waterfall=np.ones((4, time_s.size, 1000), dtype=np.uint8),
            latest_frame=np.ones((4, 1000), dtype=np.uint8),
            channels=("ch_1", "ch_2", "ch_3", "ch_4"),
            source_frame_count=time_s.size,
        ),
        imu=SignalPlayback(
            time_s=time_s,
            values=np.zeros((time_s.size, 36), dtype=np.float32),
            channels=imu_channels,
            units=("",) * 36,
            sensor_labels=("imu_trunk", "imu_left", "imu_right"),
        ),
        encoder=SignalPlayback(
            time_s=time_s,
            values=np.zeros((time_s.size, 6), dtype=np.float32),
            channels=(
                "left_position", "left_velocity", "left_torque",
                "right_position", "right_velocity", "right_torque",
            ),
            units=("deg", "deg/s", "Nm", "deg", "deg/s", "Nm"),
        ),
        sync=None,
        sync_trigger_times_s=np.empty(0),
    )


def test_playback_has_requested_modality_layout_and_fixed_sweep_axes() -> None:
    app = QApplication.instance() or QApplication(["test-data-studio-playback"])
    dialog = PlaybackDialog(_complete_playback())

    tabs = dialog.findChild(QTabWidget, "playback_tabs")
    assert tabs is not None and tabs.count() == 3
    waterfalls = dialog.findChildren(_SweepWaterfallPlot)
    signals = dialog.findChildren(_SweepSignalPlot)
    assert len(waterfalls) == 4
    assert len(signals) == 11  # 3 IMUs x 3 sensor types + 2 encoders
    assert [len(plot._curves) for plot in signals[-2:]] == [3, 3]

    dialog.set_playback_time(10.5)
    assert all(abs(float(plot.cursor.value()) - 0.5) < 1e-6 for plot in waterfalls)
    assert all(abs(float(plot.cursor.value()) - 0.5) < 1e-6 for plot in signals)
    y_range = waterfalls[0].getViewBox().viewRange()[1]
    assert y_range[0] == 0.0
    assert y_range[1] == 999.0
    # Ultrasound samples are [depth, time].  Row-major image coordinates keep
    # one A-scan vertical and advance successive frames from left to right.
    assert waterfalls[0].image.axisOrder == "row-major"
    assert waterfalls[0].image.height() == 1000
    assert waterfalls[0].image.width() == waterfalls[0]._columns

    # Exercise the actual show/timer path that previously terminated the app.
    dialog.set_playback_time(0.0)
    dialog.resize(1100, 650)
    dialog.show()
    app.processEvents()
    control_bar = dialog.findChild(QWidget, "playback_control_bar")
    assert control_bar is not None and control_bar.isVisible()
    assert dialog.play_button.isVisible()
    assert control_bar.geometry().bottom() < tabs.geometry().bottom()
    dialog.play_button.click()
    QTest.qWait(80)
    app.processEvents()
    assert dialog.isVisible()
    assert dialog._timer.isActive()
    assert dialog._current_time > 0.0
    dialog.play_button.click()

    dialog.close()
    app.processEvents()


def test_missing_modalities_are_rendered_without_synthetic_curves() -> None:
    app = QApplication.instance() or QApplication(["test-missing-playback"])
    source = _complete_playback()
    dialog = PlaybackDialog(
        TrialPlayback(
            manifest_path=source.manifest_path,
            trial_uuid=source.trial_uuid,
            condition_code=source.condition_code,
            formal_t0_host_monotonic_ns=source.formal_t0_host_monotonic_ns,
            ultrasound=None,
            imu=None,
            encoder=None,
            sync=None,
            sync_trigger_times_s=np.empty(0),
        )
    )

    assert not dialog.findChildren(_SweepWaterfallPlot)
    assert not dialog.findChildren(_SweepSignalPlot)
    dialog.close()
    app.processEvents()
