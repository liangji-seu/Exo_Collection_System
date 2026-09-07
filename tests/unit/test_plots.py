from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio.plots import TimeSeriesPlot


def test_time_series_plot_cyclic_sweep_overwrites_in_place() -> None:
    app = QApplication.instance() or QApplication(["timeseries-overwrite"])
    time_s = np.arange(0.0, 30.0, 0.01, dtype=np.float64)
    values = np.sin(time_s)[:, None]
    plot = TimeSeriesPlot(
        "信号",
        time_s,
        values,
        ("ch_1",),
        window_s=10.0,
    )

    # At t=15 the cycle is [10, 20] (phase=5). The x-axis is FIXED [0, 10] and
    # the full window is populated: the current sweep [10,15] lands at x∈[0,5],
    # the previous tail [5,10] at x∈[5,10] (overwritten, not cleared).
    plot.set_time(15.0, cycle_start_s=10.0)
    x = np.asarray(plot._curves[0].xData, dtype=np.float64)
    assert x.size > 0
    assert float(x.min()) >= -1e-6
    assert float(x.max()) <= 10.0 + 1e-6
    assert float(x.min()) <= 0.01
    assert float(x.max()) >= 9.9
    # x stays monotonic across the two segments (no wrap aliasing / gaps).
    assert bool(np.all(np.diff(x) >= -1e-6))
    assert abs(float(plot.cursor.value()) - 5.0) < 1e-6
    view = plot.getViewBox().viewRange()
    assert abs(view[0][0]) < 1e-6
    assert abs(view[0][1] - 10.0) < 1e-6


def test_time_series_plot_cycles_fixed_window_with_sweeping_cursor() -> None:
    app = QApplication.instance() or QApplication(["timeseries-cycle"])
    time_s = np.arange(0.0, 30.0, 0.01, dtype=np.float64)
    values = np.sin(time_s)[:, None]
    plot = TimeSeriesPlot(
        "信号",
        time_s,
        values,
        ("ch_1",),
        window_s=10.0,
    )

    # Fixed oscilloscope window [0, 10]; the cursor sweeps within it as
    # phase = current - cycle_start, wrapping to the left edge each cycle.
    plot.set_time(12.0, cycle_start_s=10.0)
    view = plot.getViewBox().viewRange()
    assert abs(view[0][0]) < 1e-6
    assert abs(view[0][1] - 10.0) < 1e-6
    assert abs(float(plot.cursor.value()) - 2.0) < 1e-6

    # Within the same cycle the window stays put; only the cursor moves.
    plot.set_time(15.0, cycle_start_s=10.0)
    view = plot.getViewBox().viewRange()
    assert abs(view[0][0]) < 1e-6
    assert abs(view[0][1] - 10.0) < 1e-6
    assert abs(float(plot.cursor.value()) - 5.0) < 1e-6

    # Next cycle: the window is unchanged, the cursor wraps back to the left.
    plot.set_time(21.0, cycle_start_s=20.0)
    view = plot.getViewBox().viewRange()
    assert abs(view[0][0]) < 1e-6
    assert abs(view[0][1] - 10.0) < 1e-6
    assert abs(float(plot.cursor.value()) - 1.0) < 1e-6


def test_time_series_plot_stacks_channels_with_fixed_range() -> None:
    app = QApplication.instance() or QApplication(["timeseries-stacked"])
    time_s = np.arange(0.0, 5.0, 0.01, dtype=np.float64)
    values = np.column_stack(
        (
            np.sin(time_s),
            np.cos(time_s),
        )
    )
    plot = TimeSeriesPlot(
        "EMG",
        time_s,
        values,
        ("ch_1", "ch_2"),
        window_s=5.0,
        fixed_yrange=(-1.0, 1.0),
        offset_per_channel=3.0,
    )
    plot.set_time(5.0)
    assert len(plot._curves) == 2
    first = np.asarray(plot._curves[0].yData, dtype=np.float64)
    second = np.asarray(plot._curves[1].yData, dtype=np.float64)
    # Channel 1 is unshifted sin, channel 2 is cos stacked up by the spacing.
    assert np.allclose(first, np.sin(np.asarray(plot._curves[0].xData)), atol=1e-6)
    assert np.allclose(second, np.cos(np.asarray(plot._curves[1].xData)) + 3.0, atol=1e-6)
    view = plot.getViewBox().viewRange()
    assert view[1][0] <= -1.0 + 1e-6
    assert view[1][1] >= 4.0 - 1e-6
