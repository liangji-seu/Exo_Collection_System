from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio.plots import TimeSeriesPlot


def test_time_series_plot_uses_real_time_axis_without_wrap_aliasing() -> None:
    app = QApplication.instance() or QApplication(["timeseries-plot"])
    time_s = np.arange(0.0, 30.0, 0.01, dtype=np.float64)
    values = np.sin(time_s)[:, None]
    plot = TimeSeriesPlot(
        "信号",
        time_s,
        values,
        ("ch_1",),
        window_s=10.0,
    )
    plot.set_time(15.0)
    curve = plot._curves[0]
    x = np.asarray(curve.xData, dtype=np.float64)
    y = np.asarray(curve.yData, dtype=np.float64)

    # The scroll window is the trailing 10 s ending at the cursor.
    assert x.size > 0
    assert float(x.min()) >= 5.0 - 1e-6
    assert float(x.max()) <= 15.0 + 1e-6
    # Real time is monotonic (no ring-buffer wrap points).
    assert bool(np.all(np.diff(x) > 0.0))
    assert np.allclose(y, np.sin(x), atol=1e-6)
    assert float(plot.cursor.value()) == 15.0


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
