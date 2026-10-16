"""Measure the model plot inside the complete themed dock layout."""
import json
import math
import os
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime.spec import OutputSpec
from exo_collection.apps.model_runtime.torque import TorqueCommand
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


@pytest.mark.parametrize("width,height", [(1740, 1000), (1440, 900)])
@pytest.mark.parametrize("with_command", [False, True], ids=["no_command", "with_command"])
@pytest.mark.parametrize("distribution", ["per_side", "right_to_both_same_sign"])
def test_full_window_keeps_readable_model_view_and_unclipped_axes(tmp_path, width, height, with_command, distribution):
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families() and Path("C:/Windows/Fonts/msyh.ttc").exists():
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    settings = SharedAppSettings(QSettings(str(tmp_path / "isolated.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    try:
        window.resize(width, height)
        window.show()
        window._body_splitter.setSizes([430, 1290])
        panel = window._model_output_panel
        shared = distribution == "right_to_both_same_sign"
        panel.set_output_spec(OutputSpec(unit="Nm/kg"), active_sides=("right",), motor_distribution=distribution)
        panel.set_output_status("合成输入 · 未连接电机 · 未下发")
        for index in range(261):
            t = index * .05
            panel.add_pipeline({
                "time_ns": 1_000_000_000 + int(t * 1e9), "unit": "Nm/kg",
                "raw_estimate_right": .3 + .03 * math.sin(2 * math.pi * t),
                "estimate_nm": (.3 + .03 * math.sin(2 * math.pi * t)) * 80,
                "estimate_left_nm": 0,
                "raw_estimate_left": 0, "desired_right_nm": (.3 + .03 * math.sin(2 * math.pi * t)) * 80 * .4,
                "desired_left_nm": (.3 + .03 * math.sin(2 * math.pi * t)) * 80 * .4 if shared else 0,
                "motor_distribution": distribution,
            })
            if with_command:
                # A fake successful send snapshot only: no SerialTorqueOutput
                # or port is constructed, connected or armed by this test.
                panel.add_command(TorqueCommand(
                    seq=index, host_monotonic_ns=1_000_000_000 + int(t * 1e9),
                    left_nm=9 if shared else 0, right_nm=9, torque_limit_nm=9, enabled=True, frame=b"",
                ))
        if with_command:
            panel.set_output_status("合成发送快照验收 · 未连接真实硬件")
            sent_values = "左 +9.000 / 右 +9.000 Nm" if shared else "右 +9.000 Nm"
            assert f"最近实际下发（限幅后） {sent_values}" in panel._values_label.text()
            assert "该帧阈值 ±9 Nm ｜使能" in panel._values_label.text()
            assert "+24.000 Nm" in panel._values_label.text()
            assert "+9.600 Nm" in panel._values_label.text()
        for _ in range(20):
            app.processEvents()
        assert (window.width(), window.height()) == (width, height)
        plot = panel._plot.getPlotItem()
        view = plot.vb.sceneBoundingRect()
        assert view.height() >= 180
        assert panel._assistance_view.sceneBoundingRect().height() >= 180
        assert plot.legend.sceneBoundingRect().bottom() <= view.top() + 1
        assert len(plot.legend.items) == (3 if with_command else 2)
        values_rect = panel._values_label.geometry()
        assert values_rect.bottom() < panel._plot.geometry().top()
        for name, unit in (("left", "Nm"), ("right", "Nm")):
            axis = plot.getAxis(name)
            assert unit in axis.labelText
            assert "Nm/kg" not in axis.labelText
            assert axis.label.isVisible() and axis.style["showValues"]
            rect = axis.label.mapRectToScene(axis.label.boundingRect())
            assert rect.left() >= 0
            assert rect.right() <= panel._plot.width()
            assert rect.top() >= 0 and rect.bottom() <= panel._plot.height()
            assert axis.textPen().color().name() == "#374151"
        # Original dock positions and control column remain in place.
        model = window._workspace.dock_for("model_output")
        imu = window._workspace.dock_for("imu")
        assert model.x() == imu.x() and model.y() > imu.y()
        assert abs(window._body_splitter.sizes()[0] - 430) <= 1
        screenshot_root = os.environ.get("MODEL_LAYOUT_SCREENSHOT_DIR")
        if screenshot_root:
            root = Path(screenshot_root)
            root.mkdir(parents=True, exist_ok=True)
            scale = app.devicePixelRatio()
            suffix = ("with_command" if with_command else "no_command") + f"_{distribution}"
            stem = f"model_nm_shared_full_{width}x{height}_scale{scale:g}_{suffix}"
            assert window.grab().save(str(root / f"{stem}.png"))
            assert model.grab().save(str(root / f"model_nm_shared_dock_{width}x{height}_scale{scale:g}_{suffix}.png"))
            (root / f"{stem}.json").write_text(json.dumps({
                "window_size": [window.width(), window.height()], "device_pixel_ratio": scale,
                "viewbox_height": view.height(), "dock_height": model.height(),
                "left_control_width": window._body_splitter.sizes()[0],
                "with_command": with_command, "legend_entries": len(plot.legend.items),
                "motor_distribution": distribution,
                "values_label_height": panel._values_label.height(),
            }, indent=2), encoding="utf-8")
    finally:
        window.close()
        app.processEvents()
