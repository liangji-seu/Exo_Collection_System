"""Loaded model text must fit the existing online control-column width."""
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QPushButton

from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


def test_loaded_model_log_and_errors_do_not_widen_control_column(tmp_path):
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families() and Path("C:/Windows/Fonts/msyh.ttc").exists():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        app.setFont(QFont("Microsoft YaHei", 9))
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    try:
        window.resize(1740, 1000)
        window.show()
        window._body_splitter.setSizes([430, 1290])

        def settle():
            for _ in range(12):
                app.processEvents()

        settle()
        sizes = window._body_splitter.sizes()
        scroll = window._controls_scroll
        viewport_width = scroll.viewport().width()
        cutoff_bottom = window._controls.cutoff.mapTo(scroll.widget(), QPoint()).y() + window._controls.cutoff.height()
        assert cutoff_bottom <= scroll.viewport().height()

        def assert_fits():
            settle()
            assert window._body_splitter.sizes() == sizes
            assert scroll.viewport().width() == viewport_width
            assert scroll.widget().width() == viewport_width
            assert scroll.horizontalScrollBar().maximum() == 0
            assert scroll.verticalScrollBar().maximum() >= 0
            for button in scroll.widget().findChildren(QPushButton):
                if button.isHidden():
                    continue
                left = button.mapTo(scroll.widget(), QPoint()).x()
                assert 0 <= left
                assert left + button.width() <= viewport_width

        assert_fits()
        from PySide6.QtWidgets import QGroupBox
        details = next(group for group in window.findChildren(QGroupBox) if group.title() == "详情与日志")
        assert not details.isChecked()
        details.setChecked(True)
        assert_fits()
        window._load_bundled_model()
        deadline = time.perf_counter() + 15
        while window._load_generation is not None and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert window._spec is not None
        assert_fits()

        # Smoke mode skips _start_log. Reproduce the exact normal-mode label,
        # including a full project path and the normal timestamp/UUID filename.
        root = Path(__file__).resolve().parents[2]
        log_path = root / "runtime_data/model_runtime_logs/20261015_174000_ab12cd34.jsonl"
        window._controls.log_label.setText(f"日志：{log_path}")
        assert_fits()
        assert window._controls.log_label.text() == f"日志：{log_path}"

        for label in (
            window._model_status, window._controls.preflight_label,
            window._fault_label, window._controls.motor_status,
            window._controls.command_label,
            *window._status_labels.values(),
        ):
            label.setText("故障详情：" + "continuous_device_or_backend_error_" * 18)
            assert_fits()
    finally:
        window.close()
        app.processEvents()
