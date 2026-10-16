"""Serial discovery must leave evidence without opening output hardware."""
import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


def test_initial_scan_and_refresh_failure_are_logged_without_opening_serial(tmp_path, monkeypatch):
    import serial
    import serial.tools.list_ports

    app = QApplication.instance() or QApplication([])
    port = SimpleNamespace(device="COM6", description="USB Serial Device",
                           vid=0x16C0, pid=0x0483, hwid="USB VID:PID=16C0:0483")
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [port])

    def forbid_open(*_args, **_kwargs):
        raise AssertionError("Discovery must not open a serial port")

    monkeypatch.setattr(serial, "Serial", forbid_open)
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    widget = ModelRuntimeWindow(settings, smoke_test=True)
    diagnostic_path = widget._diagnostics.path
    try:
        assert widget._controls.port.currentText() == "COM6"
        assert widget._motor is None

        def fail_scan():
            raise OSError("test enumeration failure")

        monkeypatch.setattr(serial.tools.list_ports, "comports", fail_scan)
        widget._controls.refresh_ports()
        assert widget._motor is None
    finally:
        widget.close()
        app.processEvents()

    events = [json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines()]
    scans = [event["scan"] for event in events if event["kind"] == "motor_port_scan"]
    assert scans[0]["selected"] == "COM6"
    assert scans[0]["candidates"] == ["COM6"]
    assert "test enumeration failure" in scans[-1]["error"]
