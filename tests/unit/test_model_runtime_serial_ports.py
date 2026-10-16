"""Motor inventory scans never open ports or alter assistance parameters."""
import builtins
from types import SimpleNamespace

import pytest
import serial
from serial.tools import list_ports
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from exo_collection.adapters.encoder.teensy_serial import TEENSY_PID, TEENSY_VID
from exo_collection.apps.model_runtime.control_widgets import AssistanceControls
from exo_collection.apps.model_runtime.serial_ports import scan_motor_ports


def port(device, *, teensy=False, description="USB Serial Device", hwid="USB", vid=None, pid=None):
    return SimpleNamespace(device=device, description=description, hwid=hwid,
                           vid=TEENSY_VID if teensy else vid, pid=TEENSY_PID if teensy else pid)


@pytest.fixture
def inventory(monkeypatch):
    values = []
    monkeypatch.setattr(list_ports, "comports", lambda: list(values))
    def forbidden(*args, **kwargs):
        pytest.fail("An inventory scan must never open serial.Serial")
    monkeypatch.setattr(serial, "Serial", forbidden)
    return values


@pytest.fixture
def controls(inventory):
    app = QApplication.instance() or QApplication([])
    inventory.extend([
        port("COM7", description="Xsens MTw", vid=0x2639, pid=0x0200),
        port("COM6", teensy=True, hwid="USB VID:PID=16C0:0483"),
        port("COM3", description="Standard Serial over Bluetooth link", hwid="BTHENUM"),
        port("COM4", description="蓝牙串口"),
        port("COM10", description="USB Adapter"),
    ])
    widget = AssistanceControls()
    yield widget
    widget.close()
    app.processEvents()


def test_constructor_scans_selects_only_teensy_and_keeps_com_value(controls):
    assert controls.port.currentText() == "COM6"
    assert [controls.port.itemText(i) for i in range(controls.port.count())] == ["COM6", "COM7", "COM10"]
    assert controls.last_port_scan["candidates"] == ["COM6"]
    assert controls.last_port_scan["excluded"] == ["COM3", "COM4"]
    assert controls.last_port_scan["selection_reason"] == "unique_teensy"
    assert "16C0:0483" in controls.port.itemData(0, Qt.ItemDataRole.ToolTipRole)


def test_refresh_with_empty_selection_selects_unique_candidate_and_emits(controls):
    controls.port.setCurrentText("")
    reports = []
    controls.ports_scanned.connect(reports.append)
    controls.refresh_ports()
    assert controls.port.currentText() == "COM6"
    assert reports[-1] == controls.last_port_scan
    assert reports[-1]["previous"] == "" and reports[-1]["selected"] == "COM6"


def test_valid_user_selection_survives_refresh_even_with_unique_teensy(controls):
    controls.port.setCurrentText("COM10")
    controls.refresh_ports()
    assert controls.port.currentText() == "COM10"
    assert controls.last_port_scan["selection_reason"] == "preserved"


def test_removed_port_is_cleared_and_new_unique_teensy_can_be_selected(controls, inventory):
    inventory[:] = [port("COM7", description="Xsens MTw", vid=0x2639, pid=0x0200)]
    controls.refresh_ports()
    assert controls.port.currentText() == ""
    assert controls.last_port_scan["previous_removed"]
    assert "已拔出" in controls.motor_status.text()
    inventory.append(port("COM12", teensy=True))
    controls.refresh_ports()
    assert controls.port.currentText() == "COM12"


def test_multiple_candidates_do_not_choose_first_when_selection_is_empty(controls, inventory):
    inventory.append(port("COM11", teensy=True))
    controls.port.setCurrentText("")
    controls.refresh_ports()
    assert controls.port.currentText() == ""
    assert controls.last_port_scan["selection_reason"] == "ambiguous"
    assert "多个" in controls.motor_status.text()
    controls.port.setCurrentText("COM11")
    controls.refresh_ports()
    assert controls.port.currentText() == "COM11"


def test_no_teensy_or_no_ports_never_selects_xsens_or_bluetooth(controls, inventory):
    inventory[:] = [port("COM7", description="Xsens MTw", vid=0x2639, pid=0x0200),
                    port("COM3", hwid="BTHENUM")]
    controls.refresh_ports()
    assert controls.port.currentText() == ""
    assert controls.last_port_scan["candidates"] == []
    assert controls.port.count() == 1
    inventory.clear()
    controls.refresh_ports()
    assert controls.port.count() == 0 and controls.port.currentText() == ""
    assert controls.last_port_scan["selection_reason"] == "empty"
    assert "未发现" in controls.motor_status.text()


def test_manual_keyboard_port_remains_available_without_enumeration(controls, inventory):
    controls.port.setCurrentText("COM42")
    inventory.clear()
    controls.refresh_ports()
    assert controls.port.isEditable() and controls.port.currentText() == "COM42"
    assert controls.last_port_scan["selection_reason"] == "manual_unverified"
    assert "未在本次枚举中发现" in controls.port.toolTip()


def test_scan_error_is_visible_and_preserves_previous_list_and_selection(controls, monkeypatch):
    def fail():
        raise OSError("enumeration unavailable")
    monkeypatch.setattr(list_ports, "comports", fail)
    reports = []
    controls.ports_scanned.connect(reports.append)
    controls.refresh_ports()
    assert controls.port.currentText() == "COM6" and controls.port.count() == 3
    assert "OSError" in reports[-1]["error"]
    assert "扫描失败" in controls.motor_status.text()
    assert "enumeration unavailable" in controls.port.toolTip()


def test_missing_pyserial_is_reported_without_import_side_effects(inventory, monkeypatch):
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "serial.tools":
            raise ImportError("test missing pyserial")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    report = scan_motor_ports()
    assert report["ports"] == [] and "未安装 pyserial" in report["error"]


def test_popup_scans_once_and_scan_preserves_parameters_connected_status_and_wheel(controls, inventory, monkeypatch):
    controls.mass.setValue(71)
    controls.limit.setValue(18)
    values = controls.signature()
    controls.motor_button.setText("断开电机")
    controls.motor_status.setText("已连接 COM6 · 失能")
    scans, edits = [], []
    controls.ports_scanned.connect(scans.append)
    controls.changed.connect(lambda: edits.append(True))
    monkeypatch.setattr(list_ports, "comports", lambda: list(inventory))
    controls.port.showPopup()
    controls.port.hidePopup()
    assert len(scans) == 1
    assert controls.signature() == values and not edits
    assert controls.motor_status.text() == "已连接 COM6 · 失能"
    event = QWheelEvent(QPointF(2, 2), QPointF(2, 2), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    before = controls.port.currentText()
    controls.port.wheelEvent(event)
    assert not event.isAccepted() and controls.port.currentText() == before
