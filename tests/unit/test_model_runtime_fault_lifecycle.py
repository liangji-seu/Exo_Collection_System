"""Bounded model-runtime fault cleanup tests; no hardware or real ports."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.model_runtime.run_coordinator import OutputState, RunState
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


class Handle:
    def __init__(self, events, *, close_error=False):
        self.events = events
        self.close_error = close_error
        self.is_alive = True
        self.recording_active = False
        self.recording_endpoint = None

    def request_stop(self):
        self.events.append("stop")
        self.is_alive = False

    def join(self, **kwargs):
        self.events.append("join")

    def terminate(self, **kwargs):
        self.events.append("terminate")
        self.is_alive = False

    def close(self):
        self.events.append("close")
        if self.close_error:
            raise OSError("close failed")


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    widget = ModelRuntimeWindow(settings, smoke_test=True)
    yield widget
    widget.close()
    app.processEvents()


def test_fatal_cleanup_closes_every_handle_before_one_popup(window, monkeypatch):
    order = []
    window._handles = {"imu": Handle(order, close_error=True), "emg": Handle(order)}
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: order.append("popup"))

    assert window._fatal_incident("输入故障", "worker failed", source="test")
    assert order[-1] == "popup"
    assert order.count("close") == 2
    assert window._handles == {}
    assert not window._fatal_incident("重复故障", "stale callback", source="test")
    assert order.count("popup") == 1


def test_transient_recovery_is_one_shot_and_cancelled_by_generation(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_connect_modality", lambda *args, **kwargs: calls.append(args) or True)
    assert window._request_input_recovery("imu", "USB disconnected")
    assert window._recovery_pending is not None
    window._cancel_input_recovery("manual stop")
    window._on_recovery_timer()
    assert calls == []


def test_second_transient_failure_escalates_and_display_only_wait_is_not_fatal(window, monkeypatch):
    events = []
    window._handles = {"imu": Handle(events)}
    window._recovery_attempts["imu"] = 1
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: events.append("popup"))
    window._handle_worker_event("imu", WorkerEvent(
        event_type=WorkerEventType.FAILED, modality="imu",
        message="USB disconnected", payload={},
    ))
    assert window._fatal_latched
    assert events[-1] == "popup"

    window._reset_fault_lifecycle("test display-only")
    window._run.run_state = RunState.RUNNING
    window._run.controller = SimpleNamespace(
        last_error=None, waiting_for_inputs=True, last_predict_ns=None,
    )
    window._spec = SimpleNamespace(safety=SimpleNamespace(watchdog_s=0.5))
    window._check_watchdog_impl()
    assert not window._fatal_latched
    window._run.controller = None


def test_output_fault_without_motor_uses_fatal_path(window, monkeypatch):
    shown = []
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: shown.append(args))
    window._run.output_state = OutputState.FAULT
    window._run.reason = "lost enable"
    window._poll_motor()
    assert window._fatal_latched
    assert shown


def test_fatal_keeps_origin_distinct_from_closed_and_never_connected(window, monkeypatch):
    events, captured = [], []
    window._handles = {"imu": Handle(events), "emg": Handle(events)}
    window._handles["imu"]._process = SimpleNamespace(pid=123, exitcode=None)
    window._handles["emg"]._process = SimpleNamespace(pid=456, exitcode=17)
    window._handles["emg"].is_alive = False
    original = window._diagnostics.record
    def record(kind, **fields):
        captured.append((kind, fields))
        original(kind, **fields)
    monkeypatch.setattr(window._diagnostics, "record", record)
    window._fatal_incident("EMG故障", "failed", source="worker_exit", modality="emg")
    assert window._status_strip._last["emg"] == ("bad", "故障来源")
    assert window._status_strip._last["imu"] == ("off", "因本次故障关闭")
    assert window._status_strip._last["encoder"] == ("off", "未连接")
    evidence = next(fields["snapshot"]["runtime"] for kind, fields in captured if kind == "fatal_incident")
    assert evidence["pid"] > 0
    assert evidence["workers"]["imu"] == {"pid": 123, "exitcode": None,
        "is_alive": True, "last_poll_error": None, "last_preview_ns": None, "health": None}
    assert evidence["workers"]["emg"]["exitcode"] == 17
    assert evidence["workers"]["emg"]["is_alive"] is False
    assert window._handles == {}


def test_motor_popup_reports_controller_fault_and_separates_usb_evidence(window):
    snapshot = {"fault_snapshot": {"host_monotonic_ns": 100_000_000,
        "transport": {"counts": {"rx_valid_frames": 42},
        "max_gap_ns": {"rx_valid": 5_000_000}, "last_ns": {"rx_valid": 99_000_000},
        "background": True, "threads": {"reader": {"alive": True}, "sender": {"alive": False}}}},
        "runtime": {"workers": {"imu": {"is_alive": True}}}}
    text = window._incident_message("error=253", snapshot)
    assert "CAN" in text and "253" in text
    assert "42" in text and "5.0" in text and "1.0" in text
    assert "接收线程在运行" in text and "发送线程未运行" in text
    assert "输入进程仍在运行" in text
    assert "不能确定是哪侧" in text


def test_popup_does_not_call_unknown_worker_state_alive(window):
    text = window._incident_message("failure", {"runtime": {"workers": {
        "imu": {"is_alive": {"inspection_error": "closed handle"}}}}})
    assert "状态未能读取" in text
    assert "输入进程仍在运行" not in text
