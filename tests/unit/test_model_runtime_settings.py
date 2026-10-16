"""Online model-runtime settings persistence and preview retry isolation."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.model_runtime import window as runtime_window
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class _FakeImuDialog:
    override = {
        "radio_channel": 25,
        "sample_rate_hz": 100.0,
        "sensor_ids": ("10B42626", "10B4260B", "10B4261F"),
        "stall_detection_enabled": True,
        "stall_duration_s": 5.0,
    }
    seen_current: list[dict] = []

    def __init__(self, current, _parent=None):
        self.seen_current.append(dict(current))
        self.validated_override = dict(self.override)

    def setWindowTitle(self, _title):
        pass

    def exec(self):
        return QDialog.DialogCode.Accepted


def _settings(path):
    return SharedAppSettings(QSettings(str(path), QSettings.Format.IniFormat))


def test_online_imu_settings_round_trip_without_hardware_side_effects(
    qt_app, tmp_path, monkeypatch
):
    settings_path = tmp_path / "runtime_data" / "shared.ini"
    settings_path.parent.mkdir()
    settings = _settings(settings_path)
    settings.set_device_profile_key("simulated")
    _FakeImuDialog.seen_current.clear()
    monkeypatch.setattr(
        "exo_collection.apps.collector.device_settings.ImuDeviceSettingsDialog",
        _FakeImuDialog,
    )

    window = ModelRuntimeWindow(settings, smoke_test=True)
    window._configure_imu()

    restored = _settings(settings_path)
    assert restored.hardware_device_overrides["imu"] == {
        "radio_channel": 25,
        "sample_rate_hz": 100.0,
        "sensor_ids": ["10B42626", "10B4260B", "10B4261F"],
        "stall_detection_enabled": True,
        "stall_duration_s": 5.0,
    }
    assert restored.device_profile_key == "hardware"
    assert window._handles == {}
    assert window._motor is None
    assert not window._controls.enable.isChecked()
    assert not hasattr(window._controls, "direction_confirmed")
    window.close()
    qt_app.processEvents()

    restarted = ModelRuntimeWindow(restored, smoke_test=True)
    assert restarted._handles == {}
    assert restarted._imu_binding_parameters()["sensor_ids"][1:] == ["10B4260B", "10B4261F"]
    restarted._configure_imu()
    assert _FakeImuDialog.seen_current[-1]["sample_rate_hz"] == 100.0
    assert _FakeImuDialog.seen_current[-1]["sensor_ids"] == [
        "10B42626", "10B4260B", "10B4261F"
    ]
    restarted.close()
    qt_app.processEvents()


class _FakePreviewHandle:
    created = 0

    def __init__(self, *_args, **_kwargs):
        type(self).created += 1
        self._alive = False
        self.recording_active = False
        self.recording_endpoint = None

    @property
    def is_alive(self):
        return self._alive

    def start(self):
        self._alive = True

    def request_stop(self):
        self._alive = False

    def join(self, timeout=None):
        return 0

    def terminate(self, timeout=1.0):
        self._alive = False
        return 0

    def close(self):
        self._alive = False

    def poll_events(self, limit=100):
        return []

    def end_recording(self, _trial_uuid):
        self.recording_active = False


def test_failed_preview_is_removed_and_can_be_retried(qt_app, tmp_path, monkeypatch):
    settings = _settings(tmp_path / "runtime_data" / "shared.ini")
    settings.set_device_profile_key("simulated")
    _FakePreviewHandle.created = 0
    monkeypatch.setattr(runtime_window, "ModalityPreviewProcessHandle", _FakePreviewHandle)
    window = ModelRuntimeWindow(settings, smoke_test=True)

    window._connect_modality("imu")
    assert "imu" in window._handles
    assert "#d39e00" in window._connect_dots["imu"].styleSheet()
    window._handle_worker_event(
        "imu",
        WorkerEvent(
            event_type=WorkerEventType.FAILED,
            modality="imu",
            message="simulated failure",
            payload={"fault": "simulated failure"},
        ),
    )
    assert "imu" not in window._handles
    assert window._connect_buttons["imu"].text() == "连接"
    assert window._hub.modalities == ()

    window._connect_modality("imu")
    assert "imu" in window._handles
    assert _FakePreviewHandle.created == 2
    window.close()
    qt_app.processEvents()


def test_connection_refreshes_shared_overrides_before_worker_start(
    qt_app, tmp_path, monkeypatch
):
    settings_path = tmp_path / "runtime_data" / "shared.ini"
    settings_path.parent.mkdir()
    settings = _settings(settings_path)
    settings.set_device_profile_key("hardware")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    latest = {"sample_rate_hz": 100.0, "sensor_ids": ["L", "R", "P"]}
    _settings(settings_path).set_hardware_device_override("imu", latest)
    monkeypatch.setattr(runtime_window, "ModalityPreviewProcessHandle", _FakePreviewHandle)

    window._connect_modality("imu")
    assert window._hardware_overrides["imu"] == latest
    window.close()
    qt_app.processEvents()
