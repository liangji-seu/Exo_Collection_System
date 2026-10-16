"""Committed keyboard input persists without scroll/step or run side effects."""
import json

import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QApplication

from exo_collection.apps.model_runtime.control_widgets import AssistanceControls
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings
from exo_collection.configuration.app_settings import MODEL_RUNTIME_PARAMETERS_KEY


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settings_at(path):
    return SharedAppSettings(QSettings(str(path), QSettings.Format.IniFormat))


def model_defaults():
    return ModelSpec.model_validate({"model": {"backend": "demo"},
        "features": {"control_rate_hz": 20, "window_s": .2, "inputs": [{"modality": "imu"}]},
        "output": {"torque_limit_nm": 4},
        "assistance": {"mass_kg": 75, "gain": .6, "delay_s": .1, "lowpass_hz": 6}})


def test_keyboard_commits_round_trip_all_five_parameters_and_zero_wins_over_models(app, tmp_path):
    path = tmp_path / "preferences.ini"
    settings = settings_at(path)
    window = ModelRuntimeWindow(settings, smoke_test=True)
    window.show()
    app.processEvents()
    expected = {"mass_kg": 67.25, "torque_limit_nm": 0, "gain": .375,
                "delay_ms": 125, "lowpass_hz": 0}
    try:
        for name, value in expected.items():
            field = window._controls.parameter_fields[name]
            field.setFocus()
            field.selectAll()
            QTest.keyClicks(field, str(value))
            # Keyboard tracking is off: uncommitted text is not saved.
            assert name not in settings.read_model_runtime_parameters()[0]
            QTest.keyClick(field, Qt.Key.Key_Return)
            app.processEvents()
            assert settings.read_model_runtime_parameters()[0][name] == value
        assert window._handles == {} and window._motor is None
        assert not window.controller_running
    finally:
        window.close()
        app.processEvents()
    records = [json.loads(line) for line in window._diagnostics.path.read_text(encoding="utf-8").splitlines()]
    saved = {record["parameter"]: record["value"] for record in records
             if record["kind"] == "runtime_parameter_saved"}
    assert saved == expected
    restarted = ModelRuntimeWindow(settings_at(path), smoke_test=True)
    try:
        restarted._controls.load_spec(model_defaults())
        assert {name: field.value() for name, field in restarted._controls.parameter_fields.items()} == expected
        assert restarted._handles == {} and restarted._motor is None
        assert restarted._controller is None and not restarted.controller_running
        cached = json.loads(settings_at(path)._backend.value(MODEL_RUNTIME_PARAMETERS_KEY))
        assert set(cached) == set(expected)
    finally:
        restarted.close()
        app.processEvents()


def test_same_session_edits_and_untouched_zero_are_not_overwritten_by_model_defaults(app):
    controls = AssistanceControls()
    try:
        controls.load_spec(model_defaults())
        controls.mass.setValue(0)
        controls.gain.setValue(.125)
        # A committed zero remains an explicit preference even if unchanged.
        controls.limit.setValue(0)
        controls.limit.editingFinished.emit()
        controls.load_spec(model_defaults())
        assert controls.mass.value() == 0
        assert controls.gain.value() == .125
        assert controls.limit.value() == 0
    finally:
        controls.close()


def test_wheel_keys_and_step_buttons_cannot_change_any_parameter(app):
    controls = AssistanceControls()
    controls.show()
    app.processEvents()
    try:
        for field in controls.parameter_fields.values():
            field.setValue((field.maximum() + field.minimum()) / 2)
            initial = field.value()
            assert field.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
            for key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                QTest.keyClick(field, key)
                assert field.value() == initial
            field.stepUp()
            field.stepDown()
            field.stepBy(100)
            assert field.value() == initial
            for focused in (False, True):
                if focused:
                    field.setFocus()
                else:
                    field.clearFocus()
                event = QWheelEvent(QPointF(4, 4), QPointF(field.mapToGlobal(QPoint(4, 4))),
                    QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
                QApplication.sendEvent(field, event)
                assert field.value() == initial
        field = controls.gain
        field.setFocus()
        field.selectAll()
        QApplication.clipboard().setText("0.625")
        QTest.keyClick(field, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(field, Qt.Key.Key_Return)
        assert field.value() == .625
        assert field.validate("1.500", 5)[0].name == "Invalid"
    finally:
        controls.close()


def test_corrupt_cache_recovers_per_field_logs_and_does_not_rewrite_it(app, tmp_path):
    settings = settings_at(tmp_path / "invalid.ini")
    corrupt = '{"mass_kg":"bad","gain":NaN,"delay_ms":-5,"torque_limit_nm":2.5,"lowpass_hz":0}'
    settings._backend.setValue(MODEL_RUNTIME_PARAMETERS_KEY, corrupt)
    settings._backend.sync()
    window = ModelRuntimeWindow(settings, smoke_test=True)
    try:
        values, errors = settings.read_model_runtime_parameters()
        assert values == {"torque_limit_nm": 2.5, "lowpass_hz": 0}
        assert len(errors) == 3
        window._controls.load_spec(model_defaults())
        assert window._controls.mass.value() == 75
        assert window._controls.gain.value() == .6
        assert window._controls.delay.value() == 100
        assert window._controls.limit.value() == 2.5
        assert window._controls.cutoff.value() == 0
        assert settings._backend.value(MODEL_RUNTIME_PARAMETERS_KEY) == corrupt
        records = [json.loads(line) for line in window._diagnostics.path.read_text(encoding="utf-8").splitlines()]
        assert len([item for item in records if item["kind"] == "runtime_parameter_cache_invalid"]) == 3
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize("name,value", [("mass_kg", float("inf")), ("gain", float("nan")),
    ("gain", True), ("gain", 1.01), ("delay_ms", .5), ("lowpass_hz", -1),
    ("mass_kg", 10 ** 1000), ("torque_limit_nm", 18.01), ("torque_limit_nm", float("nan"))])
def test_invalid_parameter_save_never_writes_cache(app, tmp_path, name, value):
    settings = settings_at(tmp_path / "invalid_save.ini")
    with pytest.raises(ValueError):
        settings.set_model_runtime_parameters({name: value})
    assert not settings._backend.contains(MODEL_RUNTIME_PARAMETERS_KEY)


@pytest.mark.parametrize("limit", [9, 12.5, 18])
def test_extended_torque_range_round_trips_without_raising_old_cache_or_changing_lowpass(app, tmp_path, limit):
    path = tmp_path / "torque_limit.ini"
    settings_at(path).set_model_runtime_parameters({"torque_limit_nm": limit})
    restored, errors = settings_at(path).read_model_runtime_parameters()
    assert restored == {"torque_limit_nm": limit} and not errors
    controls = AssistanceControls()
    try:
        assert controls.limit.value() == 0
        assert controls.limit.maximum() == 18
        assert controls.cutoff.maximum() == 9
        assert "0–18 Nm" in controls.limit.toolTip()
        controls.restore_parameters(restored)
        controls.load_spec(model_defaults())
        assert controls.limit.value() == limit  # A saved 9 stays 9.
    finally:
        controls.close()
