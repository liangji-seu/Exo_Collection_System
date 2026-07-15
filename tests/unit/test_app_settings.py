from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector import CollectorWindow
from exo_collection.apps.collector.main import (
    _build_parser as collector_parser,
    main as collector_entrypoint,
)
from exo_collection.apps.data_studio import CATALOG_FILENAME, DataStudioWindow
from exo_collection.apps.data_studio.main import (
    _build_parser as studio_parser,
    main as studio_entrypoint,
)
from exo_collection.configuration.app_settings import (
    SETTINGS_APPLICATION_NAME,
    SETTINGS_ORGANIZATION_NAME,
    SharedAppSettings,
    create_shared_settings_backend,
)


collector_main_module = import_module("exo_collection.apps.collector.main")
studio_main_module = import_module("exo_collection.apps.data_studio.main")


def _file_settings(path: Path) -> SharedAppSettings:
    return SharedAppSettings(QSettings(str(path), QSettings.Format.IniFormat))


class _TrackingSettings:
    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self.read_count = 0

    @property
    def data_root(self) -> Path:
        self.read_count += 1
        return self._data_root


class _StatusBackend:
    def __init__(self, status: QSettings.Status) -> None:
        self._status = status
        self.calls: list[str] = []

    def setValue(self, _key: str, _value: str) -> None:  # noqa: N802 - Qt API
        self.calls.append("setValue")

    def sync(self) -> None:
        self.calls.append("sync")

    def status(self) -> QSettings.Status:
        self.calls.append("status")
        return self._status


def test_shared_backend_does_not_follow_qapplication_name() -> None:
    app = QApplication.instance() or QApplication(["test-app-settings"])
    original_name = app.applicationName()
    try:
        app.setApplicationName("Exo Collector")
        collector_backend = create_shared_settings_backend()
        app.setApplicationName("Exo Data Studio")
        studio_backend = create_shared_settings_backend()
    finally:
        app.setApplicationName(original_name)

    assert collector_backend.organizationName() == SETTINGS_ORGANIZATION_NAME
    assert collector_backend.applicationName() == SETTINGS_APPLICATION_NAME
    assert studio_backend.fileName() == collector_backend.fileName()


def test_data_root_round_trips_through_separate_settings_instances(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "shared.ini"
    selected = tmp_path / "selected data"

    normalized = _file_settings(settings_path).set_data_root(selected)

    assert normalized == selected.resolve()
    assert _file_settings(settings_path).data_root == selected.resolve()


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (QSettings.Status.AccessError, r"权限.*AccessError"),
        (QSettings.Status.FormatError, r"格式无效.*FormatError"),
    ],
)
def test_data_root_write_reports_qsettings_failures(
    tmp_path: Path,
    status: QSettings.Status,
    message: str,
) -> None:
    backend = _StatusBackend(status)
    settings = SharedAppSettings(backend)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=message):
        settings.set_data_root(tmp_path / "selected")

    assert backend.calls == ["setValue", "sync", "status"]


def test_both_windows_confirm_changes_to_the_same_persistent_root(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["test-shared-data-root"])
    settings_path = tmp_path / "shared.ini"
    collector_root = tmp_path / "collector-selected"
    studio_root = tmp_path / "studio-selected"

    collector = CollectorWindow(
        tmp_path / "initial",
        settings=_file_settings(settings_path),
    )
    collector.data_root_edit.setText(str(collector_root))
    assert collector.build_request().data_root == collector_root.resolve()
    collector.close()
    app.processEvents()

    studio_settings = _file_settings(settings_path)
    assert studio_settings.data_root == collector_root.resolve()
    studio = DataStudioWindow(
        studio_settings.data_root,
        settings=studio_settings,
        autostart_refresh=False,
    )
    studio.set_data_root(studio_root, refresh=False)
    studio.close()
    app.processEvents()

    assert _file_settings(settings_path).data_root == studio_root.resolve()


def test_ui_smoke_tests_use_clean_temporary_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_root = tmp_path / "persistent-root"
    persistent_root.mkdir()
    marker = persistent_root / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    settings = _TrackingSettings(persistent_root)
    temporary_roots: list[Path] = []

    real_collector_window = collector_main_module.CollectorWindow
    real_studio_window = studio_main_module.DataStudioWindow

    def collector_window(data_root: Path, *args: object, **kwargs: object) -> object:
        temporary_roots.append(Path(data_root))
        return real_collector_window(data_root, *args, **kwargs)

    def studio_window(data_root: Path, *args: object, **kwargs: object) -> object:
        temporary_roots.append(Path(data_root))
        return real_studio_window(data_root, *args, **kwargs)

    monkeypatch.setattr(collector_main_module, "CollectorWindow", collector_window)
    monkeypatch.setattr(studio_main_module, "DataStudioWindow", studio_window)

    assert collector_entrypoint(
        ["--smoke-test"], settings=settings  # type: ignore[arg-type]
    ) == 0
    assert studio_entrypoint(
        ["--smoke-test"], settings=settings  # type: ignore[arg-type]
    ) == 0

    assert settings.read_count == 0
    assert len(temporary_roots) == 2
    assert all(not root.exists() for root in temporary_roots)
    assert marker.read_text(encoding="utf-8") == "untouched"
    assert not (persistent_root / CATALOG_FILENAME).exists()


def test_collection_smoke_uses_a_temporary_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_root = tmp_path / "persistent-root"
    persistent_root.mkdir()
    settings = _TrackingSettings(persistent_root)
    observed: dict[str, object] = {}

    def run_smoke(data_root: Path, duration_s: float) -> int:
        observed["root"] = data_root
        observed["duration_s"] = duration_s
        assert data_root.is_dir()
        (data_root / "smoke-output.txt").write_text("temporary", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        collector_main_module,
        "_run_collection_smoke",
        run_smoke,
    )

    assert collector_entrypoint(
        ["--collect-smoke-test", "--duration", "0.1"],
        settings=settings,  # type: ignore[arg-type]
    ) == 0

    temporary_root = observed["root"]
    assert isinstance(temporary_root, Path)
    assert observed["duration_s"] == 0.1
    assert settings.read_count == 0
    assert not temporary_root.exists()
    assert not any(persistent_root.iterdir())


@pytest.mark.parametrize("build_parser", [collector_parser, studio_parser])
def test_gui_parsers_reject_data_root_but_keep_smoke_test(build_parser: object) -> None:
    parser = build_parser()  # type: ignore[operator]
    assert parser.parse_args(["--smoke-test"]).smoke_test
    with pytest.raises(SystemExit):
        parser.parse_args(["--data-root", "somewhere"])
