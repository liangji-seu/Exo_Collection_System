"""Shared persistent preferences for both desktop applications."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths


# These names are intentionally independent of QApplication.applicationName().
# Collector and Data Studio therefore read and write the same preferences even
# though their window/process application names remain distinct.
SETTINGS_ORGANIZATION_NAME = "Exo Collection System"
SETTINGS_APPLICATION_NAME = "Shared Settings"
DATA_ROOT_KEY = "storage/data_root"


def default_data_root() -> Path:
    """Return an application-name-independent local data directory."""

    base_text = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericDataLocation
    )
    base = Path(base_text) if base_text else Path.home() / ".local" / "share"
    return (base / "ExoCollectionSystem" / "data").expanduser().resolve()


def create_shared_settings_backend() -> QSettings:
    """Create the fixed QSettings namespace used by both desktop apps."""

    return QSettings(SETTINGS_ORGANIZATION_NAME, SETTINGS_APPLICATION_NAME)


class SharedAppSettings:
    """Typed access to preferences shared by Collector and Data Studio."""

    def __init__(self, backend: QSettings | None = None) -> None:
        self._backend = (
            backend if backend is not None else create_shared_settings_backend()
        )

    @property
    def data_root(self) -> Path:
        stored = self._backend.value(DATA_ROOT_KEY)
        if isinstance(stored, str) and stored.strip():
            return Path(stored).expanduser().resolve()
        return default_data_root()

    def set_data_root(self, data_root: str | Path) -> Path:
        text = str(data_root).strip()
        if not text:
            raise ValueError("数据根目录不能为空")
        normalized = Path(text).expanduser().resolve()
        self._backend.setValue(DATA_ROOT_KEY, str(normalized))
        # Persist immediately so the other desktop process sees the selection
        # even when this process remains open or exits unexpectedly later.
        self._backend.sync()
        status = self._backend.status()
        if status == QSettings.Status.AccessError:
            raise RuntimeError(
                "无法保存数据根目录：没有权限写入共享设置"
                "（QSettings AccessError）。"
            )
        if status == QSettings.Status.FormatError:
            raise RuntimeError(
                "无法保存数据根目录：共享设置格式无效"
                "（QSettings FormatError）。"
            )
        if status != QSettings.Status.NoError:
            raise RuntimeError(
                f"无法保存数据根目录：QSettings 返回未知状态 {status!r}。"
            )
        return normalized
