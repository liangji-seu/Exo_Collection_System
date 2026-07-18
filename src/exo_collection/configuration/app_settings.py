"""Shared persistent preferences for both desktop applications."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, Mapping

from PySide6.QtCore import QSettings, QStandardPaths


# These names are intentionally independent of QApplication.applicationName().
# Collector and Data Studio therefore read and write the same preferences even
# though their window/process application names remain distinct.
SETTINGS_ORGANIZATION_NAME = "Exo Collection System"
SETTINGS_APPLICATION_NAME = "Shared Settings"
DATA_ROOT_KEY = "storage/data_root"
DEVICE_PROFILE_KEY = "collector/device_profile"
HARDWARE_OVERRIDES_KEY = "collector/hardware_device_overrides_json"
ELONXI_RUNTIME_RELATIVE_PATH = (
    Path("SDK_Transfer")
    / "Exo_Hardware_Runtime_Windows_Python311_x64"
    / "elonxi"
)


def fixed_elonxi_sdk_directory() -> Path:
    """Return the fixed Elonxi SDK directory for the deployed folder layout.

    Source checkouts use the laboratory layout where ``Exo_Collection_System``
    and ``SDK_Transfer`` are siblings. Frozen builds search the executable and
    its parents for the same layout so the complete ``Exo`` directory can be
    moved to another drive without invalidating the SDK path.
    """

    if not getattr(sys, "frozen", False):
        repository_root = Path(__file__).resolve().parents[3]
        return (repository_root.parent / ELONXI_RUNTIME_RELATIVE_PATH).resolve()

    executable_directory = Path(sys.executable).resolve().parent
    search_roots = (executable_directory, *executable_directory.parents[:4])
    for root in search_roots:
        candidate = (root / ELONXI_RUNTIME_RELATIVE_PATH).resolve()
        if (candidate / "Elonxi_SDK.dll").is_file():
            return candidate
    return (executable_directory / ELONXI_RUNTIME_RELATIVE_PATH).resolve()


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

    @property
    def device_profile_key(self) -> Literal["simulated", "hardware"]:
        # A fresh laboratory installation should visibly start in hardware
        # mode, but an operator's explicit later choice remains persistent.
        stored = str(self._backend.value(DEVICE_PROFILE_KEY, "hardware")).strip()
        return "hardware" if stored == "hardware" else "simulated"

    def set_device_profile_key(
        self, value: str
    ) -> Literal["simulated", "hardware"]:
        normalized = value.strip().lower()
        if normalized not in {"simulated", "hardware"}:
            raise ValueError(f"unsupported device profile: {value!r}")
        self._backend.setValue(DEVICE_PROFILE_KEY, normalized)
        self._sync_checked("device profile")
        return normalized  # type: ignore[return-value]

    @property
    def hardware_device_overrides(self) -> dict[str, dict[str, Any]]:
        stored = self._backend.value(HARDWARE_OVERRIDES_KEY, "{}")
        try:
            payload = json.loads(str(stored))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        allowed = {"ultrasound", "imu", "encoder", "sync_pulse"}
        result: dict[str, dict[str, Any]] = {}
        for modality, values in payload.items():
            if modality in allowed and isinstance(values, dict):
                result[modality] = dict(values)
        # The default real ultrasound transport is raw Ethernet.  Remove
        # stale settings from the earlier Elonxi/IP draft instead of feeding
        # them into the strict raw-Ethernet parameter model.
        if "ultrasound" in result:
            for stale_key in ("sdk_path", "device_ip", "port"):
                result["ultrasound"].pop(stale_key, None)
        return result

    def set_hardware_device_overrides(
        self, overrides: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        allowed = {"ultrasound", "imu", "encoder", "sync_pulse"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                "unknown hardware override modalities: "
                + ", ".join(sorted(unknown))
            )
        normalized = {
            modality: dict(values) for modality, values in overrides.items()
        }
        if "ultrasound" in normalized:
            for stale_key in ("sdk_path", "device_ip", "port"):
                normalized["ultrasound"].pop(stale_key, None)
        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._backend.setValue(HARDWARE_OVERRIDES_KEY, serialized)
        self._sync_checked("hardware device settings")
        return normalized

    def set_hardware_device_override(
        self,
        modality: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Merge and immediately persist settings for exactly one modality."""

        allowed = {"ultrasound", "imu", "encoder", "sync_pulse"}
        normalized_modality = modality.strip().lower()
        if normalized_modality not in allowed:
            raise ValueError(f"unknown hardware override modality: {modality!r}")
        merged = self.hardware_device_overrides
        merged[normalized_modality] = dict(values)
        persisted = self.set_hardware_device_overrides(merged)
        return dict(persisted[normalized_modality])

    def _sync_checked(self, subject: str) -> None:
        self._backend.sync()
        status = self._backend.status()
        if status == QSettings.Status.AccessError:
            raise RuntimeError(f"cannot save {subject}: QSettings AccessError")
        if status == QSettings.Status.FormatError:
            raise RuntimeError(f"cannot save {subject}: QSettings FormatError")
        if status != QSettings.Status.NoError:
            raise RuntimeError(f"cannot save {subject}: QSettings status {status!r}")
