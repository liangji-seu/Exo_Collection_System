"""Offline diagnostics tests for the direct-USB Xsens MTw discovery path."""

from __future__ import annotations

from types import ModuleType

import pytest

from exo_collection.adapters.base import AdapterError
from exo_collection.adapters.imu.xsens_mtw_usb import (
    XdaMtwUsbBackend,
    XsensMtwUsbConfig,
)


class _Array(list):
    def size(self):
        return len(self)


class _DeviceId:
    def __init__(self, value: str, *, is_mtw: bool) -> None:
        self.value = value
        self._is_mtw = is_mtw

    def toXsString(self) -> str:
        return self.value

    def isMtw(self) -> bool:
        return self._is_mtw


class _Port:
    def __init__(self, name: str, device_id: str, *, is_mtw: bool) -> None:
        self.name = name
        self.device_id = device_id
        self.is_mtw = is_mtw

    def deviceId(self):
        return _DeviceId(self.device_id, is_mtw=self.is_mtw)

    def portName(self) -> str:
        return self.name

    def baudrate(self) -> int:
        return 115200


class _Control:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed = False

    def openPort(self, name: str, baudrate: int) -> bool:
        self.opened.append(name)
        return True

    def close(self) -> None:
        self.closed = True


def _api(ports: list[_Port]) -> tuple[ModuleType, _Control]:
    api = ModuleType("_fake_xsens_diagnostic_api")
    control = _Control()
    api.XsScanner_scanPorts = lambda: _Array(ports)
    api.XsControl_construct = lambda: control
    api.XsCallback = type("XsCallback", (), {})
    return api, control


def _backend(ports: list[_Port], **kwargs):
    api, control = _api(ports)
    return XdaMtwUsbBackend(
        XsensMtwUsbConfig(**kwargs),
        _api_module=api,
    ), control


def test_empty_sdk_scan_reports_direct_usb_inventory() -> None:
    backend, control = _backend([])

    with pytest.raises(AdapterError) as caught:
        backend.connect(lambda *args: None)

    message = str(caught.value)
    assert "未找到任何配置的 USB MTw" in message
    assert "directUSB" in message
    assert "SDK 枚举 inventory：（空）" in message
    assert control.opened == []


def test_non_mtw_scan_is_reported_without_opening_port() -> None:
    backend, control = _backend(
        [_Port("COM7", "12345678", is_mtw=False)],
    )

    with pytest.raises(AdapterError) as caught:
        backend.connect(lambda *args: None)

    message = str(caught.value)
    assert "没有 MTw 类型设备" in message
    assert "port=COM7, id=12345678, type=non-MTw" in message
    assert control.opened == []


def test_missing_target_does_not_open_other_mtw() -> None:
    backend, control = _backend(
        [_Port("COM8", "10B426260", is_mtw=True)],
        sensor_ids=("10B42625", "", ""),
    )

    with pytest.raises(AdapterError) as caught:
        backend.connect(lambda *args: None)

    message = str(caught.value)
    assert "未发现目标 MTw 设备" in message
    assert "10B42625" in message
    assert "port=COM8, id=10B426260, type=MTw" in message
    assert control.opened == []
