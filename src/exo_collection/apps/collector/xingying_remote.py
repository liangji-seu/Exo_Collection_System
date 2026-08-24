# -*- coding: utf-8 -*-
"""XINGYING 远程控制客户端（纯标准库 UDP XML，无 nokovpy SDK 依赖）。

通过 UDP 向 XINGYING 的「捕获--远程」监听端口（默认 7060）发送 CaptureStart /
CaptureStop 命令，控制其开始/停止录制 .cap 文件。

动捕 Marker 与测力台数据由 XINGYING 原生录制为 .cap + 伴生目录，采集脚本只负责
在 Trial 开始/结束时触发录制，不再从 SDK 读取原始 analog 数据。

参考：XING Python SDK examples/远程控制.txt
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Final

DEFAULT_REMOTE_IP: Final = "127.0.0.1"  # XINGYING 通常与本机同机，监听 0.0.0.0:7060
DEFAULT_REMOTE_PORT: Final = 7060


def _xml_escape(value: str) -> str:
    """转义 XML 属性值中的特殊字符，避免破坏单个 UDP 数据包。"""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _normalize_database_path(database_path: str | Path) -> str:
    """XINGYING 的 DatabasePath 示例均为正斜杠路径，这里统一转换。"""
    return str(database_path).replace("\\", "/")


class XingYingRemoteCapture:
    """Fire-and-forget XINGYING 录制触发器（UDP XML）。"""

    def __init__(
        self,
        ip: str = DEFAULT_REMOTE_IP,
        port: int = DEFAULT_REMOTE_PORT,
    ) -> None:
        self._ip = ip or DEFAULT_REMOTE_IP
        self._port = int(port or DEFAULT_REMOTE_PORT)
        self._active_name: str | None = None

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def port(self) -> int:
        return self._port

    @property
    def active_name(self) -> str | None:
        return self._active_name

    def capture_start(self, name: str, database_path: str | Path) -> str:
        """发送 CaptureStart，XINGYING 开始录制，文件写入 ``database_path``。"""
        xml = self._build_capture_start_xml(name, _normalize_database_path(database_path))
        self._send(xml)
        self._active_name = name
        return xml

    def capture_stop(self, name: str | None = None) -> str | None:
        """发送 CaptureStop；``name`` 缺省时使用最近一次 capture_start 的名称。"""
        target = name or self._active_name
        if not target:
            return None
        xml = self._build_capture_stop_xml(target)
        self._send(xml)
        self._active_name = None
        return xml

    def clear(self) -> None:
        """丢弃当前活动录制名，不发送任何数据包。"""
        self._active_name = None

    @staticmethod
    def _build_capture_start_xml(name: str, database_path: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
            '<CaptureStart>'
            f'<Name VALUE="{_xml_escape(name)}"/>'
            '<SessionName VALUE=""/>'
            '<Notes VALUE=""/>'
            '<Description VALUE=""/>'
            '<Delay VALUE="0"/>'
            f'<DatabasePath VALUE="{_xml_escape(database_path)}"/>'
            '<TimeCode VALUE="00:00:00:00"/>'
            '<PacketID VALUE="0"/>'
            '</CaptureStart>'
        )

    @staticmethod
    def _build_capture_stop_xml(name: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
            '<CaptureStop>'
            f'<Name VALUE="{_xml_escape(name)}"/>'
            '<Notes VALUE=""/>'
            '<Assets VALUE=""/>'
            '<TimeCode VALUE="00:00:00:00"/>'
            '<PacketID VALUE="0"/>'
            '</CaptureStop>'
        )

    def _send(self, xml: str) -> None:
        data = xml.encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(data, (self._ip, self._port))
        finally:
            sock.close()


__all__ = [
    "DEFAULT_REMOTE_IP",
    "DEFAULT_REMOTE_PORT",
    "XingYingRemoteCapture",
]
