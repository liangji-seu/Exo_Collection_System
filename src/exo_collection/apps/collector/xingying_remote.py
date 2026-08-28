# -*- coding: utf-8 -*-
"""XINGYING 远程控制客户端（纯标准库 UDP XML，无 nokovpy SDK 依赖）。

通过 UDP 向 XINGYING 的「捕获--远程」监听端口（默认 7060）发送 CaptureStart /
CaptureStop 命令，控制其开始/停止录制 .cap 文件。

动捕 Marker 与测力台数据由 XINGYING 原生录制为 .cap + 伴生目录，采集脚本只负责
在 Trial 开始/结束时触发录制，不再从 SDK 读取原始 analog 数据。

参考：XING Python SDK examples/远程控制.txt
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Final

LOG = logging.getLogger("exo_collection.collector.xingying")

DEFAULT_REMOTE_IP: Final = "127.0.0.1"  # XINGYING 通常与本机同机，监听 0.0.0.0:7060
DEFAULT_REMOTE_PORT: Final = 7060
DEFAULT_TRIGGER_PORT: Final = 7061  # XINGYING「捕获--触发」广播端口，第三方监听


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


def _xml_child_value(root: ET.Element, name: str) -> str:
    """返回子元素 ``name`` 的 ``VALUE`` 属性；缺失或为空时返回空字符串。"""
    element = root.find(name)
    if element is None:
        return ""
    return str(element.get("VALUE") or "").strip()


class XingYingRemoteTrigger:
    """监听 XINGYING「捕获--触发」端口（默认 7061）的起停通知。

    XINGYING 在真正开始/停止录制时反向广播 ``CaptureStart``/``CaptureStop``
    XML。本类绑定一个 UDP socket 监听该端口，解析通知并以收到时刻的主机时钟
    回调 ``on_trigger(kind, payload, host_monotonic_ns, host_utc_ns)``。解析
    失败或缺失 ``Name`` 的通知会被静默忽略，不影响录制。
    """

    def __init__(
        self,
        ip: str,
        port: int = DEFAULT_TRIGGER_PORT,
        on_trigger: Callable[[str, dict[str, Any], int, int], None] | None = None,
    ) -> None:
        self._ip = ip or "0.0.0.0"
        self._port = int(port or DEFAULT_TRIGGER_PORT)
        self._on_trigger = on_trigger
        self._stop_event = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # INADDR_ANY：XINGYING 可能从任意网卡广播，绑定空地址最可靠。
        sock.bind(("", self._port))
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(
            target=self._loop, name="xingying-trigger", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._sock = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if data:
                self._dispatch(data)

    def _dispatch(self, data: bytes) -> None:
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            LOG.debug("忽略无法解析的 XINGYING 触发包")
            return
        if root.tag not in ("CaptureStart", "CaptureStop"):
            return
        payload = {
            "capture_name": _xml_child_value(root, "Name"),
            "database_path": _xml_child_value(root, "DatabasePath"),
            "notes": _xml_child_value(root, "Notes"),
            "description": _xml_child_value(root, "Description"),
            "delay": _xml_child_value(root, "Delay"),
            "timecode": _xml_child_value(root, "TimeCode"),
            "packet_id": _xml_child_value(root, "PacketID"),
        }
        if not payload["capture_name"]:
            LOG.debug("忽略缺少 Name 的 XINGYING 触发包")
            return
        kind = "capture_start" if root.tag == "CaptureStart" else "capture_stop"
        host_monotonic_ns = time.perf_counter_ns()
        host_utc_ns = time.time_ns()
        if self._on_trigger is not None:
            try:
                self._on_trigger(kind, payload, host_monotonic_ns, host_utc_ns)
            except Exception:
                LOG.exception("XINGYING 触发回调失败")


__all__ = [
    "DEFAULT_REMOTE_IP",
    "DEFAULT_REMOTE_PORT",
    "DEFAULT_TRIGGER_PORT",
    "XingYingRemoteCapture",
    "XingYingRemoteTrigger",
]
