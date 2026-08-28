from __future__ import annotations

import socket
import threading
import time
from typing import Any

from exo_collection.apps.collector.xingying_remote import (
    DEFAULT_TRIGGER_PORT,
    XingYingRemoteTrigger,
)


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _capture_start_xml(name: str = "trial_01") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
        "<CaptureStart>"
        f'<Name VALUE="{name}"/>'
        '<SessionName VALUE=""/>'
        '<Notes VALUE="note"/>'
        '<Description VALUE="desc"/>'
        '<Delay VALUE="0"/>'
        '<DatabasePath VALUE="C:/xing/data"/>'
        '<TimeCode VALUE="00:00:00:00"/>'
        '<PacketID VALUE="0"/>'
        "</CaptureStart>"
    )


def _capture_stop_xml(name: str = "trial_01") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
        "<CaptureStop>"
        f'<Name VALUE="{name}"/>'
        '<Notes VALUE="note"/>'
        '<Assets VALUE=""/>'
        '<TimeCode VALUE="00:00:00:00"/>'
        '<PacketID VALUE="0"/>'
        "</CaptureStop>"
    )


def _send_udp(port: int, payload: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload.encode("utf-8"), ("127.0.0.1", port))
    finally:
        sock.close()


class _Collector:
    def __init__(self) -> None:
        self.received: list[tuple[str, dict[str, Any], int, int]] = []
        self.event = threading.Event()

    def __call__(
        self,
        kind: str,
        payload: dict[str, Any],
        host_monotonic_ns: int,
        host_utc_ns: int,
    ) -> None:
        self.received.append((kind, payload, host_monotonic_ns, host_utc_ns))
        self.event.set()


def _wait_for(event: threading.Event, timeout_s: float = 3.0) -> bool:
    return event.wait(timeout_s)


def test_default_trigger_port_is_7061() -> None:
    assert DEFAULT_TRIGGER_PORT == 7061


def test_trigger_receives_capture_start_and_stop() -> None:
    collector = _Collector()
    trigger = XingYingRemoteTrigger(
        ip="127.0.0.1",
        port=_free_udp_port(),
        on_trigger=collector,
    )
    trigger.start()
    try:
        assert trigger.is_running
        _send_udp(trigger.port, _capture_start_xml("trial_01"))
        assert _wait_for(collector.event), "capture_start notification not received"

        collector.event.clear()
        _send_udp(trigger.port, _capture_stop_xml("trial_01"))
        assert _wait_for(collector.event), "capture_stop notification not received"

        assert len(collector.received) == 2
        (start_kind, start_payload, start_mono, start_utc), (
            stop_kind,
            stop_payload,
            stop_mono,
            stop_utc,
        ) = collector.received

        assert start_kind == "capture_start"
        assert start_payload["capture_name"] == "trial_01"
        assert start_payload["database_path"] == "C:/xing/data"
        assert start_payload["notes"] == "note"
        assert start_payload["description"] == "desc"
        assert start_payload["delay"] == "0"
        assert start_payload["timecode"] == "00:00:00:00"
        assert start_payload["packet_id"] == "0"
        assert start_mono > 0
        assert start_utc > 0

        assert stop_kind == "capture_stop"
        assert stop_payload["capture_name"] == "trial_01"
        assert stop_mono >= start_mono
    finally:
        trigger.stop()
    assert not trigger.is_running


def test_trigger_ignores_malformed_and_unknown_root_packets() -> None:
    collector = _Collector()
    trigger = XingYingRemoteTrigger(
        ip="127.0.0.1",
        port=_free_udp_port(),
        on_trigger=collector,
    )
    trigger.start()
    try:
        _send_udp(trigger.port, "this is not xml at all")
        _send_udp(trigger.port, "<UnknownRoot><Name VALUE=\"x\"/></UnknownRoot>")
        time.sleep(0.7)  # exceed the 0.5s recv timeout to flush any dispatch
        assert collector.received == []
    finally:
        trigger.stop()


def test_trigger_ignores_packet_without_name() -> None:
    collector = _Collector()
    trigger = XingYingRemoteTrigger(
        ip="127.0.0.1",
        port=_free_udp_port(),
        on_trigger=collector,
    )
    trigger.start()
    try:
        _send_udp(trigger.port, "<CaptureStart><SessionName VALUE=\"\"/></CaptureStart>")
        time.sleep(0.7)
        assert collector.received == []
    finally:
        trigger.stop()


def test_stop_prevents_further_callbacks() -> None:
    collector = _Collector()
    trigger = XingYingRemoteTrigger(
        ip="127.0.0.1",
        port=_free_udp_port(),
        on_trigger=collector,
    )
    trigger.start()
    assert trigger.is_running
    trigger.stop()
    assert not trigger.is_running

    _send_udp(trigger.port, _capture_start_xml("trial_after_stop"))
    time.sleep(0.7)
    assert collector.received == []

    # stop() is idempotent.
    trigger.stop()
