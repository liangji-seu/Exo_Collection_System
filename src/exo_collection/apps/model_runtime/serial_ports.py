"""Motor-port inventory only: never open, probe or write to a serial device."""
from __future__ import annotations

import re

from exo_collection.adapters.encoder.teensy_serial import TEENSY_PID, TEENSY_VID


def scan_motor_ports() -> dict:
    """Enumerate USB/serial metadata; identify Teensy without opening ports."""
    report = {"ports": [], "candidates": [], "excluded": [], "error": None}
    try:
        from serial.tools import list_ports
        inventory = list(list_ports.comports())
    except ImportError as exc:
        report["error"] = f"串口扫描不可用：未安装 pyserial（{exc}）"
        return report
    except Exception as exc:
        report["error"] = f"串口扫描失败：{type(exc).__name__}: {exc}"
        return report

    seen = set()
    for port in inventory:
        device = str(getattr(port, "device", "") or "").strip()
        if not device or device in seen:
            continue
        seen.add(device)
        description = str(getattr(port, "description", "") or device)
        hwid = str(getattr(port, "hwid", "") or "")
        identity = f"{description} {hwid}".upper()
        if any(token in identity for token in ("BTHENUM", "BTHMODEM", "BLUETOOTH", "蓝牙")):
            report["excluded"].append(device)
            continue
        vid, pid = getattr(port, "vid", None), getattr(port, "pid", None)
        report["ports"].append({
            "device": device, "description": description, "hwid": hwid,
            "vid": vid, "pid": pid, "is_teensy": vid == TEENSY_VID and pid == TEENSY_PID,
        })
    report["ports"].sort(key=lambda item: (
        not item["is_teensy"],
        [(0, int(part)) if part.isdigit() else (1, part.casefold())
         for part in re.split(r"(\d+)", item["device"]) if part],
    ))
    report["candidates"] = [item["device"] for item in report["ports"] if item["is_teensy"]]
    return report


__all__ = ["scan_motor_ports"]
